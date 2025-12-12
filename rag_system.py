"""RAG (Retrieval-Augmented Generation) system for wiki content"""

import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from functools import lru_cache
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from wiki_parser import WikiParser

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    logger.warning("transformers not available. Falling back to word counting for token estimation.")

def _sanitize_metadata(metadatas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure metadata only contains str, int, float, or bool.
    - None values are removed.
    - Lists/dicts are converted to strings.
    """
    safe_list = []
    for m in metadatas:
        safe_m = {}
        for k, v in m.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                safe_m[k] = v
            elif isinstance(v, list):
                safe_m[k] = "|".join(map(str, v))  # join lists into string
            else:
                safe_m[k] = str(v)
        safe_list.append(safe_m)
    return safe_list


class RAGSystem:
    """Manage vector database and retrieval for wiki content"""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "maplestory_wiki",
        model_name: str = "all-mpnet-base-v2",  # Upgraded model
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize tokenizer for better token counting if available
        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}. Falling back to word counting.")
                self.tokenizer = None
        else:
            self.tokenizer = None

        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 128,  # higher = more accurate but slower
                    "hnsw:M": 16,  # connectivity parameter
                },
            )
            logger.info(f"Using collection: {collection_name}")
        except AttributeError:
            try:
                self.collection = self.client.get_collection(collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name, 
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": 128,
                        "hnsw:M": 16,
                    }
                )
                logger.info(f"Created new collection: {collection_name}")

        self.parser = WikiParser(chunk_size=500, chunk_overlap=100)  # Increased overlap for better context

    def index_wiki_dump(self, xml_path: str, batch_size: int = 100):
        logger.info(f"Starting to index wiki dump: {xml_path}")

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        total_chunks = 0

        for page in self.parser.parse_wiki_xml(xml_path):
            title = str(page.get("title") or "Untitled")
            categories = page.get("categories") or ""
            content = page.get("content") or ""

            chunks = self.parser.chunk_text(
                content,
                metadata={"title": title, "categories": categories},
            )

            for chunk in chunks:
                doc_id = f"{title}_{chunk['chunk_index']}"
                documents.append(chunk["text"])

                # Enhanced metadata
                md = {
                    "title": title,
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "word_count": int(chunk.get("word_count", 0)),
                    "categories": categories,
                    "content_length": len(chunk["text"]),  # character count
                    "timestamp": datetime.now().isoformat(),  # indexing time
                    "doc_length": len(content.split()),  # total words in original doc
                    "relative_position": chunk.get("chunk_index", 0) / max(1, len(chunks)-1)  # position in doc
                }
                metadatas.append(md)
                ids.append(doc_id)

                if len(documents) >= batch_size:
                    self._add_to_collection(documents, metadatas, ids)
                    total_chunks += len(documents)
                    logger.info(f"Indexed {total_chunks} chunks")
                    documents.clear()
                    metadatas.clear()
                    ids.clear()

        if documents:
            self._add_to_collection(documents, metadatas, ids)
            total_chunks += len(documents)

        logger.info(f"Indexing complete. Total chunks indexed: {total_chunks}")

    def _add_to_collection(
        self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]
    ):
        if not documents:
            return

        embeddings = self.embedding_model.encode(documents, show_progress_bar=False)
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=_sanitize_metadata(metadatas),
            ids=ids,
        )

    @lru_cache(maxsize=1000)
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query])
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: List[Dict[str, Any]] = []
        docs = results.get("documents", [[]])
        metas = results.get("metadatas", [[]])
        dists = results.get("distances", [[]])

        if docs and docs[0]:
            for i, doc in enumerate(docs[0]):
                meta = metas[0][i] if i < len(metas[0]) else {}
                dist = dists[0][i] if (dists and dists[0] and i < len(dists[0])) else None
                score = (1 - dist) if dist is not None else None
                formatted_results.append(
                    {
                        "content": doc,
                        "title": meta.get("title", "Unknown"),
                        "categories": meta.get("categories"),
                        "chunk_index": meta.get("chunk_index"),
                        "score": score,
                        "word_count": meta.get("word_count"),
                        "content_length": meta.get("content_length"),
                        "relative_position": meta.get("relative_position"),
                    }
                )

        return formatted_results

    def get_context_for_query(self, query: str, max_tokens: int = 2000) -> str:
        results = self.search(query, n_results=20)
        if not results:
            return ""

        context_parts: List[str] = []
        current_tokens = 0

        # Sort results by score (relevance) and then by document coherence
        results.sort(key=lambda x: (x.get("score", 0), -abs(x.get("relative_position", 0.5) - 0.5)), reverse=True)
        
        # Group results by title to prioritize chunks from the same document
        grouped_results = {}
        for result in results:
            title = result.get("title", "Unknown")
            if title not in grouped_results:
                grouped_results[title] = []
            grouped_results[title].append(result)
        
        # Flatten grouped results, prioritizing documents with more relevant chunks
        flattened_results = []
        for title, chunks in grouped_results.items():
            # Sort chunks by position to maintain reading order
            chunks.sort(key=lambda x: x.get("relative_position", 0))
            flattened_results.extend(chunks)

        for result in flattened_results:
            # Use actual tokenizer for more accurate token counting if available
            if self.tokenizer:
                try:
                    result_tokens = len(self.tokenizer.encode(result["content"]))
                except Exception:
                    # Fallback to word counting if tokenizer fails
                    result_tokens = len(result["content"].split())
            else:
                # Fallback to word counting
                result_tokens = len(result["content"].split())
                
            if current_tokens + result_tokens > max_tokens:
                break

            header = f"[Source: {result.get('title','Unknown')}"
            if result.get("chunk_index") is not None:
                header += f" | chunk {result['chunk_index']}"
            header += "]"

            context_parts.append(f"{header}\n{result['content']}\n")
            current_tokens += result_tokens

        return "\n---\n".join(context_parts)

    def clear_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name, 
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 128,
                    "hnsw:M": 16,
                }
            )
            logger.info("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "db_path": self.db_path,
        }
        
    def evaluate_retrieval(self, queries_with_ground_truth: List[Dict]) -> Dict:
        """
        Evaluate the RAG system performance
        
        Args:
            queries_with_ground_truth: List of dicts with 'query' and 'relevant_doc_ids'
            
        Returns:
            Dict with evaluation metrics
        """
        total_precision = 0
        total_recall = 0
        total_reciprocal_rank = 0
        query_count = len(queries_with_ground_truth)
        
        for item in queries_with_ground_truth:
            query = item['query']
            relevant_ids = set(item['relevant_doc_ids'])
            
            # Get top 10 results
            results = self.search(query, n_results=10)
            retrieved_ids = set([f"{r['title']}_{r['chunk_index']}" for r in results])
            
            # Calculate precision and recall
            if retrieved_ids:
                precision = len(retrieved_ids.intersection(relevant_ids)) / len(retrieved_ids)
                recall = len(retrieved_ids.intersection(relevant_ids)) / len(relevant_ids) if relevant_ids else 0
            else:
                precision = 0
                recall = 0
                
            # Calculate reciprocal rank
            reciprocal_rank = 0
            for i, result in enumerate(results):
                result_id = f"{result['title']}_{result['chunk_index']}"
                if result_id in relevant_ids:
                    reciprocal_rank = 1 / (i + 1)
                    break
                    
            total_precision += precision
            total_recall += recall
            total_reciprocal_rank += reciprocal_rank
            
        # Calculate averages
        avg_precision = total_precision / query_count if query_count > 0 else 0
        avg_recall = total_recall / query_count if query_count > 0 else 0
        mean_reciprocal_rank = total_reciprocal_rank / query_count if query_count > 0 else 0
        
        # Calculate F1 score
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": f1_score,
            "mrr": mean_reciprocal_rank,
            "query_count": query_count
        }
