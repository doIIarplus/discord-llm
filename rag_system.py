"""RAG (Retrieval-Augmented Generation) system for wiki content"""

import os
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from wiki_parser import WikiParser

logger = logging.getLogger(__name__)


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
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Using collection: {collection_name}")
        except AttributeError:
            try:
                self.collection = self.client.get_collection(collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name, metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {collection_name}")

        self.parser = WikiParser(chunk_size=500, chunk_overlap=50)

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

                md = {
                    "title": title,
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "word_count": int(chunk.get("word_count", 0)),
                    "categories": categories,
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
                    }
                )

        return formatted_results

    def get_context_for_query(self, query: str, max_tokens: int = 2000) -> str:
        results = self.search(query, n_results=20)
        if not results:
            return ""

        context_parts: List[str] = []
        current_tokens = 0

        for result in results:
            result_tokens = len(result["content"]) // 4
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
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
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
