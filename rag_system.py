"""RAG (Retrieval-Augmented Generation) system for wiki content"""

import os
import logging
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from wiki_parser import WikiParser

logger = logging.getLogger(__name__)


class RAGSystem:
    """Manage vector database and retrieval for wiki content"""
    
    def __init__(self, 
                 db_path: str = "./chroma_db",
                 collection_name: str = "maplestory_wiki",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG system
        
        Args:
            db_path: Path to store ChromaDB data
            collection_name: Name of the collection in ChromaDB
            model_name: Sentence transformer model for embeddings
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        self.parser = WikiParser(chunk_size=500, chunk_overlap=50)
    
    def index_wiki_dump(self, xml_path: str, batch_size: int = 100):
        """
        Parse and index the wiki XML dump
        
        Args:
            xml_path: Path to the wiki XML file
            batch_size: Number of chunks to process in each batch
        """
        logger.info(f"Starting to index wiki dump: {xml_path}")
        
        documents = []
        metadatas = []
        ids = []
        total_chunks = 0
        
        for page in self.parser.parse_wiki_xml(xml_path):
            # Create chunks for the page
            chunks = self.parser.chunk_text(
                page['content'],
                metadata={
                    'title': page['title'],
                    'categories': '|'.join(page['categories'])
                }
            )
            
            for chunk in chunks:
                doc_id = f"{page['title']}_{chunk['chunk_index']}"
                documents.append(chunk['text'])
                metadatas.append({
                    'title': page['title'],
                    'chunk_index': chunk['chunk_index'],
                    'categories': page['categories']
                })
                ids.append(doc_id)
                
                # Process in batches
                if len(documents) >= batch_size:
                    self._add_to_collection(documents, metadatas, ids)
                    total_chunks += len(documents)
                    logger.info(f"Indexed {total_chunks} chunks")
                    documents = []
                    metadatas = []
                    ids = []
        
        # Process remaining documents
        if documents:
            self._add_to_collection(documents, metadatas, ids)
            total_chunks += len(documents)
        
        logger.info(f"Indexing complete. Total chunks indexed: {total_chunks}")
    
    def _add_to_collection(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to ChromaDB collection with embeddings"""
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search for relevant content
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'title': results['metadatas'][0][i].get('title', 'Unknown'),
                    'score': 1 - results['distances'][0][i] if results['distances'] else 0
                })
        
        return formatted_results
    
    def get_context_for_query(self, query: str, max_tokens: int = 2000) -> str:
        """
        Get context for a query to use with LLM
        
        Args:
            query: User query
            max_tokens: Approximate maximum tokens for context
            
        Returns:
            Formatted context string
        """
        results = self.search(query, n_results=5)
        
        if not results:
            return ""
        
        context_parts = []
        current_tokens = 0
        
        for result in results:
            # Rough token estimation (1 token ≈ 4 chars)
            result_tokens = len(result['content']) // 4
            
            if current_tokens + result_tokens > max_tokens:
                break
            
            context_parts.append(
                f"[Source: {result['title']}]\n{result['content']}\n"
            )
            current_tokens += result_tokens
        
        return "\n---\n".join(context_parts)
    
    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the indexed content"""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection_name,
            'db_path': self.db_path
        }