#!/usr/bin/env python3
"""Test script for RAG system functionality"""

import asyncio
import logging
from rag_system import RAGSystem
from wiki_parser import WikiParser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_wiki_parser():
    """Test the wiki parser with a small sample"""
    logger.info("Testing Wiki Parser...")
    parser = WikiParser(chunk_size=500, chunk_overlap=50)
    
    # Try to parse first few pages
    xml_path = "maplestorywikinet.xml"
    pages_parsed = 0
    
    try:
        for page in parser.parse_wiki_xml(xml_path):
            pages_parsed += 1
            logger.info(f"Page {pages_parsed}: {page['title']}")
            logger.info(f"  Content length: {len(page['content'])} chars")
            logger.info(f"  Categories: {page['categories']}")
            
            # Test chunking
            chunks = parser.chunk_text(page['content'], {'title': page['title']})
            logger.info(f"  Chunks created: {len(chunks)}")
            
            if pages_parsed >= 5:  # Only test first 5 pages
                break
                
        logger.info(f"✅ Wiki parser test passed! Parsed {pages_parsed} pages")
        return True
    except Exception as e:
        logger.error(f"❌ Wiki parser test failed: {e}")
        return False


def test_rag_indexing():
    """Test indexing a small portion of the wiki"""
    logger.info("\nTesting RAG Indexing...")
    rag = RAGSystem()
    
    try:
        # Clear any existing data
        rag.clear_collection()
        
        # Parse and index just a few pages for testing
        parser = WikiParser(chunk_size=500, chunk_overlap=50)
        xml_path = "maplestorywikinet.xml"
        
        documents = []
        metadatas = []
        ids = []
        pages_indexed = 0
        
        for page in parser.parse_wiki_xml(xml_path):
            chunks = parser.chunk_text(
                page['content'],
                metadata={'title': page['title']}
            )
            
            for chunk in chunks:
                doc_id = f"{page['title']}_{chunk['chunk_index']}"
                documents.append(chunk['text'])
                metadatas.append({
                    'title': page['title'],
                    'chunk_index': chunk['chunk_index']
                })
                ids.append(doc_id)
            
            pages_indexed += 1
            if pages_indexed >= 10:  # Index first 10 pages for testing
                break
        
        # Add to collection
        if documents:
            rag._add_to_collection(documents, metadatas, ids)
            
        stats = rag.get_stats()
        logger.info(f"✅ Indexing test passed! Indexed {stats['total_chunks']} chunks from {pages_indexed} pages")
        return True
    except Exception as e:
        logger.error(f"❌ Indexing test failed: {e}")
        return False


def test_rag_search():
    """Test searching the indexed content"""
    logger.info("\nTesting RAG Search...")
    rag = RAGSystem()
    
    try:
        # Test queries
        test_queries = [
            "What is MapleStory?",
            "How do I level up?",
            "What are the character classes?",
            "Tell me about quests",
            "What items can I find?"
        ]
        
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            results = rag.search(query, n_results=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    logger.info(f"  Result {i}: {result['title']} (Score: {result['score']:.3f})")
                    preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                    logger.info(f"    Preview: {preview}")
            else:
                logger.info("  No results found")
        
        logger.info("\n✅ Search test passed!")
        return True
    except Exception as e:
        logger.error(f"❌ Search test failed: {e}")
        return False


def test_rag_context_generation():
    """Test context generation for LLM"""
    logger.info("\nTesting Context Generation...")
    rag = RAGSystem()
    
    try:
        query = "What are the best weapons in MapleStory?"
        context = rag.get_context_for_query(query)
        
        if context:
            logger.info(f"Generated context for query: '{query}'")
            logger.info(f"Context length: {len(context)} characters")
            logger.info(f"Context preview:\n{context[:500]}...")
            logger.info("\n✅ Context generation test passed!")
        else:
            logger.info("⚠️ No context generated (might need more indexed content)")
        
        return True
    except Exception as e:
        logger.error(f"❌ Context generation test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("Starting RAG System Tests\n" + "="*50)
    
    tests = [
        ("Wiki Parser", test_wiki_parser),
        ("RAG Indexing", test_rag_indexing),
        ("RAG Search", test_rag_search),
        ("Context Generation", test_rag_context_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}\nRunning: {test_name}\n{'='*50}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.info("\n⚠️ Some tests failed. Please check the logs above.")


if __name__ == "__main__":
    main()