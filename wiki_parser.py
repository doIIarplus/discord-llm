"""Wiki XML parser for extracting and chunking content"""

import re
from typing import List, Dict, Iterator
from lxml import etree
import logging

logger = logging.getLogger(__name__)


class WikiParser:
    """Parse MediaWiki XML dumps and extract content"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def parse_wiki_xml(self, xml_path: str) -> Iterator[Dict[str, str]]:
        """
        Stream-parse wiki XML file and yield page content
        
        Args:
            xml_path: Path to the XML dump file
            
        Yields:
            Dict with 'title', 'content', and 'categories'
        """
        logger.info(f"Starting to parse wiki XML: {xml_path}")
        
        # Use iterparse for memory-efficient parsing of large XML
        context = etree.iterparse(xml_path, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        namespace = None
        page_count = 0
        
        for event, elem in context:
            # Extract namespace from root element
            if event == 'start' and namespace is None:
                namespace = elem.nsmap.get(None, '')
            
            # Process complete page elements
            if event == 'end' and elem.tag.endswith('page'):
                page_data = self._extract_page_data(elem, namespace)
                if page_data and page_data['content']:
                    page_count += 1
                    if page_count % 100 == 0:
                        logger.info(f"Processed {page_count} pages")
                    yield page_data
                
                # Clear the element to free memory
                elem.clear()
                root.clear()
        
        logger.info(f"Finished parsing. Total pages processed: {page_count}")
    
    def _extract_page_data(self, page_elem, namespace: str) -> Dict[str, str]:
        """Extract title and content from a page element"""
        ns = {'': namespace} if namespace else {}
        
        # Extract title
        title_elem = page_elem.find('.//title', ns)
        title = title_elem.text if title_elem is not None else ''
        
        # Skip special pages and templates
        if title.startswith(('MediaWiki:', 'Template:', 'Category:', 'File:')):
            return {}
        
        # Extract revision text (wiki markup)
        text_elem = page_elem.find('.//revision/text', ns)
        if text_elem is None:
            text_elem = page_elem.find('.//text', ns)
        
        content = text_elem.text if text_elem is not None and text_elem.text else ''
        
        # Clean wiki markup
        content = self._clean_wiki_markup(content)
        
        # Extract categories
        categories = self._extract_categories(content)
        
        return {
            'title': title,
            'content': content,
            'categories': categories
        }
    
    def _clean_wiki_markup(self, text: str) -> str:
        """Remove wiki markup and clean text"""
        if not text:
            return ''
        
        # Remove templates and infoboxes
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        
        # Remove references
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*/?>', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove wiki links but keep text
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
        
        # Remove external links
        text = re.sub(r'\[https?://[^\s\]]+\s*([^\]]*)\]', r'\1', text)
        
        # Remove formatting
        text = re.sub(r"'''?", '', text)
        text = re.sub(r'={2,}([^=]+)={2,}', r'\1', text)  # Headers
        
        # Remove file/image references
        text = re.sub(r'\[\[File:[^\]]+\]\]', '', text)
        text = re.sub(r'\[\[Image:[^\]]+\]\]', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _extract_categories(self, content: str) -> List[str]:
        """Extract category names from content"""
        categories = re.findall(r'\[\[Category:([^\]]+)\]\]', content)
        return categories
    
    def chunk_text(self, text: str, metadata: Dict[str, str] = None) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            metadata: Additional metadata to include with each chunk
            
        Returns:
            List of chunks with metadata
        """
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk_data = {
                'text': chunk_text,
                'chunk_index': len(chunks),
                'word_count': len(chunk_words)
            }
            
            if metadata:
                chunk_data.update(metadata)
            
            chunks.append(chunk_data)
            
            # Stop if we've reached the end
            if i + self.chunk_size >= len(words):
                break
        
        return chunks