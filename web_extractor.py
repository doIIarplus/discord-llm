"""Web page content extractor for Discord LLM Bot"""

import re
import logging
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup

# Set up logging
logger = logging.getLogger("WebExtractor")

# Default headers to mimic a browser request
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text"""
    # More robust URL pattern that handles various URL formats including those with special characters
    url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[^\s\"]*)?)?'
    urls = re.findall(url_pattern, text)
    # Filter out any URLs that might have trailing punctuation or quotes
    urls = [url.rstrip('.,;!?"\'') for url in urls]
    logger.debug(f"Extracted URLs: {urls}")
    return urls

def fetch_webpage_content(url: str, timeout: int = 10) -> Tuple[str, str]:
    """
    Fetch and extract text content from a webpage
    
    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (title, content) where content is the extracted text
    """
    try:
        logger.info(f"Fetching webpage: {url}")
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.get_text().strip()
        
        # Extract text content
        content = soup.get_text()
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        logger.info(f"Successfully extracted content from {url} (length: {len(content)})")
        return title, content
    except Exception as e:
        logger.error(f"Error fetching webpage {url}: {e}")
        return "", f"Error fetching webpage: {str(e)}"

def extract_webpage_context(text: str) -> str:
    """
    Extract content from URLs in text and return formatted context
    
    Args:
        text: Input text that may contain URLs
        
    Returns:
        Formatted string with webpage content context
    """
    urls = extract_urls(text)
    
    if not urls:
        return ""
    
    context_parts = []
    
    for url in urls:
        title, content = fetch_webpage_content(url)
        
        # Limit content length to prevent overwhelming the context
        # max_content_length = 2000
        # if len(content) > max_content_length:
        #     content = content[:max_content_length] + "..."
        
        context_parts.append(f"Web Page Content (from {url}):\nTitle: {title}\nContent: {content}\n")
    
    return "\n".join(context_parts) if context_parts else ""