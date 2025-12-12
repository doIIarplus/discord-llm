#!/usr/bin/env python3
"""Test script for complete web extractor functionality"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_extractor import extract_urls, fetch_webpage_content, extract_webpage_context

def test_complete_functionality():
    """Test the complete web extractor functionality"""
    # Test with a real URL that should work
    test_text = "Check out this page: https://httpbin.org/html"
    print(f"Testing with text: {test_text}")
    
    # Extract URLs
    urls = extract_urls(test_text)
    print(f"Extracted URLs: {urls}")
    
    if not urls:
        print("No URLs found!")
        return False
    
    # Fetch content from the first URL
    url = urls[0]
    print(f"Fetching content from: {url}")
    title, content = fetch_webpage_content(url)
    print(f"Title: {title}")
    print(f"Content length: {len(content)}")
    
    if len(content) == 0:
        print("Failed to fetch content!")
        return False
    
    # Test context extraction
    context = extract_webpage_context(test_text)
    print(f"Context length: {len(context)}")
    
    if len(context) == 0:
        print("Failed to extract context!")
        return False
    
    print("All tests passed!")
    return True

if __name__ == "__main__":
    print("Testing complete web extractor functionality...")
    success = test_complete_functionality()
    if success:
        print("Complete functionality test PASSED!")
    else:
        print("Complete functionality test FAILED!")