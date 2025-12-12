#!/usr/bin/env python3
"""Test script for web extractor functionality"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_extractor import extract_urls, fetch_webpage_content, extract_webpage_context

def test_url_extraction():
    """Test URL extraction from text"""
    test_text = "Check out this website: https://www.python.org and also https://github.com"
    urls = extract_urls(test_text)
    print(f"Extracted URLs: {urls}")
    assert len(urls) == 2
    assert "https://www.python.org" in urls
    assert "https://github.com" in urls
    print("URL extraction test passed!")

def test_problematic_url():
    """Test the previously problematic URL"""
    test_text = "trying with 'https://www.whitehouse.gov/presidential-actions/2025/09/restriction-on-entry-of-certain-nonimmigrant-workers/'"
    urls = extract_urls(test_text)
    print(f"Extracted URLs: {urls}")
    assert len(urls) == 1
    assert urls[0] == "https://www.whitehouse.gov/presidential-actions/2025/09/restriction-on-entry-of-certain-nonimmigrant-workers/"
    print("Problematic URL test passed!")

def test_webpage_fetching():
    """Test fetching webpage content"""
    title, content = fetch_webpage_content("https://httpbin.org/html")
    print(f"Title: {title}")
    print(f"Content length: {len(content)}")
    assert len(content) > 0
    print("Webpage fetching test passed!")

def test_webpage_context_extraction():
    """Test extracting webpage context"""
    test_text = "Check out this website: https://httpbin.org/html"
    context = extract_webpage_context(test_text)
    print(f"Context: {context}")
    assert len(context) > 0
    print("Webpage context extraction test passed!")

if __name__ == "__main__":
    print("Running web extractor tests...")
    test_url_extraction()
    test_problematic_url()
    test_webpage_fetching()
    test_webpage_context_extraction()
    print("All tests passed!")