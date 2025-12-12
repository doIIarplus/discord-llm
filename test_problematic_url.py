#!/usr/bin/env python3
"""Test script for the specific problematic URL"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_extractor import extract_urls, fetch_webpage_content

def test_problematic_url():
    """Test the specific problematic URL"""
    # Test with the URL that was having issues
    test_text = "trying with 'https://www.whitehouse.gov/presidential-actions/2025/09/restriction-on-entry-of-certain-nonimmigrant-workers/'"
    print(f"Testing with text: {test_text}")
    
    # Extract URLs
    urls = extract_urls(test_text)
    print(f"Extracted URLs: {urls}")
    
    if not urls:
        print("No URLs found!")
        return False
    
    # Check if the URL is correctly extracted
    expected_url = "https://www.whitehouse.gov/presidential-actions/2025/09/restriction-on-entry-of-certain-nonimmigrant-workers/"
    if urls[0] != expected_url:
        print(f"URL mismatch! Expected: {expected_url}, Got: {urls[0]}")
        return False
    
    print("URL extraction test PASSED!")
    return True

if __name__ == "__main__":
    print("Testing specific problematic URL...")
    success = test_problematic_url()
    if success:
        print("Problematic URL test PASSED!")
    else:
        print("Problematic URL test FAILED!")