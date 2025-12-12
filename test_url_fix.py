#!/usr/bin/env python3
"""Test script for URL extraction fix"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_extractor import extract_urls

def test_problematic_url():
    """Test the problematic URL that was being truncated"""
    test_text = "trying with 'https://www.whitehouse.gov/presidential-actions/2025/09/restriction-on-entry-of-certain-nonimmigrant-workers/'"
    urls = extract_urls(test_text)
    print(f"Input text: {test_text}")
    print(f"Extracted URLs: {urls}")
    
    expected_url = "https://www.whitehouse.gov/presidential-actions/2025/09/restriction-on-entry-of-certain-nonimmigrant-workers/"
    if urls and expected_url in urls[0]:
        print("URL extraction test PASSED!")
        return True
    else:
        print("URL extraction test FAILED!")
        return False

if __name__ == "__main__":
    print("Testing URL extraction fix...")
    success = test_problematic_url()
    if success:
        print("Fix successful!")
    else:
        print("Fix needs more work.")