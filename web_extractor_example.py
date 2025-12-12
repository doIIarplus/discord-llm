"""Example usage of the web page reading feature"""

# Example 1: Basic URL detection and content extraction
message_with_url = "Can you summarize the content of this page? https://example.com"

# The bot will automatically extract content from the URL and use it for context

# Example 2: Multiple URLs
message_with_multiple_urls = """
Please compare the content of these two pages:
1. https://example.com/page1
2. https://example.com/page2
"""

# Example 3: URL with question
message_with_url_and_question = """
What does this article say about climate change?
https://example.com/climate-article
"""

# The web extractor will:
# 1. Detect all URLs in the message
# 2. Fetch content from each URL
# 3. Extract text content and title
# 4. Add the content to the conversation context
# 5. The LLM will use this context to generate informed responses