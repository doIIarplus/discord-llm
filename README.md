# discord-llm

A Discord bot that integrates with Ollama to provide LLM capabilities in Discord.

## Features

- Chat with an LLM directly in Discord
- Image recognition capabilities
- Code execution for programmatic tasks
- Image generation
- RAG (Retrieval-Augmented Generation) with wiki data
- **Web page content reading** - Automatically extracts and uses content from URLs in messages

## Web Page Reading

When a message contains URLs, the bot will automatically fetch and extract the text content from those web pages and include it as context when generating responses. This allows the bot to:

- Answer questions about web page content
- Summarize articles
- Extract information from online sources
- Provide context-aware responses based on web content

The feature works automatically - just include URLs in your messages to the bot.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your environment variables in a `.env` file:
   ```
   DISCORD_BOT_TOKEN=your_discord_bot_token
   OLLAMA_API_URL=http://localhost:11434/api/generate
   OLLAMA_MODEL=your_preferred_model
   ```

3. Run the bot:
   ```
   python bot.py
   ```

## Dependencies

- discord.py
- python-dotenv
- chromadb
- lxml
- sentence-transformers
- numpy
- nltk
- requests
- beautifulsoup4