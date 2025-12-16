"""Response splitting utilities for natural multi-message output"""

import re
from typing import List

from config import MAX_DISCORD_MESSAGE_LENGTH

# Marker the LLM uses to indicate message breaks
MESSAGE_SPLIT_MARKER = "---MSG---"


def split_response_by_markers(text: str) -> List[str]:
    """
    Split response text by explicit markers placed by the LLM.

    Args:
        text: Raw response text potentially containing split markers

    Returns:
        List of individual message strings
    """
    # Remove thinking tags first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Split by the marker
    parts = text.split(MESSAGE_SPLIT_MARKER)

    # Clean up each part
    messages = []
    for part in parts:
        cleaned = part.strip()
        if cleaned:
            # Further split if any part exceeds Discord's limit
            if len(cleaned) > MAX_DISCORD_MESSAGE_LENGTH:
                messages.extend(split_long_message(cleaned))
            else:
                messages.append(cleaned)

    return messages if messages else [text.strip()]


def split_long_message(text: str, limit: int = MAX_DISCORD_MESSAGE_LENGTH) -> List[str]:
    """
    Split a long message at natural breakpoints (sentences, paragraphs).

    Args:
        text: The message to split
        limit: Maximum length per message

    Returns:
        List of message chunks
    """
    if len(text) <= limit:
        return [text]

    messages = []
    current = ""

    # Try to split by paragraphs first
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        if len(current) + len(para) + 2 <= limit:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                messages.append(current.strip())

            # If single paragraph is too long, split by sentences
            if len(para) > limit:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sentence in sentences:
                    if len(current) + len(sentence) + 1 <= limit:
                        current = current + " " + sentence if current else sentence
                    else:
                        if current:
                            messages.append(current.strip())
                        # If single sentence is too long, hard split
                        if len(sentence) > limit:
                            for i in range(0, len(sentence), limit):
                                messages.append(sentence[i:i+limit])
                            current = ""
                        else:
                            current = sentence
            else:
                current = para

    if current:
        messages.append(current.strip())

    return messages


def calculate_typing_delay(message: str, wpm: int = 80) -> float:
    """
    Calculate a realistic typing delay based on message length.

    Args:
        message: The message being "typed"
        wpm: Words per minute typing speed (default 80 for a fast typer)

    Returns:
        Delay in seconds (capped between 0.5 and 8 seconds)
    """
    word_count = len(message.split())
    # Calculate time in seconds: (words / wpm) * 60 seconds
    delay = (word_count / wpm) * 60

    # Cap between reasonable bounds
    return max(0.5, min(8.0, delay))
