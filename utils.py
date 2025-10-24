"""
utils.py
Helper utility functions for the agentic AI project
"""

import json
from typing import Any


def pretty_print_messages(messages: list):
    """
    Pretty print conversation messages.
    
    Args:
        messages: List of message dictionaries
    """
    print("\n" + "=" * 80)
    print("CONVERSATION HISTORY")
    print("=" * 80)
    
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        print(f"\n[{i}] Role: {role.upper()}")
        print("-" * 80)
        
        if isinstance(content, str):
            print(content[:500] + ("..." if len(content) > 500 else ""))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    print(json.dumps(item, indent=2)[:300])
        else:
            print(str(content)[:500])
    
    print("\n" + "=" * 80)


def format_json(data: Any, indent: int = 2) -> str:
    """
    Format data as pretty JSON string.
    
    Args:
        data: Data to format
        indent: Indentation level
        
    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def extract_urls(text: str) -> list[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Text containing URLs
        
    Returns:
        List of URLs
    """
    import re
    url_pattern = re.compile(r'https?://[^\s\]\)>\}]+', flags=re.IGNORECASE)
    return url_pattern.findall(text)


def count_tokens_estimate(text: str) -> int:
    """
    Rough estimate of token count (words * 1.3).
    
    Args:
        text: Text to estimate
        
    Returns:
        Estimated token count
    """
    words = len(text.split())
    return int(words * 1.3)