# M4 Agentic AI - Component-level eval using Anthropic Claude
# Converted from OpenAI to use Anthropic's Claude API
# FIXED: Proper tool definitions for Anthropic

from datetime import datetime
import json
import re

from anthropic import Anthropic

# Import your research tools
import research_tools
import utils

import os
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# =========================
# Tool Definitions for Anthropic
# =========================

RESEARCH_TOOLS = [
    {
        "name": "arxiv_search",
        "description": "Search arXiv for academic papers on a given topic",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for arXiv papers"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "tavily_search",
        "description": "Search the web using Tavily for general information and news",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "wikipedia_search",
        "description": "Search Wikipedia for encyclopedic summaries on a topic",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The Wikipedia search query"
                }
            },
            "required": ["query"]
        }
    }
]


def process_tool_call(tool_name: str, tool_input: dict):
    """
    Execute a tool call and return the result.
    
    Args:
        tool_name (str): Name of the tool to call
        tool_input (dict): Input parameters for the tool
        
    Returns:
        str: Result from the tool
    """
    try:
        if tool_name == "arxiv_search":
            # Call your arxiv tool from research_tools
            result = research_tools.arxiv_search(**tool_input)
        elif tool_name == "tavily_search":
            # Call your tavily tool from research_tools
            result = research_tools.tavily_search(**tool_input)
        elif tool_name == "wikipedia_search":
            # Call your wikipedia tool from research_tools
            result = research_tools.wikipedia_search(**tool_input)
        else:
            result = f"Unknown tool: {tool_name}"
        
        return json.dumps(result) if not isinstance(result, str) else result
    
    except Exception as e:
        return f"Tool error ({tool_name}): {str(e)}"


# =========================
# Research Step – `find_references`
# =========================

def find_references(task: str, model: str = "claude-opus-4-1", return_messages: bool = False):
    """
    Perform a research task using external tools (arxiv, tavily, wikipedia).
    
    Uses Anthropic's Claude API with proper tool use handling.
    
    Args:
        task (str): The research question/task to perform
        model (str): Claude model to use (default: claude-opus-4-1)
        return_messages (bool): If True, return (content, messages) tuple
        
    Returns:
        str or tuple: Research result content, optionally with messages
    """

    prompt = f"""
You are a research assistant with access to:
- arxiv_search: Search for academic papers
- tavily_search: Search the web for general information
- wikipedia_search: Search Wikipedia for summaries

Task:
{task}

Today is {datetime.now().strftime('%Y-%m-%d')}.

Please use the available tools to research this topic thoroughly and provide a comprehensive answer with URLs and citations.
""".strip()

    messages = [{"role": "user", "content": prompt}]

    try:
        # Initialize conversation
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            tools=RESEARCH_TOOLS,
            messages=messages,
        )

        # Agentic loop to handle tool calls
        max_iterations = 5
        iteration = 0

        while response.stop_reason == "tool_use" and iteration < max_iterations:
            iteration += 1
            
            # Add assistant's response to messages
            messages.append({"role": "assistant", "content": response.content})
            
            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    
                    # Execute the tool
                    tool_result = process_tool_call(tool_name, tool_input)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result,
                    })
            
            # Add tool results to messages
            messages.append({"role": "user", "content": tool_results})
            
            # Get next response
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                tools=RESEARCH_TOOLS,
                messages=messages,
            )

        # Extract final text content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return (content, messages) if return_messages else content

    except Exception as e:
        return f"[Model Error: {str(e)}]"


# =========================
# Evaluation Step – Preferred Domains
# =========================

# List of preferred domains for Tavily results
TOP_DOMAINS = {
    # General reference / institutions / publishers
    "wikipedia.org", "nature.com", "science.org", "sciencemag.org", "cell.com",
    "mit.edu", "stanford.edu", "harvard.edu", "nasa.gov", "noaa.gov", "europa.eu",

    # CS/AI venues & indexes
    "arxiv.org", "acm.org", "ieee.org", "neurips.cc", "icml.cc", "openreview.net",

    # Other reputable outlets
    "elifesciences.org", "pnas.org", "jmlr.org", "springer.com", "sciencedirect.com",

    # Extra domains (case-specific additions)
    "pbs.org", "nova.edu", "nvcc.edu", "cccco.edu",

    # Well known programming sites
    "codecademy.com", "datacamp.com"
}


def evaluate_tavily_results(TOP_DOMAINS, raw: str, min_ratio=0.4):
    """
    Evaluate whether plain-text research results mostly come from preferred domains.

    This is an objective evaluation with explicit per-example ground truth:
    each URL is checked against the predefined list of preferred domains.

    Args:
        TOP_DOMAINS (set[str]): Set of preferred domains (e.g., 'arxiv.org', 'nature.com').
        raw (str): Plain text or Markdown containing URLs.
        min_ratio (float): Minimum preferred ratio required to pass (e.g., 0.4 = 40%).

    Returns:
        tuple[bool, str]: (flag, markdown_report)
            flag -> True if PASS, False if FAIL
            markdown_report -> Markdown-formatted summary of the evaluation
    """

    # Extract URLs from the text
    url_pattern = re.compile(r'https?://[^\s\]\)>\}]+', flags=re.IGNORECASE)
    urls = url_pattern.findall(raw)

    if not urls:
        return False, """### Evaluation — Tavily Preferred Domains
No URLs detected in the provided text. 
Please include links in your research results.
"""

    # Count preferred vs total
    total = len(urls)
    preferred_count = 0
    details = []

    for url in urls:
        try:
            domain = url.split("/")[2]
        except IndexError:
            domain = url
        
        preferred = any(td in domain for td in TOP_DOMAINS)
        if preferred:
            preferred_count += 1
        details.append(f"- {url} → {'✅ PREFERRED' if preferred else '❌ NOT PREFERRED'}")

    ratio = preferred_count / total if total > 0 else 0.0
    flag = ratio >= min_ratio

    # Markdown report
    report = f"""
### Evaluation — Tavily Preferred Domains
- Total results: {total}
- Preferred results: {preferred_count}
- Ratio: {ratio:.2%}
- Threshold: {min_ratio:.0%}
- Status: {"✅ PASS" if flag else "❌ FAIL"}

**Details:**
{chr(10).join(details)}
"""
    return flag, report


# =========================
# Example Usage
# =========================

if __name__ == "__main__":
    
    # Example 1: Basic research task
    print("=" * 60)
    print("EXAMPLE 1: Research on Black Hole Science")
    print("=" * 60)
    
    research_task = "Find 2 recent papers about recent developments in black hole science"
    research_result = find_references(research_task, model="claude-opus-4-1")
    
    print("\nResearch Results:")
    print(research_result)
    
    # Evaluate the results
    flag, report = evaluate_tavily_results(TOP_DOMAINS, research_result)
    print("\n" + report)
    
    # Example 2: Try it yourself with different topic
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Research Task")
    print("=" * 60)
    
    topic = "alien life"
    min_ratio = 0.4
    
    print(f"\nTopic: {topic}")
    print(f"Min Ratio: {min_ratio:.0%}")
    print(f"\nPreferred Domains: {sorted(list(TOP_DOMAINS))[:5]}... (and {len(TOP_DOMAINS) - 5} more)")
    
    research_task = f"Find 2–3 key papers and reliable overviews about {topic}."
    research_output = find_references(research_task, model="claude-opus-4-1")
    
    print("\nResearch Results:")
    print(research_output)
    
    flag, eval_md = evaluate_tavily_results(TOP_DOMAINS, research_output, min_ratio=min_ratio)
    print("\n" + eval_md)