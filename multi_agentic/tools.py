import requests
import os
import json
from dotenv import load_dotenv
from tavily import TavilyClient
import pandas as pd

from inventory_utils import create_inventory_dataframe

# Session setup (optional)
session = requests.Session()
session.headers.update({
    "User-Agent": "LF-ADP-Agent/1.0 (mailto:your.email@example.com)"
})

load_dotenv()

# 🔧 TOOL IMPLEMENTATIONS

def tavily_search_tool(query: str, max_results: int = 5, include_images: bool = False) -> list[dict[str, str]]:
    
    params = {}
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables.")
    params['api_key'] = api_key

    #client = TavilyClient(api_key)

    api_base_url = os.getenv("DLAI_TAVILY_BASE_URL")
    if api_base_url:
        params['api_base_url'] = api_base_url

    client = TavilyClient(api_key=api_key, api_base_url=api_base_url)

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            include_images=include_images
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "url": r.get("url", "")
            })

        if include_images:
            for img_url in response.get("images", []):
                results.append({"image_url": img_url})

        return results

    except Exception as e:
        return [{"error": str(e)}]
    

def product_catalog_tool(max_items: int = 10) -> list[dict[str, str]]:
    inventory_df = create_inventory_dataframe()
    return inventory_df.head(max_items).to_dict(orient="records")


# 🧠 TOOL METADATA FOR LLM

def get_available_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "tavily_search_tool",
                "description": "Perform web search for sunglasses trends using Tavily.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "default": 5},
                        "include_images": {"type": "boolean", "default": False}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "product_catalog_tool",
                "description": "Get sunglasses products from internal inventory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_items": {"type": "integer", "default": 10}
                    }
                }
            }
        }
    ]


# 🔁 TOOL CALL DISPATCHER

def handle_tool_call(tool_call):
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    tools_map = {
        "tavily_search_tool": tavily_search_tool,
        "product_catalog_tool": product_catalog_tool,
    }

    return tools_map[function_name](**arguments)


def create_tool_response_message(tool_call, tool_result):
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": json.dumps(tool_result)
    }

# Add these to your existing tools.py file

# =========================
# Sunglasses Product Catalog (for Multi-Agent Workflow)
# =========================

SUNGLASSES_CATALOG = [
    {
        "product_id": "SG001",
        "name": "Classic Aviator Gold",
        "description": "Timeless gold-frame aviators with UV400 protection",
        "style": "aviator",
        "color": "gold",
        "price": 149.99,
        "stock": 45,
        "tags": ["classic", "retro", "luxury", "pilot"]
    },
    {
        "product_id": "SG002",
        "name": "Modern Cat-Eye Black",
        "description": "Contemporary black cat-eye frames with gradient lenses",
        "style": "cat-eye",
        "color": "black",
        "price": 129.99,
        "stock": 32,
        "tags": ["modern", "feminine", "bold", "vintage"]
    },
    {
        "product_id": "SG003",
        "name": "Sporty Wraparound Blue",
        "description": "Athletic wraparound design with polarized blue lenses",
        "style": "wraparound",
        "color": "blue",
        "price": 99.99,
        "stock": 58,
        "tags": ["sporty", "active", "outdoor", "performance"]
    },
    {
        "product_id": "SG004",
        "name": "Vintage Round Tortoise",
        "description": "Retro round frames in tortoise shell pattern",
        "style": "round",
        "color": "tortoise",
        "price": 139.99,
        "stock": 28,
        "tags": ["vintage", "retro", "hipster", "classic"]
    },
    {
        "product_id": "SG005",
        "name": "Oversized Square Rose Gold",
        "description": "Fashion-forward oversized square frames in rose gold",
        "style": "square",
        "color": "rose gold",
        "price": 169.99,
        "stock": 15,
        "tags": ["trendy", "luxury", "statement", "oversized"]
    },
    {
        "product_id": "SG006",
        "name": "Minimalist Wire Silver",
        "description": "Ultra-thin wire frames with silver finish",
        "style": "wire",
        "color": "silver",
        "price": 119.99,
        "stock": 41,
        "tags": ["minimalist", "modern", "lightweight", "subtle"]
    }
]


def product_catalog_tool() -> dict:
    """Returns the internal sunglasses product catalog."""
    return {
        "source": "Internal Product Catalog",
        "total_products": len(SUNGLASSES_CATALOG),
        "products": SUNGLASSES_CATALOG,
        "metadata": {
            "last_updated": "2025-01-15",
            "currency": "USD"
        }
    }


def get_available_tools_claude():
    """Get tool definitions for Claude format."""
    return [
        {
            "name": "tavily_search_tool",
            "description": "Search the web for fashion trends and market information.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "product_catalog_tool",
            "description": "Access internal sunglasses product catalog.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]


def handle_tool_call_claude(tool_name: str, tool_input: dict):
    """Handle tool calls for Claude."""
    if tool_name == "tavily_search_tool":
        return tavily_search_tool(**tool_input)
    elif tool_name == "product_catalog_tool":
        return product_catalog_tool()
    else:
        return {"error": f"Unknown tool: {tool_name}"}
