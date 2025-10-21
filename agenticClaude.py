from datetime import datetime
from zoneinfo import ZoneInfo
import anthropic
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_current_time(timezone):
    """Returns current time for the given time zone"""
    timezone = ZoneInfo(timezone)
    return datetime.now(timezone).strftime("%H:%M:%S")

# Use Anthropic's official SDK
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

messages = [
    {"role": "user", "content": "What time is it in New York right now?"}
]

print("Sending initial request to Claude...")
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    tools=[{
        "name": "get_current_time",
        "description": "Returns current time for the given time zone",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The IANA timezone name (e.g., 'America/New_York')"
                }
            },
            "required": ["timezone"]
        }
    }],
    messages=messages
)

print(f"\nClaude's response: {response}")
print(f"\nStop reason: {response.stop_reason}")

# Check if Claude wants to use a tool
if response.stop_reason == "tool_use":
    # Extract the tool use block
    tool_use = next(block for block in response.content if block.type == "tool_use")
    
    print(f"\nClaude wants to use tool: {tool_use.name}")
    print(f"With input: {tool_use.input}")
    
    # Call the actual function
    tool_result = get_current_time(tool_use.input["timezone"])
    print(f"Tool result: {tool_result}")
    
    # Send the tool result back to Claude
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": tool_result
        }]
    })
    
    # Get Claude's final response
    print("\nSending tool result back to Claude...")
    final_response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        tools=[{
            "name": "get_current_time",
            "description": "Returns current time for the given time zone",
            "input_schema": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The IANA timezone name (e.g., 'America/New_York')"
                    }
                },
                "required": ["timezone"]
            }
        }],
        messages=messages
    )
    
    print(f"\nClaude's final answer:")
    for block in final_response.content:
        if hasattr(block, "text"):
            print(block.text)