import json
import os
from datetime import datetime
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ============================================================================
# SECTION 3: Build your first tool
# ============================================================================

def get_current_time():
    """
    Returns the current time as a string.
    """
    return datetime.now().strftime("%H:%M:%S")


# Test the function
print("Testing get_current_time():")
print(get_current_time())
print()

# ============================================================================
# SECTION 3.2: Using the tool with Claude
# ============================================================================

def call_claude_with_tools(prompt, tools_list, model="claude-sonnet-4-20250514", max_iterations=5):
    """
    Call Claude with tools and handle tool use automatically.
    
    Args:
        prompt: The user's question/request
        tools_list: List of tool definitions for Claude
        model: Claude model to use
        max_iterations: Maximum number of back-and-forth exchanges
    
    Returns:
        The final response from Claude and the full conversation history
    """
    messages = [{"role": "user", "content": prompt}]
    iteration = 0
    
    print(f"User: {prompt}\n")
    
    while iteration < max_iterations:
        iteration += 1
        
        # Call Claude
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            tools=tools_list,
            messages=messages
        )
        
        # Check if Claude wants to use a tool
        if response.stop_reason == "tool_use":
            # Extract tool use
            tool_use_block = None
            for block in response.content:
                if block.type == "tool_use":
                    tool_use_block = block
                    break
            
            if tool_use_block:
                tool_name = tool_use_block.name
                tool_input = tool_use_block.input
                
                print(f"🔧 Claude wants to use tool: {tool_name}")
                print(f"   With input: {tool_input}")
                
                # Execute the tool locally
                tool_result = execute_tool(tool_name, tool_input)
                print(f"   Tool result: {tool_result}\n")
                
                # Add assistant's response to messages
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                # Add tool result to messages
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": str(tool_result)
                        }
                    ]
                })
        else:
            # No more tool use, extract final response
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            
            print(f"Claude: {final_text}\n")
            return final_text, messages
    
    return "Max iterations reached", messages


def execute_tool(tool_name, tool_input):
    """
    Execute a tool function based on its name and input parameters.
    """
    if tool_name == "get_current_time":
        return get_current_time()
    elif tool_name == "get_weather_from_ip":
        return get_weather_from_ip()
    elif tool_name == "write_txt_file":
        return write_txt_file(tool_input["filename"], tool_input["content"])
    elif tool_name == "generate_qr_code":
        return generate_qr_code(
            tool_input["data"],
            tool_input["filename"],
            tool_input["image_path"]
        )
    else:
        return f"Unknown tool: {tool_name}"


# Define tool schema for get_current_time
time_tool = {
    "name": "get_current_time",
    "description": "Returns the current time as a string in HH:MM:SS format.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

# Test with Claude
print("=" * 80)
print("EXAMPLE 1: Getting the current time")
print("=" * 80)
response_text, conversation = call_claude_with_tools(
    "What time is it?",
    [time_tool]
)

# ============================================================================
# SECTION 4: Additional Tools
# ============================================================================

def get_weather_from_ip():
    """Get weather information based on IP address location."""
    import requests
    try:
        # Get location from IP
        ip_response = requests.get('https://ipapi.co/json/')
        location_data = ip_response.json()
        city = location_data.get('city', 'Unknown')
        
        # Get weather (using wttr.in as a simple weather API)
        weather_response = requests.get(f'https://wttr.in/{city}?format=%C+%t')
        weather_info = weather_response.text.strip()
        
        return f"Weather in {city}: {weather_info}"
    except Exception as e:
        return f"Could not fetch weather: {str(e)}"


def write_txt_file(filename, content):
    """Write content to a text file.
    
    Args:
        filename: Name of the file to create (should end in .txt)
        content: Text content to write to the file
    """
    # Ensure filename ends with .txt
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    with open(filename, 'w') as f:
        f.write(content)
    
    return f"Successfully wrote content to {filename}"


def generate_qr_code(data, filename, image_path):
    """Generate a QR code image given data and an image path.
    
    Args:
        data: Text or URL to encode
        filename: Name for the output PNG file (without extension)
        image_path: Path to the image to be used in the QR code
    """
    import qrcode
    from qrcode.image.styledpil import StyledPilImage
    
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)
    
    img = qr.make_image(image_factory=StyledPilImage, embedded_image_path=image_path)
    output_file = f"{filename}.png"
    img.save(output_file)
    
    return f"QR code saved as {output_file} containing: {data[:50]}..."


# Define all tool schemas for Claude
all_tools = [
    {
        "name": "get_current_time",
        "description": "Returns the current time as a string in HH:MM:SS format.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_weather_from_ip",
        "description": "Get weather information for the user's current location based on their IP address.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "write_txt_file",
        "description": "Write text content to a file. Creates a new text file with the specified content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the text file to create (e.g., 'notes.txt')"
                },
                "content": {
                    "type": "string",
                    "description": "The text content to write to the file"
                }
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "generate_qr_code",
        "description": "Generate a QR code image with an embedded logo/image. Creates a PNG file with the QR code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "The text or URL to encode in the QR code"
                },
                "filename": {
                    "type": "string",
                    "description": "Name for the output file (without .png extension)"
                },
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file to embed in the QR code"
                }
            },
            "required": ["data", "filename", "image_path"]
        }
    }
]

# ============================================================================
# EXAMPLES: Using the tools
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: Getting weather")
print("=" * 80)
response_text, _ = call_claude_with_tools(
    "Can you get the weather for my location?",
    all_tools
)

print("\n" + "=" * 80)
print("EXAMPLE 3: Writing a text file")
print("=" * 80)
response_text, _ = call_claude_with_tools(
    "Can you make a txt note for me called reminders.txt that reminds me to call Daniel tomorrow at 7PM?",
    all_tools
)

# Verify the file was created
try:
    with open('reminders.txt', 'r') as file:
        contents = file.read()
        print(f"\n📄 Contents of reminders.txt:\n{contents}")
except FileNotFoundError:
    print("\n⚠️ File was not created")

print("\n" + "=" * 80)
print("EXAMPLE 4: Generating QR code")
print("=" * 80)
print("Note: This example requires 'dl_logo.jpg' to exist in the current directory")
print("If you don't have this file, the tool will fail but Claude will handle it gracefully")

response_text, _ = call_claude_with_tools(
    "Can you make a QR code for me using my company's logo that goes to www.deeplearning.ai? The logo is located at `dl_logo.jpg`. You can call it dl_qr_code.",
    all_tools
)

print("\n" + "=" * 80)
print("EXAMPLE 5: Using multiple tools")
print("=" * 80)
response_text, _ = call_claude_with_tools(
    "Can you help me create a qr code that goes to www.deeplearning.com from the image dl_logo.jpg? Also write me a txt note with the current weather please.",
    all_tools,
    max_iterations=10
)

print("\n" + "=" * 80)
print("✅ Lab complete! All examples executed.")
print("=" * 80)