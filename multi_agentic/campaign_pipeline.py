"""
M5 Agentic AI - Market Research Team (Multi-Agent Workflow)
Converted from OpenAI/AISuite to Anthropic Claude API

A fully automated creative pipeline for a summer sunglasses campaign featuring:
- Market Research Agent: Scans trends and matches products
- Graphic Designer Agent: Creates visual concepts and captions  
- Copywriter Agent: Generates marketing quotes from images
- Packaging Agent: Assembles executive-ready reports

This demonstrates multi-agent coordination, tool-calling, and multimodal AI.
"""

# =========================
# Imports
# =========================

# --- Standard library ---
import base64
import json
import os
import re
from datetime import datetime
from io import BytesIO

# --- Third-party ---
import requests
from PIL import Image
from dotenv import load_dotenv
import anthropic

# --- Local / project (you'll need to create these) ---
import tools_multi_agent as tools
import utils_multi_agent as utils


# =========================
# Environment & Client
# =========================
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = None  # For DALL-E image generation only

# Initialize OpenAI client for DALL-E (image generation)
try:
    import openai
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    print("⚠️ OpenAI not installed. Image generation will be unavailable.")


# =========================
# AGENT 1: Market Research Agent
# =========================

def market_research_agent(model: str = "claude-sonnet-4-20250514", return_messages: bool = False):
    """
    Fashion market research agent that:
    1. Explores current fashion trends using web search
    2. Reviews internal product catalog
    3. Recommends products that match trends
    
    Args:
        model: Claude model to use
        return_messages: If True, return (content, messages) tuple
        
    Returns:
        str or tuple: Trend analysis and product recommendations
    """
    
    utils.log_agent_title_html("Market Research Agent", "🕵️‍♂️")
    
    system_prompt = """You are a fashion market research agent tasked with preparing a trend analysis for a summer sunglasses campaign.

Your goal:
1. Explore current fashion trends related to sunglasses using web search.
2. Review the internal product catalog to identify items that align with those trends.
3. Recommend one or more products from the catalog that best match emerging trends.

Once your analysis is complete, summarize:
- The top 2–3 trends you found.
- The product(s) from the catalog that fit these trends.
- A justification of why they are a good fit for the summer campaign."""
    
    prompt = f"""Please conduct market research for our summer sunglasses campaign.

Today's date is {datetime.now().strftime("%Y-%m-%d")}.

Use the available tools to:
1. Search for current sunglasses fashion trends
2. Review our product catalog
3. Recommend products that match the trends

Provide a comprehensive analysis with your findings and recommendations."""
    
    messages = [{"role": "user", "content": prompt}]
    
    # Get tool definitions
    tools_list = tools.get_available_tools_claude()
    
    # Agent loop
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=tools_list
        )
        
        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Add Claude's response to messages
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Execute tools
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    
                    utils.log_tool_call_html(tool_name, json.dumps(tool_input))
                    
                    # Execute the tool
                    result = tools.handle_tool_call_claude(tool_name, tool_input)
                    utils.log_tool_result_html(result)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result) if not isinstance(result, str) else result
                    })
            
            # Add tool results back
            messages.append({
                "role": "user",
                "content": tool_results
            })
        else:
            # Final answer
            final_content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_content += block.text
            
            utils.log_final_summary_html(final_content)
            return (final_content, messages) if return_messages else final_content
    
    return "[⚠️ Max iterations reached]"


# =========================
# AGENT 2: Graphic Designer Agent
# =========================

def graphic_designer_agent(
    trend_insights: str,
    model: str = "claude-sonnet-4-20250514",
    caption_style: str = "short punchy",
    size: str = "1024x1024"
) -> dict:
    """
    Uses Claude to generate a marketing prompt/caption and OpenAI DALL-E to generate the image.
    
    Args:
        trend_insights: Trend summary from researcher agent
        model: Claude model to use
        caption_style: Style hint for caption
        size: Image resolution
        
    Returns:
        dict: Contains image_url, image_path, prompt, and caption
    """
    
    utils.log_agent_title_html("Graphic Designer Agent", "🎨")
    
    system_prompt = """You are a creative director specializing in fashion advertising visuals.

Your task is to design compelling campaign imagery for a sunglasses brand."""
    
    prompt = f"""Based on these trend insights:

\"\"\"{trend_insights}\"\"\"

Please create:
1. A detailed image generation prompt for DALL-E that captures the essence of these trends visually
2. A {caption_style} caption that would accompany this image in the campaign

Format your response as JSON:
{{
  "image_prompt": "detailed prompt for DALL-E...",
  "caption": "punchy campaign caption..."
}}

Make the image prompt vivid, specific, and aligned with luxury sunglasses marketing."""
    
    # Get prompt and caption from Claude
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract response
    response_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            response_text += block.text
    
    utils.log_tool_result_html(response_text)
    
    # Parse JSON response
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            design_data = json.loads(json_match.group())
        else:
            design_data = json.loads(response_text)
        
        image_prompt = design_data.get("image_prompt", "")
        caption = design_data.get("caption", "")
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        image_prompt = response_text[:500]
        caption = "Experience the trend."
    
    # Generate image using DALL-E
    if openai_client:
        try:
            dalle_response = openai_client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size=size,
                quality="standard",
                n=1
            )
            
            image_url = dalle_response.data[0].url
            
            # Download and save image
            img_response = requests.get(image_url, timeout=30)
            img = Image.open(BytesIO(img_response.content))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"campaign_image_{timestamp}.png"
            img.save(image_filename)
            
            utils.log_tool_result_html(f"✅ Image generated and saved as {image_filename}")
            
            return {
                "image_url": image_url,
                "image_path": image_filename,
                "prompt": image_prompt,
                "caption": caption
            }
        except Exception as e:
            utils.log_tool_result_html(f"⚠️ Image generation failed: {e}")
            return {
                "error": str(e),
                "prompt": image_prompt,
                "caption": caption
            }
    else:
        return {
            "error": "OpenAI client not available",
            "prompt": image_prompt,
            "caption": caption
        }


# =========================
# AGENT 3: Copywriter Agent
# =========================

def copywriter_agent(
    image_path: str,
    trend_summary: str,
    model: str = "claude-sonnet-4-20250514"
) -> dict:
    """
    Generates a campaign quote by analyzing the image and trends using Claude's vision capabilities.
    
    Args:
        image_path: Path to campaign image
        trend_summary: Market research findings
        model: Claude model to use
        
    Returns:
        dict: Contains quote and justification
    """
    
    utils.log_agent_title_html("Copywriter Agent", "✍️")
    
    # Read and encode image
    try:
        with open(image_path, "rb") as img_file:
            image_data = base64.standard_b64encode(img_file.read()).decode("utf-8")
        
        # Determine image type
        if image_path.lower().endswith(".png"):
            media_type = "image/png"
        elif image_path.lower().endswith((".jpg", ".jpeg")):
            media_type = "image/jpeg"
        else:
            media_type = "image/png"
    except Exception as e:
        return {
            "error": f"Failed to read image: {e}",
            "quote": "Unavailable",
            "justification": "Image could not be processed"
        }
    
    system_prompt = """You are a luxury brand copywriter specializing in fashion campaigns.

Your task is to create compelling marketing copy that resonates with style-conscious consumers."""
    
    prompt = f"""I need you to create marketing copy for this sunglasses campaign.

Here is the market research:
\"\"\"{trend_summary}\"\"\"

Please analyze the attached campaign image and create:
1. A short, memorable campaign quote (1-2 sentences max)
2. A justification explaining how the quote connects the visual to the trends

Format your response as JSON:
{{
  "quote": "Your campaign quote here",
  "justification": "Why this quote works..."
}}"""
    
    # Call Claude with vision
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    
    # Extract response
    response_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            response_text += block.text
    
    utils.log_tool_result_html(response_text)
    
    # Parse JSON response
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            copy_data = json.loads(json_match.group())
        else:
            copy_data = json.loads(response_text)
        
        return {
            "quote": copy_data.get("quote", "Experience the moment."),
            "justification": copy_data.get("justification", "This quote captures the essence of the campaign.")
        }
    except json.JSONDecodeError:
        # Fallback
        return {
            "quote": response_text[:200] if response_text else "Experience the moment.",
            "justification": "Generated from campaign analysis."
        }


# =========================
# AGENT 4: Packaging Agent
# =========================

def packaging_agent(
    trend_summary: str,
    image_url: str,
    quote: str,
    justification: str,
    output_path: str = "campaign_summary.md",
    model: str = "claude-sonnet-4-20250514"
) -> str:
    """
    Creates an executive-ready markdown report with all campaign materials.
    
    Args:
        trend_summary: Market research findings
        image_url: Path to campaign image
        quote: Campaign quote
        justification: Why the campaign works
        output_path: Where to save markdown file
        model: Claude model to use
        
    Returns:
        str: Path to saved markdown file
    """
    
    utils.log_agent_title_html("Packaging Agent", "📦")
    
    # Beautify the trend summary for executives
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system="You are a marketing communication expert writing elegant campaign summaries for executives.",
        messages=[
            {
                "role": "user",
                "content": f"""Please rewrite the following trend summary to be clear, professional, and engaging for a CEO audience:

\"\"\"{trend_summary.strip()}\"\"\"

Keep it concise but impactful."""
            }
        ]
    )
    
    beautified_summary = ""
    for block in response.content:
        if hasattr(block, "text"):
            beautified_summary += block.text
    
    utils.log_tool_result_html(beautified_summary)
    
    # Create styled image reference
    styled_image_html = f"""
![Campaign Visual]({image_url})
"""
    
    # Combine all parts into markdown
    markdown_content = f"""# 🕶️ Summer Sunglasses Campaign – Executive Summary

## 📊 Refined Trend Insights
{beautified_summary.strip()}

## 🎯 Campaign Visual
{styled_image_html}

## ✍️ Campaign Quote
{quote.strip()}

## ✅ Why This Works
{justification.strip()}

---

*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*
*Powered by Claude (Anthropic) Multi-Agent Workflow*
"""
    
    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    utils.log_tool_result_html(f"✅ Report saved to: {output_path}")
    
    return output_path


# =========================
# Full Campaign Pipeline
# =========================

def run_sunglasses_campaign_pipeline(
    output_path: str = None,
    model: str = "claude-sonnet-4-20250514"
) -> dict:
    """
    Runs the full summer sunglasses campaign pipeline:
    1. Market research (search trends + match products)
    2. Generate visual + caption
    3. Generate quote based on image + trend
    4. Create executive markdown report
    
    Args:
        output_path: Custom path for markdown report
        model: Claude model to use
        
    Returns:
        dict: Dictionary containing all intermediate results + path to final report
    """
    
    if output_path is None:
        output_path = f"campaign_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    
    print("\n" + "=" * 80)
    print("🚀 STARTING MULTI-AGENT CAMPAIGN PIPELINE")
    print("=" * 80)
    
    # 1. Run market research agent
    print("\n[1/4] Running Market Research Agent...")
    trend_summary = market_research_agent(model=model)
    print("✅ Market research completed")
    
    # 2. Generate image + caption
    print("\n[2/4] Running Graphic Designer Agent...")
    visual_result = graphic_designer_agent(trend_insights=trend_summary, model=model)
    image_path = visual_result.get("image_path", "")
    
    if "error" in visual_result:
        print(f"⚠️ Image generation had issues: {visual_result['error']}")
        print("Continuing with available data...")
    else:
        print("🖼️ Image generated successfully")
    
    # 3. Generate quote based on image + trends
    print("\n[3/4] Running Copywriter Agent...")
    if image_path and os.path.exists(image_path):
        quote_result = copywriter_agent(
            image_path=image_path,
            trend_summary=trend_summary,
            model=model
        )
        quote = quote_result.get("quote", "Experience the moment.")
        justification = quote_result.get("justification", "Campaign analysis.")
        print("💬 Quote created successfully")
    else:
        quote = "Experience the trend."
        justification = "Image unavailable for analysis."
        print("⚠️ Skipping copywriter (no image available)")
    
    # 4. Generate markdown report
    print("\n[4/4] Running Packaging Agent...")
    md_path = packaging_agent(
        trend_summary=trend_summary,
        image_url=image_path if image_path else "image_unavailable.png",
        quote=quote,
        justification=justification,
        output_path=output_path,
        model=model
    )
    print(f"📦 Report generated: {md_path}")
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)
    
    return {
        "trend_summary": trend_summary,
        "visual": visual_result,
        "quote": quote_result if 'quote_result' in locals() else {"quote": quote, "justification": justification},
        "markdown_path": md_path
    }


# =========================
# Example Usage
# =========================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  MULTI-AGENT SUNGLASSES CAMPAIGN PIPELINE                    ║
║                         Powered by Claude (Anthropic)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

This pipeline demonstrates:
- 🕵️‍♂️ Market Research Agent: Scans trends and matches products
- 🎨 Graphic Designer Agent: Creates visual concepts
- ✍️ Copywriter Agent: Generates quotes using vision
- 📦 Packaging Agent: Assembles executive reports

Note: Requires both ANTHROPIC_API_KEY and OPENAI_API_KEY (for DALL-E image generation)
""")
    
    # Run individual agent examples
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Running Individual Agents")
    print("=" * 80)
    
    # Test market research agent
    print("\n--- Testing Market Research Agent ---")
    research_result = market_research_agent()
    print(f"\nResult preview: {research_result[:200]}...")
    
    # Run full pipeline
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Running Full Pipeline")
    print("=" * 80)
    
    try:
        results = run_sunglasses_campaign_pipeline()
        
        print("\n📄 Campaign Summary:")
        print(f"- Markdown report: {results['markdown_path']}")
        print(f"- Image: {results['visual'].get('image_path', 'N/A')}")
        print(f"- Quote: {results['quote'].get('quote', 'N/A')[:100]}...")
        
        # Display the markdown content
        print("\n" + "=" * 80)
        print("FINAL REPORT PREVIEW")
        print("=" * 80)
        
        with open(results["markdown_path"], "r", encoding="utf-8") as f:
            print(f.read())
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("Make sure you have:")
        print("1. ANTHROPIC_API_KEY in your .env file")
        print("2. OPENAI_API_KEY in your .env file (for DALL-E)")
        print("3. tools_multi_agent.py and utils_multi_agent.py modules")