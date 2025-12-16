# Warning control
import warnings
warnings.filterwarnings('ignore')

# Import required libraries
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Load environment variables from .env file
load_dotenv()

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-3-haiku-20240307"  # Using Claude 3 Haiku (fast and cost-effective)

# Define the state
class ResearchState(TypedDict):
    topic: str
    plan: str
    draft: str
    final_article: str
    messages: Annotated[list, add_messages]

# Agent 1: Content Planner
def planner_agent(state: ResearchState) -> ResearchState:
    """Plan engaging and factually accurate content"""
    topic = state["topic"]

    print("\n" + "="*80)
    print("CONTENT PLANNER WORKING...")
    print("="*80)

    prompt = f"""You are a Content Planner working on planning a blog article about: {topic}

Your job is to collect information that helps the audience learn something and make informed decisions.
Your work will be the basis for the Content Writer to write an article on this topic.

Please create a comprehensive content plan that includes:
1. Latest trends, key players, and noteworthy news on {topic}
2. Target audience analysis (their interests and pain points)
3. Detailed content outline including:
   - Introduction
   - Key points (3-5 main sections)
   - Call to action
4. SEO keywords and relevant data or sources

Provide a comprehensive content plan document with an outline, audience analysis, SEO keywords, and resources."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    plan = response.content[0].text
    print(f"\nPlan created ({len(plan)} characters)")

    return {
        **state,
        "plan": plan,
        "messages": [{"role": "assistant", "content": f"Plan completed for: {topic}"}]
    }

# Agent 2: Content Writer
def writer_agent(state: ResearchState) -> ResearchState:
    """Write insightful and factually accurate content"""
    topic = state["topic"]
    plan = state["plan"]

    print("\n" + "="*80)
    print("CONTENT WRITER WORKING...")
    print("="*80)

    prompt = f"""You are a Content Writer working on writing a new opinion piece about: {topic}

You base your writing on the work of the Content Planner, who provided the following plan:

{plan}

Your task is to:
1. Use the content plan to craft a compelling blog post on {topic}
2. Incorporate SEO keywords naturally
3. Name sections/subtitles in an engaging manner
4. Structure the post with:
   - An engaging introduction
   - Insightful body (2-3 paragraphs per section)
   - A summarizing conclusion
5. Proofread for grammatical errors and alignment with a professional voice

Provide objective and impartial insights backed up with information from the Content Planner.
Acknowledge when statements are opinions versus objective facts.

Write a well-structured blog post in markdown format, ready for publication."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    draft = response.content[0].text
    print(f"\nDraft written ({len(draft)} characters)")

    return {
        **state,
        "draft": draft,
        "messages": state["messages"] + [{"role": "assistant", "content": "Draft completed"}]
    }

# Agent 3: Editor
def editor_agent(state: ResearchState) -> ResearchState:
    """Edit and polish the blog post"""
    draft = state["draft"]

    print("\n" + "="*80)
    print("EDITOR REVIEWING...")
    print("="*80)

    prompt = f"""You are an Editor who receives a blog post from the Content Writer.

Your goal is to review and polish the blog post to ensure that it:
1. Follows journalistic best practices
2. Provides balanced viewpoints when presenting opinions or assertions
3. Avoids major controversial topics or extreme opinions when possible
4. Has proper grammar, spelling, and punctuation
5. Maintains a professional and engaging tone
6. Each section has 2-3 well-developed paragraphs

Here is the draft to edit:

{draft}

Please provide the final, polished version of the blog post in markdown format, ready for publication."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    final_article = response.content[0].text
    print(f"\nFinal article ready ({len(final_article)} characters)")

    return {
        **state,
        "final_article": final_article,
        "messages": state["messages"] + [{"role": "assistant", "content": "Editing completed"}]
    }

# Build the graph
def create_research_workflow():
    """Create the LangGraph workflow"""
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("planner", planner_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("editor", editor_agent)

    # Add edges (define the flow)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "writer")
    workflow.add_edge("writer", "editor")
    workflow.add_edge("editor", END)

    return workflow.compile()

# Main execution
if __name__ == "__main__":
    # Create the workflow
    app = create_research_workflow()

    # Set your topic here
    topic = "Nicotine effects on adults with attention-deficit/hyperactivity disorder (ADHD)"

    print("\n" + "="*80)
    print(f"STARTING RESEARCH WORKFLOW FOR: {topic}")
    print("="*80)

    # Run the workflow
    initial_state = {
        "topic": topic,
        "plan": "",
        "draft": "",
        "final_article": "",
        "messages": []
    }

    result = app.invoke(initial_state)

    # Display the final result
    print("\n" + "="*80)
    print("FINAL ARTICLE")
    print("="*80)
    print("\n" + result["final_article"])

    # Optionally save to file
    output_file = f"article_{topic.replace(' ', '_').lower()}.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result["final_article"])

    print(f"\nArticle saved to: {output_file}")
