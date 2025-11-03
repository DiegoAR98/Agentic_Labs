# Warning control
import warnings
warnings.filterwarnings('ignore')

# Import libraries
from crewai import Agent, Crew, Task
from pydantic import BaseModel
import os
import anthropic

# Set up Anthropic API
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Configure CrewAI to use Anthropic/Claude
os.environ["OPENAI_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.anthropic.com/v1"
os.environ["OPENAI_MODEL_NAME"] = "claude-3-5-sonnet-20241022"

# Set up Serper API for search
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Import CrewAI Tools
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Creating Agents

# Agent 1: Venue Coordinator
venue_coordinator = Agent(
    role="Venue Coordinator",
    goal="Identify and book an appropriate venue based on event requirements",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "With a keen sense of space and understanding of event logistics, "
        "you excel at finding and securing the perfect venue that fits the event's theme, "
        "size, and budget constraints."
    )
)

# Agent 2: Logistics Manager
logistics_manager = Agent(
    role='Logistics Manager',
    goal="Manage all logistics for the event including catering and equipment",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Organized and detail-oriented, "
        "you ensure that every logistical aspect of the event "
        "from catering to equipment setup "
        "is flawlessly executed to create a seamless experience."
    )
)

# Agent 3: Marketing and Communications Agent
marketing_communications_agent = Agent(
    role="Marketing and Communications Agent",
    goal="Effectively market the event and communicate with participants",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Creative and communicative, "
        "you craft compelling messages and engage with potential attendees "
        "to maximize event exposure and participation."
    )
)

# Creating Venue Pydantic Object
class VenueDetails(BaseModel):
    name: str
    address: str
    capacity: int
    booking_status: str

# Creating Tasks

venue_task = Task(
    description="Find a venue in {event_city} that meets criteria for {event_topic}.",
    expected_output="All the details of a specifically chosen venue you found to accommodate the event.",
    human_input=True,
    output_json=VenueDetails,
    output_file="venue_details.json",
    agent=venue_coordinator
)

logistics_task = Task(
    description=(
        "Coordinate catering and equipment for an event "
        "with {expected_participants} participants on {tentative_date}."
    ),
    expected_output=(
        "Confirmation of all logistics arrangements "
        "including catering and equipment setup."
    ),
    human_input=True,
    async_execution=True,
    agent=logistics_manager
)

marketing_task = Task(
    description=(
        "Promote the {event_topic} "
        "aiming to engage at least {expected_participants} potential attendees."
    ),
    expected_output=(
        "Report on marketing activities "
        "and attendee engagement formatted as markdown."
    ),
    async_execution=True,
    output_file="marketing_report.md",
    agent=marketing_communications_agent
)

# Creating the Crew
event_management_crew = Crew(
    agents=[venue_coordinator, 
            logistics_manager, 
            marketing_communications_agent],
    
    tasks=[venue_task, 
           logistics_task, 
           marketing_task],
    
    verbose=True
)

# Running the Crew
event_details = {
    'event_topic': "Tech Innovation Conference",
    'event_description': "A gathering of tech innovators "
                         "and industry leaders "
                         "to explore future technologies.",
    'event_city': "San Francisco",
    'tentative_date': "2024-09-15",
    'expected_participants': 500,
    'budget': 20000,
    'venue_type': "Conference Hall"
}

# Execute the crew
result = event_management_crew.kickoff(inputs=event_details)

# Display the generated venue_details.json file
import json
from pprint import pprint

print("\n=== Venue Details ===")
with open('venue_details.json') as f:
    data = json.load(f)

pprint(data)

# Display the generated marketing_report.md file
# Note: Wait ~45 seconds after kickoff completes before running this
print("\n=== Marketing Report ===")
from IPython.display import Markdown, display

try:
    with open('marketing_report.md', 'r') as f:
        content = f.read()
    display(Markdown(content))
except FileNotFoundError:
    print("Marketing report not yet generated. Please wait 45 seconds and try again.")