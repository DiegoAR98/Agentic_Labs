# 1. USER calls the orchestrator
result = generate_research_report_with_tools(
    "Research quantum computing"
)

# 2. ORCHESTRATOR sets up conversation
messages = [
    {"role": "system", "content": "You are a research assistant..."},
    {"role": "user", "content": "Research quantum computing"}
]

# 3. ORCHESTRATOR gives tools TO the LLM
tools = [arxiv_search_tool, tavily_search_tool]

# 4. LLM decides: "I need arxiv_search_tool"
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools  # ← LLM can see and choose these
)

# 5. ORCHESTRATOR executes the tool the LLM requested
if response.choices[0].message.tool_calls:
    tool_name = "arxiv_search_tool"  # LLM chose this
    result = arxiv_search_tool(query="quantum computing")

# 6. ORCHESTRATOR sends result back to LLM
messages.append({"role": "tool", "content": result})

# 7. Loop continues...

## Analogy:

# Think of it like a restaurant:
# ```
# generate_research_report_with_tools = The Restaurant Manager
# ├─ Manages customer orders (user prompts)
# ├─ Talks to the chef (LLM)
# ├─ Provides kitchen equipment (tools)
# └─ Serves final dish (return final_text)

# Tools = Kitchen Equipment
# ├─ 🔪 arxiv_search_tool (knife)
# ├─ 🍳 tavily_search_tool (pan)
# └─ Chef (LLM) chooses which to use

# Messages = Order History
# ├─ Customer order
# ├─ Chef's questions
# ├─ Ingredient checks (tool results)
# └─ Final dish description

# What is Orchestrator ?

# ┌─────────────────────────────────────┐
#         │      ORCHESTRATOR                   │
#         │  (generate_research_report_with_    │
#         │         tools function)             │
#         └──────────┬──────────────────────────┘
#                    │
#         ┌──────────┴──────────┐
#         │                     │
#         ▼                     ▼
#    ┌─────────┐          ┌──────────┐
#    │   LLM   │          │  TOOLS   │
#    │         │          │          │
#    │  GPT-4  │◄────────►│ arxiv    │
#    │         │          │ tavily   │
#    └─────────┘          └──────────┘
#         ▲
#         │
#         │
#    ┌────┴─────┐
#    │   USER   │
#    └──────────┘


