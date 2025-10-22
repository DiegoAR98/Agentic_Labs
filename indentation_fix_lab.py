# GRADED FUNCTION: reflection_and_rewrite
def reflection_and_rewrite(report, model: str = "gpt-4o-mini", temperature: float = 0.3) -> dict:
    """
    Generates a structured reflection AND a revised research report.
    Accepts raw text OR the messages list returned by generate_research_report_with_tools.

    Returns:
        dict with keys:
          - "reflection": structured reflection text
          - "revised_report": improved version of the input report
    """

    # Input can be plain text or a list of messages, this function detects and parses accordingly
    report = research_tools.parse_input(report)

    ### START CODE HERE ###

    # Define the prompt. A multi-line f-string is typically used for this.
    # Remember it should ask the model to output ONLY valid JSON with this structure:
    # {{ "reflection": "<text>", "revised_report": "<text>" }}
     user_prompt = f"""Please analyze the following research report and provide:

1. A structured reflection covering:
   - Strengths of the report
   - Limitations or weaknesses
   - Suggestions for improvement
   - Opportunities for enhancement

2. A revised version of the report that incorporates your reflection to improve clarity, accuracy, and academic tone.

Output your response as ONLY valid JSON with this exact structure (no additional text or markdown):
{{
  "reflection": "Your detailed reflection here covering strengths, limitations, suggestions, and opportunities",
  "revised_report": "Your improved version of the report here with better clarity and academic tone"
}}

Research Report to Review:
{report}
"""
    # Get a response from the LLM
    response = CLIENT.chat.completions.create( 
        # Pass in the model
        model=model,
        messages=[ 
            # System prompt is already defined
            {"role": "system", "content": "You are an academic reviewer and editor."},
            # Add user prompt
            {"role": "user", "content": None},
        ],
        # Set the temperature equal to the temperature parameter passed to the function
        temperature=temperature
    )

    ### END CODE HERE ###

    # Extract output
    llm_output = response.choices[0].message.content.strip()

    # Check if output is valid JSON
    try:
        data = json.loads(llm_output)
    except json.JSONDecodeError:
        raise Exception("The output of the LLM was not valid JSON. Adjust your prompt.")

    return {
        "reflection": str(data.get("reflection", "")).strip(),
        "revised_report": str(data.get("revised_report", "")).strip(),
    }