def reflection_pattern(task):
    # STEP 1: Generate initial output
    draft = writer_agent(task)
    print("📝 Initial Draft:", draft)
    
    # STEP 2: Reflect on the draft
    reflection = editor_agent(f"Critique this draft: {draft}")
    print("🔍 Reflection:", reflection)
    
    # STEP 3: Revise based on reflection
    final = writer_agent(f"Improve this draft based on feedback:\n\nDraft: {draft}\n\nFeedback: {reflection}")
    print("✨ Final Version:", final)
    
    return final


## Real-World Analogy:

# **Without Reflection** (like turning in first draft):
# ```
# Write essay → Submit
# ```

# **With Reflection** (like self-editing):
# ```
# Write essay → Read it over → Notice issues → Rewrite → Submit
# (Much better!)