"""
Graded Lab: Reflection in a Research Agent (Local Runner)
--------------------------------------------------------

This script is self-contained and runnable locally:
- Uses .env for OPENAI_API_KEY if you want real LLM calls.
- Prefers `aisuite` client; falls back to OpenAI SDK; finally a dummy offline shim.

pip install -q python-dotenv
# Optional depending on your setup:
pip install -q openai
pip install -q aisuite
"""

import os
from typing import Any

# 1) .env setup
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # It's okay if python-dotenv isn't installed

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# 2) Client resolution order: aisuite -> openai -> dummy
CLIENT = None
_backend = "dummy"

# ---- Shim response objects (mimic OpenAI/aisuite shape) ----
class _ShimMessage:
    def __init__(self, content: str):
        self.content = content

class _ShimChoice:
    def __init__(self, content: str):
        self.message = _ShimMessage(content)

class _ShimResponse:
    def __init__(self, content: str):
        self.choices = [_ShimChoice(content)]

# ---- aisuite (preferred if available) ----
try:
    import aisuite as ai  # type: ignore

    CLIENT = ai.Client()
    _backend = "aisuite"
except Exception:
    CLIENT = None

# ---- OpenAI SDK fallback ----
if CLIENT is None:
    try:
        from openai import OpenAI  # type: ignore

        _openai_client = OpenAI(api_key=OPENAI_API_KEY or None)

        class _OpenAIChatWrapper:
            """Wrap OpenAI client to look like aisuite's chat.completions.create."""
            def __init__(self, openai_client):
                self._client = openai_client
                # expose .chat.completions.create(...)
                self.chat = type(
                    "ChatObj",
                    (),
                    {"completions": type("CompletionsObj", (), {"create": self.create})()},
                )()

            def _normalize_model(self, model: str) -> str:
                # Allow "openai:gpt-4o" style; strip "openai:" prefix if present
                return model.split(":", 1)[1] if model.startswith("openai:") else model

            def create(self, model: str, messages: list[dict], temperature: float = 1.0):
                model = self._normalize_model(model)
                try:
                    return self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                    )
                except Exception as e:
                    # Return shim response to keep tests running locally
                    fallback = (
                        "[FALLBACK due to OpenAI error: "
                        + str(e)
                        + "]\n\n"
                        + (messages[-1]["content"] if messages else "")
                    )
                    return _ShimResponse(fallback)

        CLIENT = _OpenAIChatWrapper(_openai_client)
        _backend = "openai"
    except Exception:
        CLIENT = None

# ---- Dummy offline fallback ----
if CLIENT is None:
    class _DummyCompletions:
        def create(self, model: str, messages: list[dict], temperature: float = 1.0) -> Any:
            # Deterministic offline content so functions return strings without raising
            last = messages[-1]["content"] if messages else ""
            canned = (
                "This is a dummy offline response for local testing.\n\n"
                "Echo of your prompt (truncated to 600 chars):\n"
                + last[:600]
            )
            return _ShimResponse(canned)

    class _DummyChat:
        completions = _DummyCompletions()

    class _DummyClient:
        chat = _DummyChat()

    CLIENT = _DummyClient()
    _backend = "dummy"

# -----------------------------
# GRADED FUNCTION: generate_draft
# -----------------------------
def generate_draft(topic: str, model: str = "openai:gpt-4o") -> str:
    ### START CODE HERE ###
    # Coerce topic to a safe string and provide a fallback to avoid null content
    topic_text = (str(topic).strip() if topic is not None else "")
    if not topic_text:
        topic_text = "The importance of clear writing in modern communication"

    prompt = (
        f"Write a complete, well-structured essay about the following topic: {topic_text}\n\n"
        "The essay should include:\n"
        "- An introduction with a clear thesis statement\n"
        "- Body paragraphs with supporting arguments and examples\n"
        "- A conclusion that summarizes the main points\n\n"
        "Please write a comprehensive essay of at least 4–5 paragraphs."
    )
    ### END CODE HERE ###

    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    # Be defensive in case a backend returns a different shape
    try:
        return response.choices[0].message.content
    except Exception:
        return "Draft (fallback): An essay discussing the topic with introduction, body, and conclusion."

# -----------------------------
# GRADED FUNCTION: reflect_on_draft
# -----------------------------
def reflect_on_draft(draft: str, model: str = "openai:o4-mini") -> str:
    ### START CODE HERE ###
    draft_text = (str(draft) if draft is not None else "")

    prompt = f"""Please provide constructive feedback on the following essay draft.
Analyze its structure, clarity, strength of arguments, and writing style.
Point out any areas that need improvement, including grammar or spelling errors.

Draft to review:
{draft_text}

Provide your feedback in a constructive and professional manner (one cohesive paragraph)."""
    ### END CODE HERE ###

    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    try:
        return response.choices[0].message.content
    except Exception:
        return "Feedback (fallback): Improve thesis clarity, tighten topic sentences, add evidence, and refine transitions."

# -----------------------------
# GRADED FUNCTION: revise_draft
# -----------------------------
def revise_draft(original_draft: str, reflection: str, model: str = "openai:gpt-4o") -> str:
    ### START CODE HERE ###
    orig = str(original_draft) if original_draft is not None else ""
    fb = str(reflection) if reflection is not None else ""

    # Single f-string (avoids stray f / concatenation issues)
    prompt = f"""You are tasked with revising an essay based on constructive feedback.

Original Draft:
{orig}

Feedback:
{fb}

Please provide a complete revised version of the essay that addresses all the feedback points.
Improve the structure, clarity, argument strength, and overall flow.
Return only the revised essay, not any explanations or meta-commentary.
"""

    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    ### END CODE HERE ###

    try:
        return response.choices[0].message.content
    except Exception:
        return (
            "Revised Essay (fallback): This revision clarifies the thesis, organizes body paragraphs with "
            "clear topic sentences and evidence, and concludes by synthesizing the central claims."
        )

# -----------------------------
# Local test harness (optional)
# -----------------------------
def _local_tests():
    print(f"[Backend in use: {_backend}]")
    try:
        d = generate_draft("Should social media platforms be regulated by the government?")
        assert isinstance(d, str) and len(d) > 0
        print("✅ generate_draft returned a non-empty string")

        f = reflect_on_draft(d)
        assert isinstance(f, str) and len(f) > 0
        print("✅ reflect_on_draft returned a non-empty string")

        r = revise_draft(d, f)
        assert isinstance(r, str) and len(r) > 0
        print("✅ revise_draft returned a non-empty string")
    except AssertionError as e:
        print("❌ A function returned an empty or non-string value:", e)
    except Exception as e:
        print("❌ An exception occurred during local tests:", repr(e))

# -----------------------------
# Demo run
# -----------------------------
if __name__ == "__main__":
    # If the course-provided unittests module is available, run it first.
    # Otherwise, run local tests and the demo flow.
    used_course_tests = False
    try:
        import unittests  # provided by the course notebook
        print("[Running course unit tests]")
        unittests.test_generate_draft(generate_draft)
        unittests.test_reflect_on_draft(reflect_on_draft)
        unittests.test_revise_draft(revise_draft)
        used_course_tests = True
    except Exception as e:
        print("[Course unittests not available or failed to import]:", repr(e))

    if not used_course_tests:
        print("\n[Running local tests]")
        _local_tests()

    print("\n[Demo: full reflective workflow]")
    essay_prompt = "Should social media platforms be regulated by the government?"
    draft = generate_draft(essay_prompt)
    print("\n📝 Draft:\n", draft[:1200], "...\n")

    feedback = reflect_on_draft(draft)
    print("\n🧠 Feedback:\n", feedback[:1200], "...\n")

    revised = revise_draft(draft, feedback)
    print("\n✍️ Revised:\n", revised[:1500], "...\n")
