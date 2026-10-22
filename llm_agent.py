"""
llm_agent.py — Step 2: Basic LLM chatbot using the new google-genai SDK.

Uses google.genai (google-genai package) — the current supported SDK.
The older google.generativeai package is deprecated as of late 2026.
"""
from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_TOKENS,
    AGENT_NAME, STORE_NAME, REFERENCE_DATE
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are {AGENT_NAME}, the AI customer support assistant for {STORE_NAME} — \
a leading Indian online store selling groceries and electronics.

Today's date is {REFERENCE_DATE}. Use this date for ALL deadline and timeline calculations.

## Your responsibilities
- Help customers with: order status, delivery, cancellations, returns, refunds, \
warranties, product questions, and ZentroMart Plus membership.
- Answer clearly and politely in the customer's language (English, Hindi, or Hinglish).
- Keep responses concise and natural — they may be read aloud by a voice system.

## Rules you must never break
1. NEVER invent policies, prices, delivery dates, or refund timelines that are not \
in the provided documents or order data.
2. NEVER reveal an order's details unless identity has been verified (last 4 digits of \
registered phone number OR 6-digit delivery pincode).
3. NEVER claim a ticket or action succeeded without tool confirmation.
4. If the question is outside your knowledge, say so honestly and offer to connect \
the customer with a human agent.
5. Ask only ONE clarifying question at a time. Never repeat a question already asked.
6. Business days are Monday–Saturday, excluding Indian public holidays. \
Return windows use calendar days.
7. For electronics returned after {REFERENCE_DATE}, apply the 7-day return window \
(v4 policy effective 2026-08-01). Electronics delivered before 2026-08-01 keep the 10-day window.

## Tone
- Warm, professional, and brief.
- Do not use filler phrases like "Of course!" or "Certainly!".
- When you don't know, say: "I don't have that information right now. \
Would you like me to connect you with a human agent?"
"""


def _get_client() -> genai.Client:
    """Return a configured google-genai Client."""
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file.\n"
            "Get a free key at: https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def _make_config() -> types.GenerateContentConfig:
    """Return the generation config with system prompt baked in."""
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
    )


def simple_respond(
    user_message: str,
    history: list[dict] | None = None,
    context_chunks: list[dict] | None = None,
) -> str:
    """
    Single-turn or multi-turn LLM response.

    Args:
        user_message:   The customer's latest message (may include prepended RAG evidence).
        history:        List of {"role": "user"|"model", "parts": [str]} dicts.
        context_chunks: RAG chunks to prepend as evidence (formatted into user_message).

    Returns:
        The assistant's reply as a plain string.
    """
    client = _get_client()

    # Prepend evidence if provided separately
    if context_chunks:
        evidence_text = _format_evidence(context_chunks)
        prompt = (
            f"## Relevant company documents (use these to answer):\n"
            f"{evidence_text}\n\n"
            f"## Customer question:\n{user_message}"
        )
    else:
        prompt = user_message

    # Convert history dicts → google.genai Content objects
    contents: list[types.Content] = []
    for turn in (history or []):
        role = turn.get("role", "user")
        text = turn.get("parts", [""])[0] if turn.get("parts") else ""
        contents.append(
            types.Content(role=role, parts=[types.Part(text=text)])
        )

    # Append the current user message
    contents.append(
        types.Content(role="user", parts=[types.Part(text=prompt)])
    )

    # Try configured model first, then fallback models if rate-limited / unavailable
    candidate_models = [LLM_MODEL] + [
        m for m in ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3-flash-preview", "gemini-3.8-flash"]
        if m != LLM_MODEL
    ]

    last_error = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=_make_config(),
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            continue

    return f"I apologize, but I am currently having trouble connecting to my knowledge service. Error: {last_error}"


def _format_evidence(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable evidence block."""
    parts = []
    for chunk in chunks:
        section = chunk.get("section_id", chunk.get("doc_id", "?"))
        status  = chunk.get("status", "")
        text    = chunk.get("text", "")
        parts.append(f"[{section}] ({status})\n{text}")
    return "\n\n---\n\n".join(parts)
