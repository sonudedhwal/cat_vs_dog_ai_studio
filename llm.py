"""Generative AI / LLM helpers (Anthropic Claude).

Set ANTHROPIC_API_KEY in your environment (or paste it in the app sidebar).
"""
import base64
import os

import anthropic

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")


def _client():
    key = os.getenv("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else None


def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def describe_pet(jpeg_bytes: bytes, predicted_label: str) -> str:
    """Vision + text generation: breed guess, traits, care tips, fun fact."""
    client = _client()
    if client is None:
        return "Add an Anthropic API key in the sidebar to enable AI descriptions."
    b64 = base64.standard_b64encode(jpeg_bytes).decode()
    prompt = (
        f"A classifier predicted this image is: {predicted_label}. "
        "Look at the image yourself and reply in short markdown with: "
        "1) what animal you actually see (say if the classifier seems wrong), "
        "2) likely breed or type with a confidence note, "
        "3) three care tips, 4) one fun fact. Keep it under 150 words."
    )
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return _text(resp)
    except Exception as e:  # network, auth, quota...
        return f"AI description unavailable: {e}"


def chat(history: list[dict], context: str = "") -> str:
    """Multi-turn pet-care assistant. history = [{'role','content'}, ...]."""
    client = _client()
    if client is None:
        return "Add an Anthropic API key in the sidebar to chat with the AI pet expert."
    system = (
        "You are a friendly pet expert for cats and dogs. Give practical, safe advice, "
        "and recommend a vet for medical concerns. Be concise."
    )
    if context:
        system += f"\nRecent classified images from the user's history: {context}"
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=700, system=system, messages=history)
        return _text(resp)
    except Exception as e:
        return f"Chat unavailable: {e}"
