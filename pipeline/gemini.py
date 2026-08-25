"""Gemini as a clip-selection backend.

Written against the REST API with urllib rather than the google-genai SDK,
for the same reason ``youtube.py`` is: this repo ships to a worker box where
every added dependency is another thing that can fail to install, and the
surface actually used here is one POST.

``score.py`` owns the prompt, the windowing and the parsing. This module owns
exactly one thing: turning a (system, user) pair into text, or raising.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

API = "https://generativelanguage.googleapis.com/v1beta/models"

DEFAULT_MODEL = "gemini-3.6-flash"

#: These models think before answering, and the thinking tokens are drawn from
#: the same budget as the answer. A budget sized only for the JSON array gets
#: spent on reasoning and returns an empty candidate, which looks exactly like
#: a refusal. Ask for enough that both fit.
OUTPUT_TOKENS = 8192

TIMEOUT = 120


class GeminiError(RuntimeError):
    """The API refused, or returned something unusable."""


def api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("GEMINI_API_KEY", "") or os.environ.get(
        "GOOGLE_API_KEY", ""
    )
    if not key:
        raise GeminiError(
            "GEMINI_API_KEY is not set. Get one from https://aistudio.google.com/apikey "
            "and put it in .env"
        )
    return key


def generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    key: str | None = None,
    max_output_tokens: int = OUTPUT_TOKENS,
    json_only: bool = True,
) -> str:
    """One completion. Returns the text, or raises GeminiError."""
    model = model or DEFAULT_MODEL
    body: dict = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"maxOutputTokens": max_output_tokens},
    }
    if json_only:
        # Constrained decoding beats asking politely for no markdown fences.
        body["generationConfig"]["responseMimeType"] = "application/json"

    req = urllib.request.Request(
        f"{API}/{model}:generateContent?key={api_key(key)}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _detail(exc)
        raise GeminiError(f"Gemini API {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise GeminiError(f"cannot reach the Gemini API: {exc.reason}") from None
    except json.JSONDecodeError:
        raise GeminiError("Gemini returned a body that is not JSON") from None

    return _text(data)


def _detail(exc: urllib.error.HTTPError) -> str:
    """The one line of an error body worth showing, without the key in it."""
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        message = payload.get("error", {}).get("message", "")
        return message[:300] or exc.reason
    except Exception:
        return exc.reason


def _text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        # A prompt refused outright reports why here rather than in the HTTP code.
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        raise GeminiError(
            f"Gemini returned no candidates (blockReason={blocked})" if blocked
            else "Gemini returned no candidates"
        )

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if text:
        return text

    reason = candidate.get("finishReason", "")
    if reason == "MAX_TOKENS":
        raise GeminiError(
            "Gemini hit the output limit before writing an answer — the "
            "thinking budget consumed it. Raise max_output_tokens."
        )
    raise GeminiError(f"Gemini returned an empty answer (finishReason={reason or '?'})")
