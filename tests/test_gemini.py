"""The Gemini backend: response shapes, and the failures that look like success.

Every test here is offline — urllib is stubbed. What is being pinned is how
this module behaves when the API returns something other than the happy path,
because those are the responses that otherwise reach the caller as an
AttributeError three frames away.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from pipeline import gemini


def reply(payload: dict, monkeypatch):
    """Make urlopen return this JSON body."""
    class Fake(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(
        gemini.urllib.request, "urlopen",
        lambda req, timeout=None: Fake(json.dumps(payload).encode()),
    )


def http_error(code: int, body: dict | str, monkeypatch):
    def boom(req, timeout=None):
        raw = json.dumps(body).encode() if isinstance(body, dict) else body.encode()
        raise urllib.error.HTTPError("u", code, "reason", {}, io.BytesIO(raw))

    monkeypatch.setattr(gemini.urllib.request, "urlopen", boom)


class TestApiKey:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        assert gemini.api_key("explicit") == "explicit"

    def test_falls_back_to_gemini_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g")
        assert gemini.api_key() == "g"

    def test_accepts_google_api_key_too(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "gg")
        assert gemini.api_key() == "gg"

    def test_missing_key_names_where_to_get_one(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(gemini.GeminiError, match="aistudio.google.com"):
            gemini.api_key()


class TestGenerate:
    @pytest.fixture(autouse=True)
    def key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")

    def test_happy_path(self, monkeypatch):
        reply({"candidates": [{"content": {"parts": [{"text": "[1,2]"}]}}]}, monkeypatch)
        assert gemini.generate("sys", "user") == "[1,2]"

    def test_multiple_parts_are_joined(self, monkeypatch):
        reply({"candidates": [{"content": {"parts": [{"text": "[1,"}, {"text": "2]"}]}}]},
              monkeypatch)
        assert gemini.generate("sys", "user") == "[1,2]"

    def test_no_candidates_is_an_error_not_an_index_crash(self, monkeypatch):
        reply({"candidates": []}, monkeypatch)
        with pytest.raises(gemini.GeminiError, match="no candidates"):
            gemini.generate("sys", "user")

    def test_a_blocked_prompt_says_why(self, monkeypatch):
        reply({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}, monkeypatch)
        with pytest.raises(gemini.GeminiError, match="SAFETY"):
            gemini.generate("sys", "user")

    def test_thinking_eating_the_budget_is_named(self, monkeypatch):
        # The failure that looks like a refusal: an empty answer because the
        # thinking tokens consumed maxOutputTokens.
        reply({"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]},
              monkeypatch)
        with pytest.raises(gemini.GeminiError, match="thinking budget"):
            gemini.generate("sys", "user")

    def test_empty_answer_reports_the_finish_reason(self, monkeypatch):
        reply({"candidates": [{"finishReason": "RECITATION", "content": {"parts": []}}]},
              monkeypatch)
        with pytest.raises(gemini.GeminiError, match="RECITATION"):
            gemini.generate("sys", "user")

    def test_http_error_surfaces_the_api_message(self, monkeypatch):
        http_error(404, {"error": {"message": "model X is no longer available"}}, monkeypatch)
        with pytest.raises(gemini.GeminiError, match="no longer available"):
            gemini.generate("sys", "user")

    def test_error_body_that_is_not_json_still_raises_cleanly(self, monkeypatch):
        http_error(500, "<html>gateway</html>", monkeypatch)
        with pytest.raises(gemini.GeminiError, match="Gemini API 500"):
            gemini.generate("sys", "user")

    def test_unreachable_api(self, monkeypatch):
        def boom(req, timeout=None):
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr(gemini.urllib.request, "urlopen", boom)
        with pytest.raises(gemini.GeminiError, match="cannot reach"):
            gemini.generate("sys", "user")

    def test_non_json_body(self, monkeypatch):
        class Fake(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(gemini.urllib.request, "urlopen",
                            lambda req, timeout=None: Fake(b"not json"))
        with pytest.raises(gemini.GeminiError, match="not JSON"):
            gemini.generate("sys", "user")


class TestRequestShape:
    def test_json_mode_and_system_instruction_are_sent(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        seen = {}

        class Fake(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def capture(req, timeout=None):
            seen["url"] = req.full_url
            seen["body"] = json.loads(req.data)
            return Fake(json.dumps(
                {"candidates": [{"content": {"parts": [{"text": "[]"}]}}]}).encode())

        monkeypatch.setattr(gemini.urllib.request, "urlopen", capture)
        gemini.generate("be terse", "find clips", model="gemini-test")

        assert "gemini-test:generateContent" in seen["url"]
        assert seen["body"]["system_instruction"]["parts"][0]["text"] == "be terse"
        assert seen["body"]["contents"][0]["parts"][0]["text"] == "find clips"
        # constrained decoding beats asking politely for no markdown fences
        assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"

    def test_output_budget_leaves_room_for_thinking(self):
        assert gemini.OUTPUT_TOKENS >= 8192
