"""Clip selection: parsing what the model actually returns, and the three
post-processing steps that decide whether a customer gets five good clips or
the same moment five times."""

from __future__ import annotations

import json

import pytest

from pipeline import score
from pipeline.score import Clip


class TestParseClip:
    def test_happy_path(self):
        clip = score._parse_clip({"start": 1.5, "end": 40.0, "title": "t", "score": 82})
        assert clip.start == 1.5 and clip.score == 82

    def test_stringified_numbers(self):
        clip = score._parse_clip({"start": "1.5", "end": "40", "score": "82.0"})
        assert clip is not None and clip.duration == pytest.approx(38.5)

    @pytest.mark.parametrize(
        "raw",
        [
            {},                                   # nothing at all
            {"start": 1},                         # no end
            {"start": "abc", "end": 10},          # unparseable
            {"start": 10, "end": 5},              # end before start
            {"start": 10, "end": 10},             # zero length
            {"start": -5, "end": 10},             # negative start
            {"start": None, "end": 10},
        ],
    )
    def test_unusable_candidates_return_none(self, raw):
        assert score._parse_clip(raw) is None

    def test_score_is_clamped_to_the_range(self):
        assert score._parse_clip({"start": 0, "end": 20, "score": 500}).score == 100
        assert score._parse_clip({"start": 0, "end": 20, "score": -20}).score == 0

    def test_missing_score_is_zero_not_a_crash(self):
        assert score._parse_clip({"start": 0, "end": 20}).score == 0

    def test_title_falls_back_and_is_bounded(self):
        assert score._parse_clip({"start": 0, "end": 20}).title == "clip"
        long = score._parse_clip({"start": 0, "end": 20, "title": "x" * 500})
        assert len(long.title) <= 120


class TestExtractJson:
    def test_bare_array(self):
        assert score._extract_json_array('[{"a": 1}]') == [{"a": 1}]

    def test_fenced(self):
        assert score._extract_json_array('```json\n[{"a": 1}]\n```') == [{"a": 1}]

    def test_wrapped_in_prose(self):
        got = score._extract_json_array('Sure, here you go:\n[{"a": 1}]\nHope that helps.')
        assert got == [{"a": 1}]

    @pytest.mark.parametrize("key", ["clips", "results", "moments", "data"])
    def test_object_wrapper(self, key):
        assert score._extract_json_array('{"%s": [{"a": 1}]}' % key) == [{"a": 1}]

    @pytest.mark.parametrize("text", ["not json at all", "", "{}", "null"])
    def test_unrecoverable_returns_none(self, text):
        assert score._extract_json_array(text) is None


class TestSnapping:
    def test_snaps_to_a_nearby_sentence_edge(self, segments):
        clip = score._snap_to_sentences(Clip(2.1, 8.9, "t", 80), segments)
        assert clip.start == 2.0

    def test_does_not_drag_a_boundary_across_a_monologue(self):
        from pipeline.transcribe import Segment

        long_take = [Segment(0.0, 120.0, "one long take", [])]
        clip = score._snap_to_sentences(Clip(40.0, 70.0, "t", 80), long_take)
        # snapping to 0/120 would replace a 30s clip with a two-minute one
        assert (clip.start, clip.end) == (40.0, 70.0)

    def test_reverts_when_snapping_would_collapse_the_clip(self):
        from pipeline.transcribe import Segment

        segs = [Segment(0.0, 30.0, "a", []), Segment(30.0, 32.0, "b", [])]
        clip = score._snap_to_sentences(Clip(29.5, 51.0, "t", 80), segs)
        assert clip.duration > 0

    def test_no_segments_is_a_no_op(self):
        clip = score._snap_to_sentences(Clip(1.0, 30.0, "t", 80), [])
        assert (clip.start, clip.end) == (1.0, 30.0)


class TestDedupe:
    def test_keeps_the_higher_score_of_an_overlapping_pair(self):
        kept = score._dedupe([Clip(0, 40, "low", 60), Clip(2, 42, "high", 90)])
        assert [c.title for c in kept] == ["high"]

    def test_keeps_clips_that_merely_touch(self):
        kept = score._dedupe([Clip(0, 30, "a", 80), Clip(30, 60, "b", 70)])
        assert len(kept) == 2

    def test_keeps_clips_with_a_small_overlap(self):
        kept = score._dedupe([Clip(0, 30, "a", 80), Clip(28, 58, "b", 70)])
        assert len(kept) == 2

    def test_empty_input(self):
        assert score._dedupe([]) == []


class TestWindows:
    def test_short_transcript_is_one_window(self, segments):
        assert len(score._windows(segments)) == 1

    def test_long_transcript_is_split(self):
        from pipeline.transcribe import Segment

        segs = [Segment(i, i + 1, "x" * 400, []) for i in range(400)]
        windows = score._windows(segs)
        assert len(windows) > 1

    def test_windows_overlap_so_nothing_falls_between_them(self):
        from pipeline.transcribe import Segment

        segs = [Segment(i, i + 1, "y" * 400, []) for i in range(400)]
        windows = score._windows(segs)
        first_ends = windows[0][-1].start
        second_starts = windows[1][0].start
        assert second_starts < first_ends

    def test_empty(self):
        assert score._windows([]) == []


class TestWordsInRange:
    def test_words_are_rebased_to_the_clip_start(self, segments):
        words = score.words_in_range(segments, 3.0, 6.0)
        assert words and words[0].start >= 0
        assert all(0 <= w.start <= 3.1 for w in words)

    def test_range_outside_the_transcript_is_empty(self, segments):
        assert score.words_in_range(segments, 500.0, 520.0) == []

    def test_output_is_sorted(self, segments):
        words = score.words_in_range(segments, 0.0, 10.0)
        assert words == sorted(words, key=lambda w: w.start)


class TestFindClips:
    def test_empty_transcript_returns_nothing_without_calling_the_model(self):
        assert score.find_clips([]) == []

    def test_missing_key_raises_a_message_a_customer_can_act_on(self, segments, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(score.settings(), "anthropic_api_key", "", raising=False)
        with pytest.raises(score.ScoringError, match="ANTHROPIC_API_KEY"):
            score.find_clips(segments)


class TestProviderDispatch:
    """Either backend, one code path above the call.

    The windowing, JSON recovery, retries, snapping and dedupe are the parts
    that took work to get right. They must not fork per provider.
    """

    def _settings(self, monkeypatch, **env):
        import os

        for key in ("REELS_LLM_PROVIDER", "ANTHROPIC_API_KEY",
                    "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        from pipeline.config import settings

        return settings(refresh=True)

    def test_gemini_key_alone_selects_gemini(self, monkeypatch):
        assert self._settings(monkeypatch, GEMINI_API_KEY="g").resolve_provider() == "gemini"

    def test_anthropic_key_alone_selects_anthropic(self, monkeypatch):
        cfg = self._settings(monkeypatch, ANTHROPIC_API_KEY="a")
        assert cfg.resolve_provider() == "anthropic"

    def test_both_keys_keeps_anthropic_so_deployments_do_not_shift(self, monkeypatch):
        cfg = self._settings(monkeypatch, ANTHROPIC_API_KEY="a", GEMINI_API_KEY="g")
        assert cfg.resolve_provider() == "anthropic"

    def test_an_explicit_setting_overrides_the_keys(self, monkeypatch):
        cfg = self._settings(
            monkeypatch, ANTHROPIC_API_KEY="a", GEMINI_API_KEY="g",
            REELS_LLM_PROVIDER="gemini",
        )
        assert cfg.resolve_provider() == "gemini"

    def test_google_api_key_counts_as_a_gemini_key(self, monkeypatch):
        assert self._settings(monkeypatch, GOOGLE_API_KEY="g").resolve_provider() == "gemini"

    def test_no_key_at_all_still_names_a_way_forward(self, monkeypatch, segments):
        self._settings(monkeypatch)
        with pytest.raises(score.ScoringError, match="GEMINI_API_KEY"):
            score.find_clips(segments)

    def test_gemini_path_runs_the_shared_post_processing(self, monkeypatch, segments):
        """A Gemini reply goes through dedupe, snapping and the duration filter."""
        self._settings(monkeypatch, GEMINI_API_KEY="g")
        from pipeline import gemini

        # two overlapping candidates plus one too short to survive
        monkeypatch.setattr(gemini, "generate", lambda *a, **k: json.dumps([
            {"start": 0.5, "end": 30.0, "title": "lower", "score": 70},
            {"start": 1.0, "end": 31.0, "title": "higher", "score": 95},
            {"start": 40.0, "end": 42.0, "title": "too short", "score": 99},
        ]))
        clips = score.find_clips(segments, max_clips=5, min_score=50)
        assert [c.title for c in clips] == ["higher"]

    def test_a_gemini_failure_is_retried_then_reported(self, monkeypatch, segments):
        self._settings(monkeypatch, GEMINI_API_KEY="g", REELS_LLM_RETRIES="2")
        from pipeline import gemini

        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise gemini.GeminiError("overloaded")

        monkeypatch.setattr(gemini, "generate", boom)
        monkeypatch.setattr(score.time, "sleep", lambda s: None)
        with pytest.raises(score.ScoringError, match="overloaded"):
            score.find_clips(segments)
        assert len(calls) == 2

    def test_fenced_output_is_recovered_on_either_backend(self, monkeypatch, segments):
        self._settings(monkeypatch, GEMINI_API_KEY="g")
        from pipeline import gemini

        monkeypatch.setattr(gemini, "generate", lambda *a, **k:
                            '```json\n[{"start":0,"end":30,"title":"x","score":80}]\n```')
        assert len(score.find_clips(segments, min_score=50)) == 1
