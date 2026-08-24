"""The caption layer: escaping, timing sanity, presets, RTL detection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pipeline import captions


@dataclass
class W:
    start: float
    end: float
    text: str


def events(path):
    body = path.read_text(encoding="utf-8")
    return [l for l in body.splitlines() if l.startswith("Dialogue:")]


def ts(value: str) -> float:
    h, m, rest = value.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def span(line: str) -> tuple[float, float]:
    parts = line.split(",")
    return ts(parts[1]), ts(parts[2])


class TestEscape:
    @pytest.mark.parametrize(
        "raw",
        ["{\\pos(0,0)}", "a{b", "}c", "back\\slash", "{\\c&HFF0000&}red"],
    )
    def test_no_override_syntax_survives(self, raw):
        out = captions.escape(raw)
        assert "{" not in out and "}" not in out and "\\" not in out

    def test_ordinary_text_is_untouched(self):
        assert captions.escape("  hello, world!  ") == "hello, world!"

    def test_arabic_is_untouched(self):
        assert captions.escape("مرحبا بالعالم") == "مرحبا بالعالم"

    def test_newlines_collapse(self):
        assert "\n" not in captions.escape("two\nlines")

    def test_hostile_word_cannot_reach_the_event_line(self, tmp_path):
        path = captions.build_ass(
            [W(0, 0.5, "{\\pos(0,0)}"), W(0.5, 1.0, "safe")], tmp_path / "x.ass"
        )
        body = "".join(events(path))
        assert "\\pos(0,0)}" not in body.replace("\\pos(540,1400)", "")
        assert "safe" in body


class TestTiming:
    def test_backwards_word_never_yields_a_negative_event(self, tmp_path):
        path = captions.build_ass([W(1.0, 0.2, "a"), W(0.4, 0.4, "b")], tmp_path / "x.ass")
        for line in events(path):
            start, end = span(line)
            assert end > start

    def test_identical_timestamps_get_a_minimum_length(self, tmp_path):
        path = captions.build_ass([W(2.0, 2.0, "only")], tmp_path / "x.ass")
        start, end = span(events(path)[0])
        assert end - start >= captions.MIN_EVENT

    def test_empty_words_are_dropped(self, tmp_path):
        path = captions.build_ass(
            [W(0, 0.5, "   "), W(0.5, 1.0, "kept")], tmp_path / "x.ass"
        )
        body = "".join(events(path))
        assert "kept" in body


class TestStyles:
    def test_every_preset_builds(self, tmp_path):
        for name in captions.STYLES:
            path = captions.build_ass(
                [W(0, 0.5, "one"), W(0.5, 1.0, "two")], tmp_path / f"{name}.ass",
                style=name,
            )
            assert events(path)

    def test_unknown_style_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="unknown caption style"):
            captions.get_style("does-not-exist")

    def test_uppercase_preset_uppercases_latin(self, tmp_path):
        path = captions.build_ass([W(0, 0.5, "quiet")], tmp_path / "x.ass", style="beast")
        assert "QUIET" in "".join(events(path))

    def test_uppercase_preset_leaves_arabic_alone(self, tmp_path):
        path = captions.build_ass([W(0, 0.5, "مرحبا")], tmp_path / "x.ass", style="beast")
        assert "مرحبا" in "".join(events(path))

    def test_explicit_kwargs_win_over_the_preset(self, tmp_path):
        path = captions.build_ass(
            [W(0, 0.5, "x")], tmp_path / "x.ass", style="beast", size=42
        )
        assert ",42," in path.read_text(encoding="utf-8")


class TestDirection:
    @pytest.mark.parametrize("text", ["مرحبا بالعالم", "هذا اختبار"])
    def test_arabic_is_rtl(self, text):
        assert captions.is_rtl(text)

    @pytest.mark.parametrize("text", ["hello world", "", "12345", "Bonjour"])
    def test_latin_is_not_rtl(self, text):
        assert not captions.is_rtl(text)

    def test_rtl_emits_one_event_per_line_not_per_word(self, tmp_path):
        words = [W(i * 0.4, i * 0.4 + 0.35, w) for i, w in enumerate("مرحبا بك في".split())]
        path = captions.build_ass(words, tmp_path / "ar.ass")
        # line-level pop: three words, one event — intra-line tags would break shaping
        assert len(events(path)) == 1


class TestFonts:
    def test_missing_font_raises(self, tmp_path):
        with pytest.raises(captions.MissingFontError):
            captions.build_ass([W(0, 1, "x")], tmp_path / "x.ass", fonts_dir=tmp_path)

    def test_unknown_font_name_raises(self):
        with pytest.raises(captions.MissingFontError, match="not a known font"):
            captions.require_font("Comic Sans")

    def test_empty_track_still_checks_the_font(self, tmp_path):
        with pytest.raises(captions.MissingFontError):
            captions.build_ass([], tmp_path / "x.ass", fonts_dir=tmp_path)


class TestHook:
    def test_picks_the_arabic_face_for_an_arabic_title(self, tmp_path):
        path = captions.build_hook("عنوان جذاب", 4.0, tmp_path / "h.ass")
        assert "Tajawal ExtraBold" in path.read_text(encoding="utf-8")

    def test_picks_the_latin_face_for_a_latin_title(self, tmp_path):
        path = captions.build_hook("A catchy title", 4.0, tmp_path / "h.ass")
        assert "Montserrat ExtraBold" in path.read_text(encoding="utf-8")

    def test_long_title_wraps(self, tmp_path):
        path = captions.build_hook(
            "a very long hook title that would otherwise run off the frame entirely",
            4.0, tmp_path / "h.ass",
        )
        assert "\\N" in "".join(events(path))

    def test_zero_duration_still_produces_a_visible_event(self, tmp_path):
        path = captions.build_hook("x", 0.0, tmp_path / "h.ass")
        start, end = span(events(path)[0])
        assert end > start
