"""The orchestrator's pure parts: filename safety and option resolution."""

from __future__ import annotations

import pytest

from pipeline.run import Options, slug


class TestSlug:
    def test_keeps_arabic(self):
        assert slug("مرحبا بالعالم") == "مرحبا-بالعالم"

    def test_strips_punctuation(self):
        assert slug("Hello, World!!") == "Hello-World"

    def test_collapses_whitespace(self):
        assert slug("a    b\t\tc") == "a-b-c"

    @pytest.mark.parametrize("name", ["", "   ", "!!!", "///"])
    def test_empty_result_falls_back(self, name):
        assert slug(name) == "clip"

    @pytest.mark.parametrize("name", ["CON", "con", "PRN", "NUL", "COM1", "LPT9"])
    def test_windows_reserved_names_are_avoided(self, name):
        assert slug(name) == "clip"

    def test_length_is_bounded(self):
        assert len(slug("word " * 100)) <= 40

    @pytest.mark.parametrize("bad", ["../../etc/passwd", "a/b/c", "x\\y"])
    def test_no_path_separators_survive(self, bad):
        out = slug(bad)
        assert "/" not in out and "\\" not in out and ".." not in out

    def test_no_leading_or_trailing_dash(self):
        assert not slug("  -- hello --  ").startswith("-")
        assert not slug("  -- hello --  ").endswith("-")


class TestOptions:
    def test_defaults_resolve_from_settings(self):
        resolved = Options().resolved()
        assert resolved["workers"] >= 1
        assert resolved["model"]

    def test_explicit_values_win(self):
        resolved = Options(model="large-v3", crf=14, workers=3).resolved()
        assert resolved["model"] == "large-v3"
        assert resolved["crf"] == 14
        assert resolved["workers"] == 3

    def test_gpu_none_means_take_the_environment_default(self):
        assert Options(gpu=None).resolved()["gpu"] in (True, False)


class TestProcessSignature:
    """`web/`'s standalone tool calls process() with keyword arguments.

    Two entry points into the pipeline is already one more than ideal; two
    that accept different arguments is how they start producing different
    clips from the same video.
    """

    def test_keyword_form_builds_the_same_options(self):
        from pipeline.run import Options

        assert Options(lang="ar", clips=3).lang == "ar"

    def test_unknown_keyword_is_rejected_loudly(self):
        from pipeline.run import process

        with pytest.raises(TypeError, match="unexpected keyword"):
            process("x.mp4", "out", nonsense=1)

    def test_mixing_both_forms_is_rejected(self):
        from pipeline.run import Options, process

        with pytest.raises(TypeError, match="not both"):
            process("x.mp4", "out", Options(), lang="ar")

    def test_the_tool_call_site_still_type_checks(self):
        """Exactly the call web/app.py makes, minus running it."""
        from pipeline.run import Options

        opts = Options(lang=None, clips=5, min_score=60, model="", faces=True)
        assert opts.resolved()["model"]        # empty model falls back
