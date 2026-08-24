"""Shared fixtures.

The suite never touches the network, never loads a model, and never needs an
API key — the two stages that would (Whisper and clip selection) are the two
that get stubbed. What is left is the logic that actually breaks in practice:
parsing, clamping, dedupe, path safety and job state transitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.transcribe import Segment, Word  # noqa: E402


@pytest.fixture
def segments() -> list[Segment]:
    """Ten seconds of speech, one segment a second, four words each."""
    out = []
    for i in range(10):
        words = [
            Word(start=i + j * 0.2, end=i + j * 0.2 + 0.18, text=f"w{i}{j}")
            for j in range(4)
        ]
        out.append(Segment(start=float(i), end=i + 0.98, text=f"sentence {i}", words=words))
    return out


@pytest.fixture
def store(tmp_path):
    from server.store import Store

    return Store(tmp_path / "jobs.db")
