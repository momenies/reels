"""Step 2: transcript -> ranked clip candidates.

This is the part users actually pay for. Cropping is a commodity;
picking the right 40 seconds is not.

Strategy: feed the LLM a timestamped transcript and ask for self-contained
moments, then snap the returned boundaries to real sentence edges so clips
never start or end mid-word.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from anthropic import Anthropic

from .transcribe import Segment

MODEL = "claude-sonnet-5"

SYSTEM = """You select short-form clips from long-form video transcripts.

A good clip:
- is SELF-CONTAINED: understandable with zero context from the rest of the video
- opens on a hook in the first 3 seconds (a claim, a question, a number, tension)
- has a payoff or punchline before it ends
- is 20-60 seconds long
- starts and ends on a complete sentence

A bad clip: setup with no payoff, inside jokes, rambling, mid-thought starts.

Score 0-100 on standalone virality. Be harsh. A typical hour of talk contains
maybe 3-6 clips above 70. Returning fewer good clips beats padding the list.

Write "title" as a scroll-stopping hook in the SAME LANGUAGE as the transcript.
Under 60 characters. No clickbait punctuation spam.

Return ONLY a JSON array, no prose, no markdown fences:
[{"start": 123.4, "end": 168.0, "title": "...", "score": 82, "reason": "..."}]"""


@dataclass
class Clip:
    start: float
    end: float
    title: str
    score: int
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def _transcript_text(segments: list[Segment]) -> str:
    return "\n".join(f"[{s.start:.1f}] {s.text}" for s in segments)


def _parse_clip(raw: dict) -> Clip | None:
    """One candidate from the model -> Clip, or None if it is unusable.

    The model is asked for a fixed shape but is not bound to it: extra keys,
    stringified numbers and missing fields all show up in practice, and
    Clip(**raw) turns any of them into a crash after the transcript has
    already been paid for.
    """
    try:
        return Clip(
            start=float(raw["start"]),
            end=float(raw["end"]),
            title=str(raw.get("title") or "clip").strip(),
            score=int(float(raw.get("score", 0))),
            reason=str(raw.get("reason") or "").strip(),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _snap_to_sentences(clip: Clip, segments: list[Segment]) -> Clip:
    """Pull boundaries out to the nearest segment edge so speech isn't cut."""
    starts = [s.start for s in segments]
    ends = [s.end for s in segments]
    clip.start = min(starts, key=lambda t: abs(t - clip.start))
    clip.end = min(ends, key=lambda t: abs(t - clip.end))
    return clip


def find_clips(
    segments: list[Segment],
    max_clips: int = 5,
    min_score: int = 60,
) -> list[Clip]:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Find up to {max_clips} clips in this transcript.\n\n"
                    f"{_transcript_text(segments)}"
                ),
            }
        ],
    )

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        print("[score] model did not return valid JSON:\n", text[:500])
        return []

    if not isinstance(raw, list):
        print("[score] expected a JSON array, got", type(raw).__name__)
        return []

    parsed = [_parse_clip(c) for c in raw if isinstance(c, dict)]
    dropped = sum(1 for c in parsed if c is None)
    if dropped:
        print(f"[score] dropped {dropped} malformed candidate(s)")

    clips = [
        _snap_to_sentences(c, segments)
        for c in parsed
        if c is not None and c.score >= min_score
    ]
    clips = [c for c in clips if 15 <= c.duration <= 90]
    clips.sort(key=lambda c: c.score, reverse=True)

    print(f"[score] kept {len(clips)} clips above {min_score}")
    return clips[:max_clips]


def words_in_range(segments: list[Segment], start: float, end: float):
    """Words for a clip, re-based so the clip starts at t=0."""
    out = []
    for s in segments:
        for w in s.words:
            if w.start >= start - 0.05 and w.end <= end + 0.05:
                out.append(type(w)(start=w.start - start, end=w.end - start, text=w.text))
    return out
