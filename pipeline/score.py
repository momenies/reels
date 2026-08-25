"""Step 2: transcript -> ranked clip candidates.

This is the part users actually pay for. Cropping is a commodity;
picking the right 40 seconds is not.

Strategy: feed the LLM a timestamped transcript and ask for self-contained
moments, then snap the returned boundaries to real sentence edges so clips
never start or end mid-word.

Three things a naive version of this gets wrong, all of which cost money
*after* the transcript has already been paid for:

* A three-hour podcast does not fit in one prompt. Long transcripts are
  scored in overlapping windows and the results merged.
* The model returns overlapping ranges. Without a dedupe pass the customer
  gets the same moment three times with three different titles.
* Snapping to the nearest segment edge is unbounded — a boundary the model
  put inside a long monologue can snap thirty seconds away and turn a good
  clip into a bad one. Snapping is now capped, and falls back to the model's
  own boundary.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field

from . import gemini
from .config import settings
from .log import get
from .transcribe import Segment, Word

log = get("score")

#: Roughly four characters to a token, and we leave room for the reply.
#: A window of ~60k characters is comfortably inside any current context
#: window while still giving the model enough material to judge a moment.
WINDOW_CHARS = 60_000
WINDOW_OVERLAP_CHARS = 4_000

MIN_DURATION = 15.0
MAX_DURATION = 90.0
SNAP_TOLERANCE = 2.5     # seconds; beyond this the model's boundary wins
OVERLAP_TOLERANCE = 0.5  # fraction of the shorter clip that may overlap

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


class ScoringError(RuntimeError):
    """Clip selection could not run at all — as opposed to finding nothing."""


@dataclass
class Clip:
    start: float
    end: float
    title: str
    score: int
    reason: str = ""
    keywords: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "title": self.title,
            "score": self.score,
            "reason": self.reason,
        }


def _transcript_text(segments: list[Segment]) -> str:
    return "\n".join(f"[{s.start:.1f}] {s.text}" for s in segments)


def _windows(segments: list[Segment]) -> list[list[Segment]]:
    """Split a long transcript into overlapping windows.

    The overlap matters: a clip that straddles a window boundary would
    otherwise be invisible to both halves.
    """
    if not segments:
        return []
    total = sum(len(s.text) + 12 for s in segments)
    if total <= WINDOW_CHARS:
        return [segments]

    windows: list[list[Segment]] = []
    cur: list[Segment] = []
    size = 0
    for seg in segments:
        cur.append(seg)
        size += len(seg.text) + 12
        if size >= WINDOW_CHARS:
            windows.append(cur)
            # rewind by the overlap so a straddling moment is seen twice
            back, tail = 0, []
            for s in reversed(cur):
                if back >= WINDOW_OVERLAP_CHARS:
                    break
                tail.append(s)
                back += len(s.text) + 12
            cur = list(reversed(tail))
            size = back
    if len(cur) > 1 or not windows:
        windows.append(cur)
    log.info("transcript split into %d windows", len(windows))
    return windows


def _parse_clip(raw: dict) -> Clip | None:
    """One candidate from the model -> Clip, or None if it is unusable.

    The model is asked for a fixed shape but is not bound to it: extra keys,
    stringified numbers and missing fields all show up in practice, and
    Clip(**raw) turns any of them into a crash after the transcript has
    already been paid for.
    """
    try:
        start = float(raw["start"])
        end = float(raw["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (end > start >= 0) or end != end or start != start:  # NaN-safe
        return None
    try:
        score = int(float(raw.get("score", 0)))
    except (TypeError, ValueError):
        score = 0
    return Clip(
        start=start,
        end=end,
        title=str(raw.get("title") or "clip").strip()[:120],
        score=max(0, min(100, score)),
        reason=str(raw.get("reason") or "").strip()[:400],
    )


def _extract_json_array(text: str) -> list | None:
    """Pull the JSON array out of a reply, fences and preamble included."""
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Models occasionally wrap the array in prose or an object. Take the
        # outermost bracketed run rather than giving up on a paid call.
        first, last = text.find("["), text.rfind("]")
        if first == -1 or last <= first:
            return None
        try:
            data = json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            return None
    if isinstance(data, dict):
        for key in ("clips", "results", "moments", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return None
    return data if isinstance(data, list) else None


def _snap_to_sentences(clip: Clip, segments: list[Segment]) -> Clip:
    """Pull boundaries out to the nearest segment edge so speech isn't cut.

    Capped by ``SNAP_TOLERANCE``: a boundary that has no sentence edge nearby
    sits inside a long monologue, and dragging it to the far edge of that
    monologue is worse than leaving it where the model put it.
    """
    if not segments:
        return clip
    original = (clip.start, clip.end)
    start = min((s.start for s in segments), key=lambda t: abs(t - clip.start))
    end = min((s.end for s in segments), key=lambda t: abs(t - clip.end))
    if abs(start - clip.start) <= SNAP_TOLERANCE:
        clip.start = start
    if abs(end - clip.end) <= SNAP_TOLERANCE:
        clip.end = end
    collapsed = clip.duration <= 0 or (
        clip.duration < MIN_DURATION <= original[1] - original[0]
    )
    if collapsed:
        # snapping folded the clip in on itself — keep the model's boundaries
        clip.start, clip.end = original
    return clip


def _overlaps(a: Clip, b: Clip) -> bool:
    inter = min(a.end, b.end) - max(a.start, b.start)
    if inter <= 0:
        return False
    return inter / min(a.duration, b.duration) > OVERLAP_TOLERANCE


def _dedupe(clips: list[Clip]) -> list[Clip]:
    """Keep the highest-scoring clip out of each overlapping group.

    Windowed scoring guarantees duplicates (the overlap is seen twice), and
    even a single call returns near-identical ranges regularly.
    """
    kept: list[Clip] = []
    for clip in sorted(clips, key=lambda c: (-c.score, c.start)):
        if not any(_overlaps(clip, k) for k in kept):
            kept.append(clip)
    return kept


def _anthropic_caller():
    """A ``(prompt) -> text`` callable backed by Claude."""
    cfg = settings()
    key = cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ScoringError(
            "ANTHROPIC_API_KEY is not set. Clip selection is the one step that "
            "needs a model — copy .env.example to .env and add a key from "
            "https://console.anthropic.com/, or set GEMINI_API_KEY instead."
        )
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ScoringError(
            "the 'anthropic' package is not installed: pip install -r requirements.txt"
        ) from exc

    client = Anthropic(api_key=key)

    def call(prompt: str) -> str:
        resp = client.messages.create(
            model=cfg.model,
            max_tokens=4000,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    return call, cfg.model


def _gemini_caller():
    """A ``(prompt) -> text`` callable backed by Gemini."""
    cfg = settings()
    try:
        gemini.api_key(cfg.gemini_api_key or None)
    except gemini.GeminiError as exc:
        raise ScoringError(str(exc)) from None

    def call(prompt: str) -> str:
        return gemini.generate(
            SYSTEM, prompt, model=cfg.gemini_model, key=cfg.gemini_api_key or None
        )

    return call, cfg.gemini_model


def _caller():
    """Whichever backend is configured, behind one signature.

    Only the request differs between providers. Windowing, JSON recovery,
    retries, snapping and dedupe are the parts that took work to get right,
    and they are worth exactly as much on one backend as the other — so they
    live above this line, not inside it.
    """
    provider = settings().resolve_provider()
    call, model = _gemini_caller() if provider == "gemini" else _anthropic_caller()
    log.info("clip selection via %s (%s)", provider, model)
    return call


def _ask(call, prompt: str) -> list[dict]:
    """One scoring call, retried on the transient failures that do happen."""
    cfg = settings()
    last: Exception | None = None
    for attempt in range(cfg.llm_max_retries):
        try:
            text = call(prompt)
        except Exception as exc:  # overloaded, rate limited, connection reset
            last = exc
            if attempt == cfg.llm_max_retries - 1:
                break
            delay = 2**attempt + random.random()
            log.warning("scoring call failed (%s); retrying in %.1fs", exc, delay)
            time.sleep(delay)
            continue

        raw = _extract_json_array(text)
        if raw is None:
            last = ValueError("model did not return a JSON array")
            log.warning("model did not return valid JSON:\n%s", text[:400])
            if attempt == cfg.llm_max_retries - 1:
                break
            continue
        return [c for c in raw if isinstance(c, dict)]

    raise ScoringError(f"clip selection failed after {cfg.llm_max_retries} attempts: {last}")


def find_clips(
    segments: list[Segment],
    max_clips: int = 5,
    min_score: int = 60,
    *,
    video_duration: float | None = None,
) -> list[Clip]:
    """Rank the moments worth cutting. Raises ScoringError if it cannot run."""
    if not segments:
        log.warning("empty transcript — nothing to score")
        return []

    call = _caller()
    candidates: list[Clip] = []
    windows = _windows(segments)

    for i, window in enumerate(windows, 1):
        if len(windows) > 1:
            log.info("scoring window %d/%d", i, len(windows))
        prompt = (
            f"Find up to {max_clips} clips in this transcript.\n\n"
            f"{_transcript_text(window)}"
        )
        for raw in _ask(call, prompt):
            clip = _parse_clip(raw)
            if clip is not None:
                candidates.append(clip)

    dropped = 0
    clips: list[Clip] = []
    hard_end = video_duration if video_duration else None
    for clip in candidates:
        if clip.score < min_score:
            continue
        clip = _snap_to_sentences(clip, segments)
        if hard_end:
            clip.start = max(0.0, min(clip.start, hard_end - MIN_DURATION))
            clip.end = min(clip.end, hard_end)
        if not (MIN_DURATION <= clip.duration <= MAX_DURATION):
            dropped += 1
            continue
        clips.append(clip)

    clips = _dedupe(clips)
    clips.sort(key=lambda c: c.score, reverse=True)

    if dropped:
        log.info("dropped %d candidate(s) outside %g-%gs", dropped, MIN_DURATION, MAX_DURATION)
    log.info(
        "kept %d clip(s) above %d from %d candidate(s)",
        min(len(clips), max_clips), min_score, len(candidates),
    )
    return clips[:max_clips]


def words_in_range(segments: list[Segment], start: float, end: float) -> list[Word]:
    """Words for a clip, re-based so the clip starts at t=0."""
    out: list[Word] = []
    for s in segments:
        if s.end < start - 0.05 or s.start > end + 0.05:
            continue  # segment is entirely outside the clip
        for w in s.words:
            if w.start >= start - 0.05 and w.end <= end + 0.05:
                out.append(Word(start=w.start - start, end=w.end - start, text=w.text))
    out.sort(key=lambda w: w.start)
    return out
