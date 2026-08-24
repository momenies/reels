"""Step 1: audio -> word-level transcript.

Uses faster-whisper (CTranslate2). On CPU use model="small" or "medium";
on an NVIDIA GPU use "large-v3" with compute_type="float16".

Arabic note: large-v3 is markedly better than medium on Gulf/Egyptian
dialects. If you target Arabic seriously, budget for a GPU worker.

Loading a Whisper model costs seconds and hundreds of megabytes, so models are
cached per (size, device, compute_type). A server that transcribes back-to-back
jobs pays that cost once for the life of the process rather than once a job.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable

from .log import get

log = get("transcribe")


class TranscriptionError(RuntimeError):
    """Audio could not be extracted or decoded into a transcript."""


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def has_audio_stream(video: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def extract_audio(video: Path, out_wav: Path) -> Path:
    """16 kHz mono PCM — what every ASR model wants."""
    video, out_wav = Path(video), Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if not has_audio_stream(video):
        raise TranscriptionError(
            f"{video.name} has no audio track, so there is nothing to transcribe "
            f"and no way to pick clips from speech."
        )
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(out_wav),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not out_wav.exists():
        raise TranscriptionError(
            f"could not extract audio from {video.name}: {result.stderr.strip()[-300:]}"
        )
    return out_wav


_MODELS: dict[tuple, object] = {}
_MODELS_LOCK = threading.Lock()


def load_model(model_size: str, device: str, compute_type: str):
    """Cached WhisperModel. Loading one is expensive; reuse it."""
    key = (model_size, device, compute_type)
    with _MODELS_LOCK:
        if key not in _MODELS:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriptionError(
                    "faster-whisper is not installed: pip install -r requirements.txt"
                ) from exc
            log.info("loading whisper %s on %s (%s)", model_size, device, compute_type)
            try:
                _MODELS[key] = WhisperModel(
                    model_size, device=device, compute_type=compute_type
                )
            except Exception as exc:
                raise TranscriptionError(
                    f"could not load whisper model {model_size!r} on {device}: {exc}"
                ) from exc
        return _MODELS[key]


def transcribe(
    wav: Path,
    language: str | None = None,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    *,
    on_progress: Callable[[float], None] | None = None,
    duration: float | None = None,
) -> list[Segment]:
    model = load_model(model_size, device, compute_type)

    raw_segments, info = model.transcribe(
        str(wav),
        language=language,          # None = auto-detect
        word_timestamps=True,       # required for caption highlighting
        vad_filter=True,            # drops silence, big speedup on podcasts
        vad_parameters={"min_silence_duration_ms": 500},
        # Whisper's default carries the previous window's text into the next
        # prompt, which is what makes it loop a phrase forever on noisy audio.
        condition_on_previous_text=False,
    )

    total = duration or getattr(info, "duration", 0.0) or 0.0
    segments: list[Segment] = []
    for s in raw_segments:                       # generator: work happens here
        words = []
        for w in (s.words or []):
            text = (w.word or "").strip()
            if not text or w.start is None or w.end is None:
                continue
            words.append(Word(start=float(w.start), end=float(max(w.end, w.start)), text=text))
        segments.append(
            Segment(start=float(s.start), end=float(s.end),
                    text=(s.text or "").strip(), words=words)
        )
        if on_progress and total > 0:
            on_progress(max(0.0, min(1.0, float(s.end) / total)))

    log.info("language=%s segments=%d", getattr(info, "language", "?"), len(segments))
    if not segments:
        raise TranscriptionError(
            "no speech was detected in the audio — check the track is not silent"
        )
    return segments


def save(segments: list[Segment], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then move: a worker killed mid-write must not leave a half-parsed
    # cache behind that the next run happily loads.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load(path: Path) -> list[Segment]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Segment(
            start=d["start"],
            end=d["end"],
            text=d["text"],
            words=[Word(**w) for w in d.get("words", [])],
        )
        for d in data
    ]


def full_text(segments: list[Segment]) -> str:
    return " ".join(s.text for s in segments if s.text).strip()
