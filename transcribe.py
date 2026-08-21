"""Step 1: audio -> word-level transcript.

Uses faster-whisper (CTranslate2). On CPU use model="small" or "medium";
on an NVIDIA GPU use "large-v3" with compute_type="float16".

Arabic note: large-v3 is markedly better than medium on Gulf/Egyptian
dialects. If you target Arabic seriously, budget for a GPU worker.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


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
    words: list[Word]


def extract_audio(video: Path, out_wav: Path) -> Path:
    """16 kHz mono PCM — what every ASR model wants."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(out_wav),
        ],
        check=True,
    )
    return out_wav


def transcribe(
    wav: Path,
    language: str | None = None,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[Segment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(
        str(wav),
        language=language,          # None = auto-detect
        word_timestamps=True,       # required for caption highlighting
        vad_filter=True,            # drops silence, big speedup on podcasts
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments: list[Segment] = []
    for s in raw_segments:
        words = [
            Word(start=w.start, end=w.end, text=w.word.strip())
            for w in (s.words or [])
            if w.word.strip()
        ]
        segments.append(
            Segment(start=s.start, end=s.end, text=s.text.strip(), words=words)
        )

    print(f"[transcribe] language={info.language} segments={len(segments)}")
    return segments


def save(segments: list[Segment], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load(path: Path) -> list[Segment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Segment(
            start=d["start"],
            end=d["end"],
            text=d["text"],
            words=[Word(**w) for w in d["words"]],
        )
        for d in data
    ]
