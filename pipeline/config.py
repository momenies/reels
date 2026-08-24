"""Runtime settings, read once from the environment.

Everything the pipeline needs to be told about its machine lives here rather
than being threaded through five call signatures. Defaults are the ones that
work on a plain CPU box, so an unconfigured clone still runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
FONTS_DIR = ASSETS / "fonts"
MODELS_DIR = ASSETS / "models"


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    """Process-wide knobs. Instantiate once via :func:`settings`."""

    # --- LLM ---------------------------------------------------------------
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("REELS_MODEL", "claude-sonnet-5")
    )
    llm_max_retries: int = field(default_factory=lambda: _int("REELS_LLM_RETRIES", 3))

    # --- transcription -----------------------------------------------------
    whisper_model: str = field(
        default_factory=lambda: os.environ.get("REELS_WHISPER_MODEL", "small")
    )
    device: str = field(default_factory=lambda: os.environ.get("REELS_DEVICE", "cpu"))
    compute_type: str = field(
        default_factory=lambda: os.environ.get("REELS_COMPUTE_TYPE", "")
    )

    # --- rendering ---------------------------------------------------------
    gpu: bool = field(default_factory=lambda: _bool("REELS_GPU"))
    crf: int = field(default_factory=lambda: _int("REELS_CRF", 20))
    preset: str = field(
        default_factory=lambda: os.environ.get("REELS_X264_PRESET", "veryfast")
    )
    render_workers: int = field(default_factory=lambda: _int("REELS_RENDER_WORKERS", 0))
    loudnorm: bool = field(default_factory=lambda: _bool("REELS_LOUDNORM", True))

    # --- reframing ---------------------------------------------------------
    sample_fps: float = field(default_factory=lambda: _float("REELS_SAMPLE_FPS", 4.0))
    detect_width: int = field(default_factory=lambda: _int("REELS_DETECT_WIDTH", 640))

    # --- server ------------------------------------------------------------
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("REELS_DATA_DIR", str(ROOT / "data"))
        ).expanduser()
    )
    max_upload_mb: int = field(default_factory=lambda: _int("REELS_MAX_UPLOAD_MB", 2048))
    job_workers: int = field(default_factory=lambda: _int("REELS_JOB_WORKERS", 1))
    api_key: str = field(default_factory=lambda: os.environ.get("REELS_API_KEY", ""))
    retention_hours: int = field(
        default_factory=lambda: _int("REELS_RETENTION_HOURS", 72)
    )

    def whisper_compute_type(self) -> str:
        """int8 on CPU, float16 on CUDA — unless told otherwise."""
        if self.compute_type:
            return self.compute_type
        return "float16" if self.device == "cuda" else "int8"

    def effective_render_workers(self) -> int:
        """Clip renders are ffmpeg processes; more than a few thrash the box."""
        if self.render_workers > 0:
            return self.render_workers
        return max(1, min(4, (os.cpu_count() or 2) // 2))


_cached: Settings | None = None


def settings(refresh: bool = False) -> Settings:
    global _cached
    if _cached is None or refresh:
        _cached = Settings()
    return _cached
