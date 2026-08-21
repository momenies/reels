"""Step 5: cut, reframe, caption, encode.

One ffmpeg pass. Putting -ss before -i makes the seek fast and rebases
timestamps to zero, which is what the sendcmd and ASS timings assume.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def render(
    src: Path,
    out: Path,
    *,
    start: float,
    duration: float,
    crop_w: int,
    crop_x0: int,
    sendcmd_file: Path | None,
    ass_file: Path | None,
    fonts_dir: Path | None = None,
    gpu: bool = False,
    crf: int = 20,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)

    chain = []
    if sendcmd_file:
        chain.append(f"sendcmd=f={_esc(sendcmd_file)}")
    chain.append(f"crop=w={crop_w}:h=ih:x={crop_x0}:y=0")
    chain.append("scale=1080:1920:flags=lanczos")
    chain.append("setsar=1")
    if ass_file:
        f = f"ass=filename={_esc(ass_file)}"
        if fonts_dir and fonts_dir.exists():
            f += f":fontsdir={_esc(fonts_dir)}"
        chain.append(f)

    if gpu:
        vcodec = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(crf)]
    else:
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-stats",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src),
        "-vf", ",".join(chain),
        *vcodec,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out),
    ]
    print("[render]", " ".join(shlex.quote(c) for c in cmd[:12]), "...")
    subprocess.run(cmd, check=True)
    return out


def thumbnail(clip: Path, out: Path, at: float = 0.5) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(at), "-i", str(clip),
         "-frames:v", "1", "-q:v", "3", str(out)],
        check=True,
    )
    return out


def _esc(p: Path) -> str:
    """ffmpeg filter args: escape : and \\ and '."""
    return str(p).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
