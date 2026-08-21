"""Step 0: a URL -> a local video file.

Wraps yt-dlp rather than reimplementing extraction. Sites change their
players constantly; yt-dlp tracks that, this repo should not try to.

On what you may fetch: the source has to be yours or licensed. Pulling other
creators' videos and reposting them is what gets accounts terminated and
platform API access revoked — and losing API access kills the whole product,
not one account. The caller is responsible for that call; this module only
performs the download it is asked for.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FetchError(RuntimeError):
    """yt-dlp could not produce a file."""


def available() -> bool:
    return shutil.which("yt-dlp") is not None


def _reason(stderr: str) -> str:
    """The one useful line out of yt-dlp's error output.

    Its failures carry a paragraph of boilerplate asking you to file an issue
    and upgrade. Shown verbatim in a job card that buries the actual cause,
    which is usually the first clause.
    """
    for line in reversed((stderr or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("ERROR:"):
            continue
        line = line[len("ERROR:"):].strip()
        for boilerplate in ("; please report this issue", " (caused by "):
            line = line.split(boilerplate)[0]
        return line.strip()
    return ""


def download(url: str, dest_dir: Path, *, max_height: int = 1080) -> Path:
    """Download `url` into `dest_dir` and return the file.

    Caps the height because the pipeline scales to 1080x1920 anyway — pulling
    a 4K source costs bandwidth and decode time for pixels that get thrown
    away. Prefers an already-muxed mp4 so there is no remux step.
    """
    if not available():
        raise FetchError("yt-dlp is not installed: pip install -r requirements-web.txt")

    dest_dir.mkdir(parents=True, exist_ok=True)
    fmt = (f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
           f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best")

    proc = subprocess.run(
        ["yt-dlp", "--no-playlist", "--no-progress", "--newline",
         "--restrict-filenames", "--merge-output-format", "mp4",
         "-f", fmt, "-o", str(dest_dir / "%(title).80s.%(ext)s"), url],
        capture_output=True, text=True,
    )
    for line in proc.stdout.splitlines():
        if line.strip():
            print(f"[fetch] {line.strip()}")

    if proc.returncode != 0:
        raise FetchError(_reason(proc.stderr) or f"yt-dlp exited {proc.returncode}")

    files = [p for p in dest_dir.iterdir() if p.is_file()]
    if not files:
        raise FetchError("yt-dlp reported success but wrote no file")

    video = max(files, key=lambda p: p.stat().st_size)
    print(f"[fetch] {video.name}  {video.stat().st_size // 1024} KB")
    return video
