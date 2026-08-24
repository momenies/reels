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
    """yt-dlp could not produce a file.

    ``kind`` names the cause when it is one of the recognised ones, so a
    caller can present it in its own words instead of re-matching strings.
    """

    def __init__(self, message: str, kind: str = ""):
        super().__init__(message)
        self.kind = kind


def available() -> bool:
    return shutil.which("yt-dlp") is not None


#: Failures worth translating, because the raw text sends you somewhere
#: unhelpful. Matched case-insensitively against yt-dlp's ERROR line.
_KNOWN_FAILURES = (
    ("confirm you\u2019re not a bot", "BOT_CHECK"),
    ("confirm you're not a bot", "BOT_CHECK"),
    ("sign in to confirm", "BOT_CHECK"),
    ("private video", "PRIVATE"),
    ("members-only", "PRIVATE"),
    ("join this channel", "PRIVATE"),
    ("video unavailable", "UNAVAILABLE"),
    ("removed by the uploader", "UNAVAILABLE"),
    ("available in your country", "GEOBLOCKED"),
    ("blocked it in your country", "GEOBLOCKED"),
    ("this live event", "LIVE"),
    ("live stream recording is not available", "LIVE"),
)

_ADVICE = {
    "BOT_CHECK": (
        "YouTube refused the download: it treats this machine as automated "
        "traffic. Cloud hosts — Codespaces, most VPSs — are blocked by "
        "default and no yt-dlp flag reliably changes that. For your own "
        "video, download the original from YouTube Studio and upload the "
        "file here; the master is better quality than YouTube's re-encode "
        "anyway."
    ),
    "PRIVATE": "That video is private or members-only, so it cannot be fetched.",
    "UNAVAILABLE": "That video is unavailable — removed, or the link is wrong.",
    "GEOBLOCKED": "That video is not available from this machine's region.",
    "LIVE": "That is a live stream. Wait for the recording to be published.",
}


def _reason(stderr: str) -> tuple[str, str]:
    """The useful part of yt-dlp's error output.

    Its failures carry a paragraph of boilerplate asking you to file an issue
    and upgrade, which buries the cause. Worse, the bot-check failure points
    at cookie flags — the wrong fix for a datacenter IP, and one that risks
    the account it borrows. Known causes get an explanation instead.
    """
    for line in reversed((stderr or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("ERROR:"):
            continue
        line = line[len("ERROR:"):].strip()
        lowered = line.lower()
        for needle, kind in _KNOWN_FAILURES:
            if needle in lowered:
                return _ADVICE[kind], kind
        for boilerplate in ("; please report this issue", " (caused by "):
            line = line.split(boilerplate)[0]
        return line.strip(), ""
    return "", ""


def download(url: str, dest_dir: Path, *, max_height: int = 1080) -> Path:
    """Download `url` into `dest_dir` and return the file.

    Caps the height because the pipeline scales to 1080x1920 anyway — pulling
    a 4K source costs bandwidth and decode time for pixels that get thrown
    away. Prefers an already-muxed mp4 so there is no remux step.
    """
    if not available():
        raise FetchError("yt-dlp is not installed: pip install -r requirements-web.txt",
                         "NOT_INSTALLED")

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
        message, kind = _reason(proc.stderr)
        raise FetchError(message or f"yt-dlp exited {proc.returncode}", kind)

    files = [p for p in dest_dir.iterdir() if p.is_file()]
    if not files:
        raise FetchError("yt-dlp reported success but wrote no file", "EMPTY")

    video = max(files, key=lambda p: p.stat().st_size)
    print(f"[fetch] {video.name}  {video.stat().st_size // 1024} KB")
    return video
