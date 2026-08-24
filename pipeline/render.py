"""Step 5: cut, reframe, caption, encode.

One ffmpeg pass. Putting -ss before -i makes the seek fast and rebases
timestamps to zero, which is what the sendcmd and ASS timings assume.

Two things here exist because the output is going straight to a social feed:
loudness normalisation (TikTok, Reels and Shorts all normalise on ingest, and
a clip that arrives quiet stays quiet next to everything else), and an
explicit stream map so a source with no audio track produces a silent clip
instead of an ffmpeg error halfway through a paid job.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Callable

from .log import get

log = get("render")

#: What the platforms normalise to. Matching it on the way out means the clip
#: is not turned down again on ingest.
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"

OUT_W, OUT_H = 1080, 1920


class RenderError(RuntimeError):
    """ffmpeg refused to produce the clip."""


class RenderedNothingError(RenderError):
    """ffmpeg exited 0 but produced a clip with no frames."""


def build_filter_chain(
    *,
    crop_w: int,
    crop_x0: int,
    sendcmd_file: Path | None,
    ass_file: Path | None,
    hook_file: Path | None = None,
    fonts_dir: Path | None = None,
) -> str:
    chain = []
    if sendcmd_file:
        chain.append(f"sendcmd=f={_esc(sendcmd_file)}")
    chain.append(f"crop=w={crop_w}:h=ih:x={crop_x0}:y=0")
    chain.append(f"scale={OUT_W}:{OUT_H}:flags=lanczos")
    chain.append("setsar=1")
    for subs in (ass_file, hook_file):
        if not subs:
            continue
        f = f"ass=filename={_esc(subs)}"
        if fonts_dir and Path(fonts_dir).exists():
            f += f":fontsdir={_esc(fonts_dir)}"
        chain.append(f)
    return ",".join(chain)


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
    hook_file: Path | None = None,
    fonts_dir: Path | None = None,
    gpu: bool = False,
    crf: int = 20,
    preset: str | None = None,
    has_audio: bool = True,
    loudnorm: bool = False,
    threads: int = 0,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    vf = build_filter_chain(
        crop_w=crop_w,
        crop_x0=crop_x0,
        sendcmd_file=sendcmd_file,
        ass_file=ass_file,
        hook_file=hook_file,
        fonts_dir=fonts_dir,
    )

    if gpu:
        vcodec = ["-c:v", "h264_nvenc", "-preset", preset or "p5", "-cq", str(crf),
                  "-rc", "vbr", "-b:v", "0"]
    else:
        vcodec = ["-c:v", "libx264", "-preset", preset or "veryfast", "-crf", str(crf)]
        if crf > 0:
            # High profile is what phones decode in hardware — but it cannot
            # express lossless, and crf 0 is how the smoke test gets bit-exact
            # comparison renders.
            vcodec += ["-profile:v", "high", "-level", "4.1"]

    # Inputs first: an -i that appears after output options is a syntax error.
    inputs = [
        "-accurate_seek",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src),
    ]
    if has_audio:
        amap = ["-map", "0:a:0?"]
        acodec = ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]
        afilter = ["-af", LOUDNORM] if loudnorm else []
    else:
        # A silent track beats no track: several platforms reject audioless uploads.
        inputs += ["-f", "lavfi", "-t", f"{duration:.3f}",
                   "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        amap = ["-map", "1:a:0"]
        acodec = ["-c:a", "aac", "-b:a", "64k"]
        afilter = []

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-vf", vf,
        "-map", "0:v:0",
        *amap,
        *afilter,
        *acodec,
        *vcodec,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-map_metadata", "-1",
        "-threads", str(threads),
        "-progress", "pipe:1", "-nostats",
        str(out),
    ]

    log.info("%s", " ".join(shlex.quote(c) for c in cmd[:14]) + " ...")
    _run(cmd, duration=duration, on_progress=on_progress)
    if not out.exists() or out.stat().st_size == 0:
        raise RenderError(f"ffmpeg produced no output for {out.name}")

    # ffmpeg exits 0 having written a valid container with no frames in it —
    # asked to cut past the end of the source, for one. Catch it here, where
    # the reason is still obvious, rather than in thumbnail() three lines on.
    if not _has_frames(out):
        raise RenderedNothingError(
            f"{out.name} has no video frames: cut {start:.2f}s->{start + duration:.2f}s "
            f"is outside {Path(src).name}"
        )
    return out


def _has_frames(video: Path) -> bool:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True,
    )
    return probe.returncode == 0 and probe.stdout.strip() not in ("", "0", "N/A")


def _run(cmd: list[str], *, duration: float, on_progress) -> None:
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment guard
        raise RenderError("ffmpeg is not on PATH; install ffmpeg") from exc

    assert proc.stdout is not None
    for line in proc.stdout:
        if on_progress and line.startswith("out_time_us="):
            try:
                done = int(line.split("=", 1)[1]) / 1_000_000
            except ValueError:
                continue
            if duration > 0:
                on_progress(max(0.0, min(1.0, done / duration)))
    proc.stdout.close()
    stderr = proc.stderr.read() if proc.stderr else ""
    if proc.stderr:
        proc.stderr.close()
    if proc.wait() != 0:
        raise RenderError(_explain(stderr) or f"ffmpeg exited {proc.returncode}")


#: ffmpeg reports the consequence last and the cause first, so the final
#: line is usually the least useful one in the log.
_SYMPTOM = ("Nothing was written", "Conversion failed", "Error while filtering")


def _explain(stderr: str) -> str:
    """Turn ffmpeg's wall of text into the one line that matters."""
    stderr = stderr.strip()
    if not stderr:
        return ""
    lines = [
        ln.strip() for ln in stderr.splitlines()
        if ln.strip() and not ln.startswith(("frame=", "size=", "  "))
    ]
    if not lines:
        return stderr[-400:]

    for line in lines:
        if "No such filter" in line and "ass" in line:
            return (
                "this ffmpeg was built without libass, so captions cannot be "
                "burned in (check: ffmpeg -filters | grep ass)"
            )

    causes = [ln for ln in lines if not any(sym in ln for sym in _SYMPTOM)]
    return re.sub(r"\s+", " ", (causes or lines)[0])[:400]


def thumbnail(clip: Path, out: Path, at: float = 0.5) -> Path:
    """A poster frame for the clip gallery."""
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-ss", str(at), "-i", str(clip),
         "-frames:v", "1", "-q:v", "3", "-vf", "scale=540:-2", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not out.exists():
        # A clip shorter than `at` has no frame there; fall back to the first.
        result = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(clip), "-frames:v", "1", "-q:v", "3",
             "-vf", "scale=540:-2", str(out)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RenderError(_explain(result.stderr) or "thumbnail failed")
    return out


def _esc(p: Path) -> str:
    """ffmpeg filter args: escape : and \\ and '."""
    return str(p).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
