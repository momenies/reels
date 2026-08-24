"""End-to-end: long video in, vertical captioned clips out.

    python -m pipeline.run input.mp4 --lang ar --clips 5 --out out/

Cached artifacts (audio, transcript) are reused, so re-running to tweak
captions or crop costs seconds instead of minutes.

The work is exposed as :func:`process` rather than living inside ``main()``
so the API server runs exactly the same code path as the CLI — a job queue
that reimplements the pipeline is a job queue that drifts away from it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import FONTS_DIR, settings
from .log import get, setup
from . import captions, reframe, render, score, transcribe

log = get("run")

Progress = Callable[[str, float, str], None]
"""``(stage, fraction_complete, message)`` — stage is one of STAGES."""

STAGES = ("probe", "transcribe", "score", "render", "done")


def _noop(stage: str, pct: float, message: str) -> None:
    pass


def slug(text: str, n: int = 40) -> str:
    """A filename-safe stub that keeps Arabic readable.

    Unicode is normalised first so visually identical titles do not produce
    two different filenames, and the result is checked against the handful of
    names Windows reserves — customers download these onto real machines.
    """
    text = unicodedata.normalize("NFC", text or "")
    s = re.sub(r"[^\w؀-ۿ\s-]", "", text).strip()
    s = re.sub(r"[\s_]+", "-", s).strip("-")[:n].strip("-")
    if not s or s.upper() in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(COM|LPT)\d", s.upper()):
        return "clip"
    return s


@dataclass
class Options:
    """Everything the pipeline can be told to do, in one object."""

    lang: str | None = None
    clips: int = 5
    min_score: int = 60
    model: str | None = None
    device: str | None = None
    gpu: bool | None = None
    faces: bool = True
    style: str = captions.DEFAULT_STYLE
    hook: bool = False
    loudnorm: bool | None = None
    crf: int | None = None
    workers: int | None = None

    def resolved(self):
        cfg = settings()
        return dict(
            model=self.model or cfg.whisper_model,
            device=self.device or cfg.device,
            gpu=cfg.gpu if self.gpu is None else self.gpu,
            loudnorm=cfg.loudnorm if self.loudnorm is None else self.loudnorm,
            crf=cfg.crf if self.crf is None else self.crf,
            workers=self.workers or cfg.effective_render_workers(),
        )


@dataclass
class Result:
    clips: list[dict] = field(default_factory=list)
    out_dir: Path = Path("out")
    duration: float = 0.0
    language: str | None = None
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "clips": self.clips,
            "source_duration": round(self.duration, 2),
            "language": self.language,
            "elapsed_seconds": round(self.seconds, 1),
            "errors": self.errors,
        }


def _transcript_cache_path(work: Path, model: str, lang: str | None) -> Path:
    """Cache per (model, language).

    Keying on the output directory alone meant that re-running with
    ``--model large-v3`` silently reused the ``small`` transcript, which is
    the single most confusing way for a quality flag to do nothing.
    """
    return work / f"transcript.{model}.{lang or 'auto'}.json"


def process(
    video: Path,
    out_dir: Path,
    options: Options | None = None,
    *,
    on_progress: Progress | None = None,
) -> Result:
    """Run the whole pipeline. Returns what was produced, including failures."""
    started = time.monotonic()
    options = options or Options()
    on_progress = on_progress or _noop
    opts = options.resolved()

    video = Path(video)
    out_dir = Path(out_dir)
    work = out_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = Result(out_dir=out_dir)

    # 0. probe -------------------------------------------------------------
    on_progress("probe", 0.0, "reading the source file")
    info = reframe.probe(video)
    result.duration = info.duration
    log.info(
        "%dx%d %.2ffps %.0fs audio=%s",
        info.width, info.height, info.fps, info.duration, info.has_audio,
    )
    on_progress("probe", 1.0, f"{info.width}x{info.height}, {info.duration:.0f}s")

    # 1 + 2. transcript (cached) -------------------------------------------
    tpath = _transcript_cache_path(work, opts["model"], options.lang)
    if tpath.exists():
        log.info("cached transcript %s", tpath.name)
        segments = transcribe.load(tpath)
        on_progress("transcribe", 1.0, "using the cached transcript")
    else:
        on_progress("transcribe", 0.0, "extracting audio")
        wav = transcribe.extract_audio(video, work / "audio.wav")
        segments = transcribe.transcribe(
            wav,
            language=options.lang,
            model_size=opts["model"],
            device=opts["device"],
            compute_type=settings().whisper_compute_type(),
            duration=info.duration,
            on_progress=lambda p: on_progress("transcribe", p, "transcribing speech"),
        )
        transcribe.save(segments, tpath)
        on_progress("transcribe", 1.0, f"{len(segments)} segments")
    result.language = options.lang

    # 3. pick moments -------------------------------------------------------
    on_progress("score", 0.0, "finding the moments worth cutting")
    clips = score.find_clips(
        segments,
        max_clips=options.clips,
        min_score=options.min_score,
        video_duration=info.duration,
    )
    if not clips:
        on_progress("done", 1.0, "no clip scored high enough")
        log.warning("no clip scored high enough — try a lower --min-score")
        result.seconds = time.monotonic() - started
        return result
    on_progress("score", 1.0, f"{len(clips)} moment(s) selected")

    # 4-6. reframe, caption, render ----------------------------------------
    total = len(clips)
    done = [0]
    style = captions.get_style(options.style)

    def build_one(index_clip):
        i, clip = index_clip
        name = f"{i:02d}-{slug(clip.title)}"
        log.info("%d/%d  score=%d  %s", i, total, clip.score, clip.title)
        log.info("    %.1fs -> %.1fs (%.1fs)", clip.start, clip.end, clip.duration)

        # 4. reframe
        if options.faces:
            crop_w, plan = reframe.build_crop_plan(video, clip.start, clip.duration, info)
        else:
            crop_w, plan = reframe.static_plan(info)
        crop_x0 = plan[0][1]
        cmdfile = (
            reframe.write_sendcmd(plan, work / f"{name}.cmd") if len(plan) > 1 else None
        )

        # 5. captions
        words = score.words_in_range(segments, clip.start, clip.end)
        if not words:
            log.warning("%s: no words in range — rendering without captions", name)
        ass = captions.build_ass(
            words, work / f"{name}.ass", style=style, fonts_dir=FONTS_DIR
        )
        hook = (
            captions.build_hook(
                clip.title, min(clip.duration, 4.0), work / f"{name}.hook.ass",
                fonts_dir=FONTS_DIR,
            )
            if options.hook
            else None
        )

        # 6. render
        mp4 = render.render(
            video,
            out_dir / f"{name}.mp4",
            start=clip.start,
            duration=clip.duration,
            crop_w=crop_w,
            crop_x0=crop_x0,
            sendcmd_file=cmdfile,
            ass_file=ass,
            hook_file=hook,
            fonts_dir=FONTS_DIR,
            gpu=opts["gpu"],
            crf=opts["crf"],
            loudnorm=opts["loudnorm"],
            has_audio=info.has_audio,
        )
        thumb = out_dir / f"{name}.jpg"
        render.thumbnail(mp4, thumb)

        entry = clip.to_dict()
        entry.update({
            "file": mp4.name,
            "thumbnail": thumb.name,
            "bytes": mp4.stat().st_size,
            "words": len(words),
            "style": style.name,
        })
        return entry

    workers = max(1, min(opts["workers"], total))
    on_progress("render", 0.0, f"rendering {total} clip(s) on {workers} worker(s)")

    def guarded(item):
        i, clip = item
        try:
            entry = build_one(item)
        except Exception as exc:
            # One bad clip must not throw away the other four, nor the
            # transcript the customer has already paid for.
            log.error("clip %d (%s) failed: %s", i, clip.title, exc)
            result.errors.append(f"clip {i} ({clip.title}): {exc}")
            entry = None
        done[0] += 1
        on_progress("render", done[0] / total, f"{done[0]}/{total} clips rendered")
        return entry

    if workers == 1:
        entries = [guarded(item) for item in enumerate(clips, 1)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            entries = list(pool.map(guarded, list(enumerate(clips, 1))))

    result.clips = [e for e in entries if e]
    result.seconds = time.monotonic() - started

    (out_dir / "clips.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    on_progress("done", 1.0, f"{len(result.clips)} clip(s) ready")
    log.info("done — %d clip(s) in %s (%.1fs)", len(result.clips), out_dir, result.seconds)
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # python-dotenv is optional for everything but score
        pass
    setup()
    settings(refresh=True)

    ap = argparse.ArgumentParser(
        prog="python -m pipeline.run",
        description="Long video in, vertical captioned clips out.",
    )
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--lang", default=None, help="ar, en, ... (default: auto)")
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--min-score", type=int, default=60)
    ap.add_argument("--model", default=None, help="faster-whisper size")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--gpu", action="store_true", help="NVENC encoding")
    ap.add_argument("--no-faces", action="store_true", help="static centre crop")
    ap.add_argument(
        "--style", default=captions.DEFAULT_STYLE, choices=sorted(captions.STYLES),
        help="caption preset",
    )
    ap.add_argument("--hook", action="store_true", help="burn the title into the top third")
    ap.add_argument("--no-loudnorm", action="store_true", help="skip loudness matching")
    ap.add_argument("--crf", type=int, default=None, help="lower is better quality")
    ap.add_argument("--workers", type=int, default=None, help="parallel clip renders")
    args = ap.parse_args(argv)

    options = Options(
        lang=args.lang,
        clips=args.clips,
        min_score=args.min_score,
        model=args.model,
        device=args.device,
        gpu=args.gpu or None,
        faces=not args.no_faces,
        style=args.style,
        hook=args.hook,
        loudnorm=False if args.no_loudnorm else None,
        crf=args.crf,
        workers=args.workers,
    )

    def show(stage: str, pct: float, message: str) -> None:
        bar = "█" * int(pct * 24) + "░" * (24 - int(pct * 24))
        end = "\n" if stage == "done" or pct >= 1.0 else "\r"
        print(f"  {stage:<10} {bar} {pct * 100:3.0f}%  {message}", end=end, flush=True)

    try:
        result = process(args.video, args.out, options, on_progress=show)
    except (reframe.ProbeError, transcribe.TranscriptionError, score.ScoringError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    if not result.clips:
        print("No clips were produced. Try --min-score 45.", file=sys.stderr)
        return 1
    print(f"\n{len(result.clips)} clip(s) in {result.out_dir}/  ({result.seconds:.0f}s)")
    for c in result.clips:
        print(f"  {c['score']:>3}  {c['duration']:>5.1f}s  {c['file']}")
    if result.errors:
        print(f"\n{len(result.errors)} clip(s) failed:", file=sys.stderr)
        for e in result.errors:
            print(f"  - {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
