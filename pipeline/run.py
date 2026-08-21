"""End-to-end: long video in, vertical captioned clips out.

    python -m pipeline.run input.mp4 --lang ar --clips 5 --out out/

Cached artifacts (audio, transcript) are reused, so re-running to tweak
captions or crop costs seconds instead of minutes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv

from . import captions, reframe, render, score, transcribe

load_dotenv()


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^\w\u0600-\u06FF\s-]", "", text).strip()
    return re.sub(r"\s+", "-", s)[:n] or "clip"


def process(
    video: Path,
    out: Path,
    *,
    lang: str | None = None,
    clips: int = 5,
    min_score: int = 60,
    model: str = "small",
    device: str = "cpu",
    gpu: bool = False,
    faces: bool = True,
) -> list[dict]:
    """Long video in, vertical captioned clips out. Returns the manifest.

    Split out of main() so callers other than the CLI — the web UI, a queue
    worker — drive the same code path rather than a copy of it.
    """
    work = out / ".work"
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    info = reframe.probe(video)
    print(f"[probe] {info.width}x{info.height} {info.fps:.2f}fps {info.duration:.0f}s")

    # 1 + 2. transcript (cached)
    tpath = work / "transcript.json"
    if tpath.exists():
        print("[transcribe] cached")
        segments = transcribe.load(tpath)
    else:
        wav = transcribe.extract_audio(video, work / "audio.wav")
        segments = transcribe.transcribe(
            wav,
            language=lang,
            model_size=model,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )
        transcribe.save(segments, tpath)

    # 3. pick moments
    found = score.find_clips(segments, max_clips=clips, min_score=min_score)
    if not found:
        print("No clip scored high enough. Try a lower --min-score.")
        return []

    manifest = []
    for i, clip in enumerate(found, 1):
        # the model returns timestamps, not guarantees; a clip that starts at
        # or past the end of the source renders an empty file
        clip.start = max(0.0, min(clip.start, info.duration))
        clip.end = max(clip.start, min(clip.end, info.duration))
        if clip.duration < 1.0:
            print(f"\n=== {i}/{len(found)}  skipped: {clip.title}")
            print(f"    {clip.start:.1f}s -> {clip.end:.1f}s falls outside the "
                  f"{info.duration:.1f}s source")
            continue

        name = f"{i:02d}-{slug(clip.title)}"
        print(f"\n=== {i}/{len(found)}  {clip.score}  {clip.title}")
        print(f"    {clip.start:.1f}s -> {clip.end:.1f}s ({clip.duration:.1f}s)")

        try:
            manifest.append(_render_one(
                video, out, work, name, clip, info,
                faces=faces, gpu=gpu, segments=segments,
            ))
        except Exception as exc:
            # One bad clip must not throw away the ones already rendered, nor
            # the transcript that was paid for to find them.
            print(f"    [skip] {type(exc).__name__}: {exc}")

    (out / "clips.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDone. {len(manifest)} clips in {out}/")
    return manifest


def _render_one(
    video: Path, out: Path, work: Path, name: str, clip, info,
    *, faces: bool, gpu: bool, segments,
) -> dict:
    """Reframe, caption and encode one clip. Returns its manifest entry."""
    # 4. reframe
    if not faces:
        crop_w = min(int(info.height * 9 / 16) // 2 * 2, info.width)
        crop_x0, cmdfile = reframe.even_offset((info.width - crop_w) / 2), None
    else:
        crop_w, plan = reframe.build_crop_plan(video, clip.start, clip.duration, info)
        crop_x0 = plan[0][1]
        cmdfile = (
            reframe.write_sendcmd(plan, work / f"{name}.cmd")
            if len(plan) > 1
            else None
        )

    # 5. captions
    words = score.words_in_range(segments, clip.start, clip.end)
    ass = captions.build_ass(words, work / f"{name}.ass")
    if not words:
        print("    [captions] no words in range — rendering without captions")

    # 6. render
    mp4 = render.render(
        video,
        out / f"{name}.mp4",
        start=clip.start,
        duration=clip.duration,
        crop_w=crop_w,
        crop_x0=crop_x0,
        sendcmd_file=cmdfile,
        ass_file=ass,
        fonts_dir=captions.FONTS_DIR,
        gpu=gpu,
    )
    render.thumbnail(mp4, out / f"{name}.jpg")

    return {
        "file": mp4.name,
        "thumb": mp4.with_suffix(".jpg").name,
        "title": clip.title,
        "score": clip.score,
        "reason": clip.reason,
        "start": clip.start,
        "end": clip.end,
        "duration": round(clip.duration, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--lang", default=None, help="ar, en, ... (default: auto)")
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--min-score", type=int, default=60)
    ap.add_argument("--model", default="small", help="faster-whisper size")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gpu", action="store_true", help="NVENC encoding")
    ap.add_argument("--no-faces", action="store_true", help="static centre crop")
    args = ap.parse_args()

    process(
        args.video,
        args.out,
        lang=args.lang,
        clips=args.clips,
        min_score=args.min_score,
        model=args.model,
        device=args.device,
        gpu=args.gpu,
        faces=not args.no_faces,
    )


if __name__ == "__main__":
    main()
