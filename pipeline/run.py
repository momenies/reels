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

    work = args.out / ".work"
    work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    info = reframe.probe(args.video)
    print(f"[probe] {info.width}x{info.height} {info.fps:.2f}fps {info.duration:.0f}s")

    # 1 + 2. transcript (cached)
    tpath = work / "transcript.json"
    if tpath.exists():
        print("[transcribe] cached")
        segments = transcribe.load(tpath)
    else:
        wav = transcribe.extract_audio(args.video, work / "audio.wav")
        segments = transcribe.transcribe(
            wav,
            language=args.lang,
            model_size=args.model,
            device=args.device,
            compute_type="float16" if args.device == "cuda" else "int8",
        )
        transcribe.save(segments, tpath)

    # 3. pick moments
    clips = score.find_clips(segments, max_clips=args.clips, min_score=args.min_score)
    if not clips:
        print("No clip scored high enough. Try --min-score 45.")
        return

    manifest = []
    for i, clip in enumerate(clips, 1):
        name = f"{i:02d}-{slug(clip.title)}"
        print(f"\n=== {i}/{len(clips)}  {clip.score}  {clip.title}")
        print(f"    {clip.start:.1f}s -> {clip.end:.1f}s ({clip.duration:.1f}s)")

        # 4. reframe
        if args.no_faces:
            crop_w = min(int(info.height * 9 / 16) // 2 * 2, info.width)
            crop_x0, cmdfile = reframe.even_offset((info.width - crop_w) / 2), None
        else:
            crop_w, plan = reframe.build_crop_plan(
                args.video, clip.start, clip.duration, info
            )
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
            args.video,
            args.out / f"{name}.mp4",
            start=clip.start,
            duration=clip.duration,
            crop_w=crop_w,
            crop_x0=crop_x0,
            sendcmd_file=cmdfile,
            ass_file=ass,
            fonts_dir=captions.FONTS_DIR,
            gpu=args.gpu,
        )
        render.thumbnail(mp4, args.out / f"{name}.jpg")

        manifest.append(
            {
                "file": mp4.name,
                "title": clip.title,
                "score": clip.score,
                "reason": clip.reason,
                "start": clip.start,
                "end": clip.end,
                "duration": round(clip.duration, 2),
            }
        )

    (args.out / "clips.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDone. {len(manifest)} clips in {args.out}/")


if __name__ == "__main__":
    main()
