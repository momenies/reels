"""Smoke test: exercises the ffmpeg half (crop plan + ASS + render) with no
model downloads and no API key.

It does not just check that ffmpeg exits 0 — that is exactly the failure this
pipeline is prone to. libass draws nothing when it cannot find a font and
ffmpeg still succeeds, so the test renders each clip twice, with and without
the subtitle filter, and compares the two with ffmpeg's own psnr filter:

  * full frame must differ      -> captions were actually burned in
  * top half must be identical  -> captions sit in the lower third, clear of
                                   the TikTok/Reels UI chrome

The crop planner is covered too, with face positions injected instead of
detected, so the hysteresis is tested without pulling in mediapipe.

The comparison renders are lossless (crf 0). At the pipeline's normal crf,
x264 spends bits differently once captions are on screen, so even untouched
regions drift by a dB or two and "identical" stops meaning identical.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pipeline import captions, reframe, render

WORK = Path("smoke")
FONTS = captions.FONTS_DIR


@dataclass
class W:
    start: float
    end: float
    text: str


def psnr(a: Path, b: Path, crop: str | None = None) -> float:
    """Average PSNR between two renders. inf == pixel-identical."""
    pre = f"[0:v]crop={crop}[x];[1:v]crop={crop}[y];[x][y]" if crop else "[0:v][1:v]"
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(a), "-i", str(b),
         "-lavfi", f"{pre}psnr", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr
    m = re.search(r"average:(\S+)", out)
    if not m:
        raise AssertionError(f"psnr produced no result:\n{out[-800:]}")
    return math.inf if m.group(1) == "inf" else float(m.group(1))


def _secs(ts: str) -> float:
    """ASS timestamps are H:MM:SS.cc."""
    h, m, rest = ts.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return ok


def main() -> int:
    WORK.mkdir(exist_ok=True)
    src = WORK / "src.mp4"

    # synthetic 16:9 source with a moving marker so we can see the crop track
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc2=size=1920x1080:rate=30:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)],
        check=True,
    )

    info = reframe.probe(src)
    print(f"probe: {info.width}x{info.height} {info.fps:.2f}fps {info.duration:.0f}s")

    crop_w = int(info.height * 9 / 16) // 2 * 2
    plan = [(0.0, 100), (1.0, 400), (2.5, 900), (4.0, 300)]
    cmd = reframe.write_sendcmd(plan, WORK / "t.cmd")

    words_en = [W(i * 0.4, i * 0.4 + 0.38, w) for i, w in
                enumerate("this is a smoke test of the caption renderer".split())]
    words_ar = [W(i * 0.4, i * 0.4 + 0.38, w) for i, w in
                enumerate("هذا اختبار سريع لمحرك الكابشن العربي".split())]

    results = []
    for tag, words in (("en", words_en), ("ar", words_ar)):
        print(f"\n[{tag}]")
        ass = captions.build_ass(words, WORK / f"{tag}.ass", fonts_dir=FONTS)

        common = dict(start=0, duration=4, crop_w=crop_w, crop_x0=plan[0][1],
                      sendcmd_file=cmd, gpu=False, crf=0)
        burned = render.render(src, WORK / f"out-{tag}.mp4",
                               ass_file=ass, fonts_dir=FONTS, **common)
        plain = render.render(src, WORK / f"plain-{tag}.mp4",
                              ass_file=None, fonts_dir=None, **common)

        full = psnr(burned, plain)
        top = psnr(burned, plain, crop="w=iw:h=ih/2:x=0:y=0")

        results.append(check(f"{tag}: captions burned in", full < 50,
                             f"(psnr {full:.1f} dB)"))
        results.append(check(f"{tag}: top half untouched", top == math.inf,
                             f"(psnr {'inf' if top == math.inf else f'{top:.1f}'} dB)"))
        print(f"  {burned} {burned.stat().st_size} bytes")

    # dynamic crop must actually retarget mid-clip
    print("\n[crop]")
    static = render.render(src, WORK / "static.mp4", start=0, duration=4,
                           crop_w=crop_w, crop_x0=plan[0][1],
                           sendcmd_file=None, ass_file=None, gpu=False, crf=0)
    moved = render.render(src, WORK / "moved.mp4", start=0, duration=4,
                          crop_w=crop_w, crop_x0=plan[0][1],
                          sendcmd_file=cmd, ass_file=None, gpu=False, crf=0)
    results.append(check("sendcmd retargets the crop", psnr(static, moved) < 50))

    # the crop planner: hold while the subject is still, ease when they move.
    # Faces are injected, so this runs without mediapipe or opencv.
    print("\n[plan]")
    info = reframe.VideoInfo(width=1920, height=1080, fps=25.0, duration=8.0)
    still_left = [(i * 0.25, 0.18) for i in range(8)]
    sweep = [(2.0 + i * 0.25, 0.18 + (i + 1) / 12 * (0.78 - 0.18)) for i in range(12)]
    still_right = [(5.0 + i * 0.25, 0.78) for i in range(8)]
    injected = still_left + sweep + still_right

    real_detect = reframe._detect_face_centers
    reframe._detect_face_centers = lambda *a, **k: injected
    try:
        crop_w, plan = reframe.build_crop_plan(src, 0.0, 8.0, info)
        xs = [x for _, x in plan]
        max_x = info.width - crop_w
        results.append(check("crop offsets are even", all(x % 2 == 0 for x in xs)))
        results.append(check("crop stays inside the frame",
                             all(0 <= x <= max_x for x in xs)))
        results.append(check("sendcmd timestamps increase",
                             all(b[0] > a[0] for a, b in zip(plan, plan[1:]))))
        results.append(check("crop follows the subject", max(xs) - min(xs) > crop_w // 2,
                             f"(x {min(xs)} -> {max(xs)})"))
        # nothing should be emitted while the subject holds still at the start
        early = [t for t, _ in plan if t < 1.5]
        results.append(check("crop holds while the subject is still", len(early) <= 1,
                             f"({len(early)} move(s) before t=1.5)"))

        reframe._detect_face_centers = lambda *a, **k: []
        crop_w2, plan2 = reframe.build_crop_plan(src, 0.0, 8.0, info)
        centred = reframe.even_offset((info.width - crop_w2) / 2)
        results.append(check("no faces falls back to a centred crop",
                             plan2 == [(0.0, centred)], f"(x={centred})"))
    finally:
        reframe._detect_face_centers = real_detect

    # a missing font must fail loudly rather than render blank captions
    print("\n[fonts]")
    try:
        captions.build_ass(words_en, WORK / "nofont.ass", fonts_dir=WORK / "nope")
    except captions.MissingFontError:
        results.append(check("missing font raises instead of rendering blank", True))
    else:
        results.append(check("missing font raises instead of rendering blank", False))

    # ASS is a markup format and transcript text is untrusted input: a stray
    # brace would open an override block and eat the rest of the line.
    print("\n[escaping]")
    hostile = [W(0.0, 0.4, "{\\pos(0,0)}"), W(0.4, 0.8, "safe")]
    ass = captions.build_ass(hostile, WORK / "esc.ass", fonts_dir=FONTS)
    body = ass.read_text(encoding="utf-8").split("[Events]")[1]
    results.append(check("brace in a word cannot open an override block",
                         "{\\pos(0,0)}" not in body))
    results.append(check("the word still renders", "safe" in body))

    # Out-of-order word timings must not produce a negative-length event:
    # libass stops rendering at the first one it sees.
    backwards = [W(1.0, 0.2, "first"), W(0.5, 0.5, "second")]
    ass = captions.build_ass(backwards, WORK / "back.ass", fonts_dir=FONTS)
    spans = []
    for line in ass.read_text(encoding="utf-8").splitlines():
        if line.startswith("Dialogue:"):
            _, start, end = line.split(",", 3)[:3]
            spans.append((_secs(start), _secs(end)))
    results.append(check("no zero or negative length caption events",
                         all(e > s for s, e in spans), f"({len(spans)} events)"))

    # Every preset has to name a font that is actually on disk.
    print("\n[styles]")
    ok = True
    for name in sorted(captions.STYLES):
        try:
            captions.build_ass(words_en, WORK / f"s-{name}.ass", style=name, fonts_dir=FONTS)
        except Exception as exc:
            print(f"    {name}: {exc}")
            ok = False
    results.append(check("every caption preset renders", ok,
                         f"({len(captions.STYLES)} presets)"))

    # A source with no audio track still has to produce a playable clip:
    # several platforms reject an upload with no audio stream at all.
    print("\n[silent source]")
    silent = WORK / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc2=size=640x360:rate=25:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)],
        check=True,
    )
    sinfo = reframe.probe(silent)
    results.append(check("probe reports the missing audio track", not sinfo.has_audio))
    scrop, splan = reframe.static_plan(sinfo)
    out = render.render(silent, WORK / "silent-out.mp4", start=0, duration=2,
                        crop_w=scrop, crop_x0=splan[0][1], sendcmd_file=None,
                        ass_file=None, has_audio=False, gpu=False)
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    results.append(check("a silent source still gets an audio track",
                         "audio" in streams and "video" in streams,
                         f"({', '.join(streams)})"))

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
