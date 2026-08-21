"""Smoke test: exercises the ffmpeg half (crop plan + ASS + render) with no
model downloads and no API key.

It does not just check that ffmpeg exits 0 — that is exactly the failure this
pipeline is prone to. libass draws nothing when it cannot find a font and
ffmpeg still succeeds, so the test renders each clip twice, with and without
the subtitle filter, and compares the two with ffmpeg's own psnr filter:

  * full frame must differ      -> captions were actually burned in
  * top half must be identical  -> captions sit in the lower third, clear of
                                   the TikTok/Reels UI chrome

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

    # a missing font must fail loudly rather than render blank captions
    print("\n[fonts]")
    try:
        captions.build_ass(words_en, WORK / "nofont.ass", fonts_dir=WORK / "nope")
    except captions.MissingFontError:
        results.append(check("missing font raises instead of rendering blank", True))
    else:
        results.append(check("missing font raises instead of rendering blank", False))

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
