"""Smoke test: exercises the ffmpeg half (crop plan + ASS + render) with no
model downloads and no API key. Verifies the filter graph actually runs."""
from dataclasses import dataclass
from pathlib import Path
import subprocess
from pipeline import captions, reframe, render

@dataclass
class W:
    start: float; end: float; text: str

work = Path("smoke"); work.mkdir(exist_ok=True)
src = work / "src.mp4"

# synthetic 16:9 source with a moving marker so we can see the crop track
subprocess.run(["ffmpeg","-y","-loglevel","error","-f","lavfi",
    "-i","testsrc2=size=1920x1080:rate=30:duration=6",
    "-f","lavfi","-i","sine=frequency=440:duration=6",
    "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac",str(src)], check=True)

info = reframe.probe(src)
print("probe:", info)

crop_w = int(info.height*9/16)//2*2
plan = [(0.0, 100), (1.0, 400), (2.5, 900), (4.0, 300)]
cmd = reframe.write_sendcmd(plan, work/"t.cmd")
print("sendcmd:\n" + cmd.read_text())

words_en = [W(i*0.4, i*0.4+0.38, w) for i, w in
            enumerate("this is a smoke test of the caption renderer".split())]
ass_en = captions.build_ass(words_en, work/"en.ass")

words_ar = [W(i*0.4, i*0.4+0.38, w) for i, w in
            enumerate("هذا اختبار سريع لمحرك الكابشن العربي".split())]
ass_ar = captions.build_ass(words_ar, work/"ar.ass")

for tag, ass in (("en", ass_en), ("ar", ass_ar)):
    out = render.render(src, work/f"out-{tag}.mp4", start=0, duration=4,
                        crop_w=crop_w, crop_x0=plan[0][1],
                        sendcmd_file=cmd, ass_file=ass, gpu=False)
    print(tag, "->", out, out.stat().st_size, "bytes")
