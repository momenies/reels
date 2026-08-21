# reels-engine

The core of a Ssemble-style product: long video in, vertical captioned clips out.

This is deliberately the hard half. Auth, dashboards, and scheduling are
solved problems you can build in a week. Clip quality is what people pay for,
and it's where a clone either works or doesn't.

## Pipeline

```
video ──▶ ffmpeg ──▶ faster-whisper ──▶ Claude ──▶ MediaPipe ──▶ ASS ──▶ ffmpeg
          audio      word timings      moments    face track   captions  render
```

| Step | Doing what | Cost driver |
|---|---|---|
| `transcribe.py` | 16 kHz audio → word-level transcript | GPU seconds |
| `score.py` | transcript → ranked clip candidates | LLM tokens |
| `reframe.py` | face track → smoothed crop plan | CPU seconds |
| `captions.py` | word timings → styled ASS subtitles | free |
| `render.py` | cut + crop + burn + encode | GPU/CPU seconds |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
```

Requires `ffmpeg` and `ffprobe` on PATH, built with `--enable-libass`
(check: `ffmpeg -filters | grep ass`).

Caption fonts are committed in `assets/fonts/` and loaded from there only,
never from the system — see the Arabic section below for why. The face
detection model is committed too, so a worker needs no egress to reframe.

`mediapipe` and `opencv` are only needed for face tracking; without them the
pipeline falls back to a static centre crop and still runs. On a slim image
they also need `libGLESv2`/`libEGL`:

```bash
apt-get install -y libgles2 libegl1
```

## Smoke test

```bash
python smoke.py      # ~20s, no model downloads, no API key
```

Renders the ffmpeg half twice — with and without the subtitle filter — and
compares the two with ffmpeg's `psnr`. Checks that captions are actually
burned in (rather than silently skipped), that they stay out of the top half,
that `sendcmd` really retargets the crop, and that a missing font raises
instead of producing blank output. The crop planner is covered too, with face
positions injected rather than detected, so the hysteresis is tested without
pulling in mediapipe.

Runs on the standard library alone — no `pip install` — so a broken
dependency can never mask a renderer regression. CI runs it on every push
and pull request.

## Run

### As a web UI

```bash
pip install -r requirements-web.txt
python -m web.app                  # http://127.0.0.1:8000
```

Drop a video in, watch the pipeline's own output stream as it works, play the
clips in the page and download the ones you want. It calls the same
`pipeline.run.process()` the CLI does, one job at a time — a single render
already saturates the CPU, so a second job in parallel only makes both slower.

It binds to localhost and trusts whoever is at the keyboard. Don't expose it
to a network you don't control: it accepts uploads and spends money on the
Claude call for each one.

### As a CLI

```bash
python -m pipeline.run input.mp4 --lang ar --clips 5 --out out/
```

| Flag | Purpose |
|---|---|
| `--lang ar` | skip auto-detect; more accurate on dialect |
| `--model medium` | whisper size — `small` on CPU, `large-v3` on GPU |
| `--device cuda --gpu` | GPU transcription + NVENC encoding |
| `--no-faces` | static centre crop, ~3x faster |
| `--min-score 45` | loosen if nothing passes the bar |

Output: `NN-title.mp4`, `NN-title.jpg`, plus `clips.json` with scores and
reasons. The transcript is cached in `out/.work/`, so re-running to tweak
caption styling costs seconds, not minutes.

## Verified

`python smoke.py` — 13/13 checks, ffmpeg 6.1.1 with libass:

- Dynamic crop via `sendcmd` retargets mid-clip without re-encoding twice ✓
- Latin word-level highlight (active word amber + scaled) ✓
- Arabic cursive shaping and RTL ordering, correct ✓
- Captions land in the lower third, top half of the frame untouched ✓
- Crop offsets even, in bounds, timestamps strictly increasing ✓
- Crop holds while the subject is still and follows when they move ✓
- No faces falls back to a centred crop ✓
- A cut past the end of the source raises instead of writing an empty clip ✓
- A missing font raises `MissingFontError` instead of rendering blank ✓

Beyond the smoke test, on real footage:

- **Face tracking**, against a 1920x1080 clip of a real face crossing the
  frame: 32/32 samples detected, crop plan tracked 0.18 → 0.77 of frame
  width, holding at both ends.
- **`run.py` end to end** with that clip: 1080x1920 H.264/AAC out, captions
  burned in, thumbnail and `clips.json` written. The subject stays framed
  from first frame to last.
- **`transcribe.extract_audio`** produces 16 kHz mono PCM, and the mapping
  into `Segment`/`Word` survives the shapes faster-whisper actually emits —
  `words=None` on a segment, whitespace-only words — plus an exact
  save/load roundtrip.
- **The web UI**, driven in a real browser: upload, queue, live log, clip
  cards, click-to-play, download, and range requests (206) so seeking works.

Two steps could not be run here and are **not** verified: Whisper itself
(model weights are fetched from Hugging Face, blocked by this network) and
`score.find_clips` (needs an `ANTHROPIC_API_KEY`). Everything downstream of
them was driven with a real transcript file and a stubbed clip list.

## The Arabic detail your competitors get wrong

Arabic is cursive — letters change shape based on neighbours, and libass runs
HarfBuzz shaping plus FriBidi reordering per rendered line. Inject ASS override
tags mid-word and shaping breaks; mix Latin and Arabic on one line with
per-word colouring and bidi can land the highlight on the wrong word.

So `captions.py` uses **word-level pop for LTR, line-level pop for RTL**.
The fonts are Tajawal ExtraBold (Arabic) and Montserrat ExtraBold (Latin),
committed in `assets/fonts/` — system fonts are not reliable in containers,
and a missing Arabic font fails *silently*: libass draws nothing and ffmpeg
exits 0. This exact failure happened during testing, so
`captions.require_font()` checks the file is on disk before any encoding
starts. `scripts/fetch_fonts.py` regenerates them from google/fonts.

One more placement trap: ASS ignores `MarginV` for the middle alignments
(`\an4`-`\an6`), so `\an5` plus a generous margin still lands the caption
dead centre. Captions carry an explicit `\pos()` instead.

## Why the crop doesn't jitter

Following the face every frame looks seasick. Instead: sample at 4 fps, median
filter, EMA, then **lock** the crop and only move when the subject drifts past
6% of frame width — easing across with smoothstep over 0.5s. The camera holds
still during a shot and repositions on speaker changes, like a human operator.

## What's not built yet

- **Sports mode** — swap face detection for ball/action tracking in `reframe.py`
- **Caption translation** — translate the ASS text layer, keep original audio
- **B-roll / gameplay strip** — `vstack` a second source under the main frame
- **Hook titles** — `captions.build_hook()` exists, not wired into `run.py`

## Then the product layer

Once clip quality is good enough that you'd post the output unedited:

1. **Worker** — this repo behind BullMQ + Redis on a GPU box. Not serverless;
   ffmpeg and Whisper need real machines.
2. **Storage** — Cloudflare R2 (zero egress fees, which matters a lot for video)
3. **Publishing** — YouTube Data API v3 (**1600 quota units per upload against a
   10,000/day default = 6 uploads/day** until you get an increase; apply early,
   it takes weeks). TikTok Content Posting API requires an app audit. Instagram
   Reels needs a Business account plus Facebook app review.
4. **Scheduling** — delayed jobs *or* a cron sweep, never both without an atomic
   status transition, or you will double-post.
5. **Dashboard** — the easy part. Build it last.

## Before you scale

Source video has to be yours or licensed. Re-uploading other creators' content
is what gets accounts terminated and platform API access revoked — and API
access revocation kills the whole product, not just one account. The
sponsored-campaign model (creator opts in and pays for clips) is the version
of this business with a legal foundation under it.
