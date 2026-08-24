# reels-engine

Long video in, vertical captioned clips out — with a studio in front of it.

The hard half is clip quality: face-tracked 9:16 reframing, word-level captions
that shape Arabic correctly, and a model that picks the moments instead of the
middle. The product layer around it — upload, queue, live progress, gallery,
download — now ships too.

```
                          ┌─ web/     drag-drop studio, AR/EN, live progress
   browser ──▶ FastAPI ──▶├─ SQLite   job store, survives a restart
                          └─ worker ──▶ pipeline ──▶ clips
```

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
| `fetch.py` | a link → a local file (yt-dlp) | bandwidth |

## Quick start

```bash
docker compose up --build          # needs ANTHROPIC_API_KEY in .env
```

Then open <http://localhost:8000>. Or without Docker:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # add your ANTHROPIC_API_KEY
make serve                         # http://localhost:8000
```

Requires `ffmpeg` and `ffprobe` on PATH, built with `--enable-libass`
(check: `ffmpeg -filters | grep ass`).

## Two ways in

| | `server/` — the product | `web/` — the local tool |
|---|---|---|
| Run | `make serve` | `python -m web.app` |
| Queue | SQLite, survives a restart | in-memory, single worker |
| Progress | SSE + polling fallback | captured pipeline log |
| Exposure | optional `REELS_API_KEY` | binds to 127.0.0.1, trusts the keyboard |

Both call the same `pipeline.run.process`. Keep the local tool for quick
one-offs on your own machine; deploy `server/`.

## The studio

Drag a video in, pick a caption style, watch it work, download the clips.

- **Live progress over SSE**, with a polling fallback for proxies that buffer
  event streams. Progress is stored on the job row, so a refresh, a second
  tab, or a reconnect all see the same truth.
- **Bilingual, RTL-native.** Arabic flips the whole document, not just the
  strings — every rule uses logical properties.
- **Hover-to-preview clips**, score badges, a lightbox player, per-clip
  download, and a zip of the lot.
- **Job history** that survives a restart: interrupted jobs are requeued, and
  the cached transcript means a rerun costs seconds.
- Dark and light, and it works on a phone.
- **Paste a link instead of uploading.** Needs `yt-dlp` on PATH; the download
  runs on the worker, because no proxy holds an HTTP request open for the
  minutes a two-hour video takes. `GET /api/health` reports whether it is
  available. Set `YOUTUBE_API_KEY` for title/channel metadata.

## CLI

The studio and the CLI call the same `pipeline.run.process` — a queue that
reimplements the pipeline is a queue that drifts away from it.

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
| `--style beast` | caption preset — `pop`, `beast`, `clean`, `neon`, `boxed` |
| `--hook` | burn the generated headline into the top third |
| `--workers 4` | parallel clip renders |

Output: `NN-title.mp4`, `NN-title.jpg`, plus `clips.json` with scores and
reasons. The transcript is cached in `out/.work/` **per model and language**,
so re-running to tweak caption styling costs seconds, not minutes.

## API

| Route | Does |
|---|---|
| `POST /api/jobs` | multipart upload **or** a `url` field, + options → queued job |
| `GET /api/jobs/{id}` | status, progress, clips |
| `GET /api/jobs/{id}/events` | SSE progress stream |
| `GET /api/jobs/{id}/files/{name}` | a clip or thumbnail (supports Range) |
| `GET /api/jobs/{id}/download` | every clip as one zip |
| `DELETE /api/jobs/{id}` | cancel if queued, then remove job and media |
| `GET /api/styles` | caption presets, with CSS colours for a picker |
| `GET /api/health` | ffmpeg present, queue depth |

Interactive docs at `/docs`. Set `REELS_API_KEY` and every `/api` route needs
an `X-API-Key` header — that is the lock you put on a single-tenant box before
it is on the open internet, not user accounts.

Configuration is environment-driven; see `.env.example` for the full list.

## Tests

```bash
make test     # 178 unit tests — no ffmpeg, no models, no API key, ~1.5s
make smoke    # 18 renderer checks — needs ffmpeg with libass, ~40s
make check    # both, which is what CI runs
```

`smoke.py` does not just check that ffmpeg exits 0 — that is exactly the
failure this pipeline is prone to. libass draws nothing when it cannot find a
font and ffmpeg still succeeds, so it renders each clip twice, with and
without the subtitle filter, and compares the two with ffmpeg's own `psnr`:

- full frame must differ → captions were actually burned in
- top half must be identical → captions sit clear of the TikTok/Reels chrome

It runs on the standard library alone, so a broken dependency can never mask a
renderer regression. It also covers ASS escaping, caption event timing, every
style preset, the crop planner's hysteresis (faces injected, so no mediapipe),
and that a silent source still produces a playable clip.

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

Transcript text is never trusted as markup. A single `{` from a transcript
opens an ASS override block and swallows the rest of the caption line, so
`captions.escape()` neutralises braces and backslashes before they reach an
event. Out-of-order Whisper word timings are clamped too — libass stops
rendering at the first negative-length event it sees.

One more placement trap: ASS ignores `MarginV` for the middle alignments
(`\an4`-`\an6`), so `\an5` plus a generous margin still lands the caption
dead centre. Captions carry an explicit `\pos()` instead.

## Why the crop doesn't jitter

Following the face every frame looks seasick. Instead: sample at 4 fps, median
filter, EMA, then **lock** the crop and only move when the subject drifts past
6% of frame width — easing across with smoothstep over 0.5s. The camera holds
still during a shot and repositions on speaker changes, like a human operator.

Sampling reads the clip forward once and drops the frames it does not need.
The obvious implementation — `cap.set(POS_MSEC)` before every sample — makes
OpenCV re-seek and re-decode from the preceding keyframe each time, which on a
long-GOP H.264 file costs more than decoding the whole clip. Detection also
runs on a downscaled copy: face position is normalised to frame width, so the
answer is identical and the detector is several times faster.

## What makes the output postable

- **Loudness matched to what the platforms normalise to** (`I=-14 TP=-1.5`).
  A clip that arrives quiet stays quiet next to everything else in the feed.
- **A silent source still gets an audio track** — several platforms reject an
  upload with no audio stream at all.
- **`+faststart`**, so the file starts playing before it finishes downloading.
- **One bad clip never kills the batch.** Each clip is rendered in isolation;
  a failure is reported against that clip and the rest still ship.
- **A frameless render is caught where the reason is still obvious.** ffmpeg
  exits 0 having written a valid container with no frames in it — asked to cut
  past the end of the source, for one — so `render()` checks and raises
  `RenderedNothingError` rather than failing three lines later in `thumbnail()`.

## Speed

| Change | Effect |
|---|---|
| Sequential decode for face sampling | no re-seek per sample on long-GOP files |
| Detection on a 640px copy | several times faster, same crop plan |
| Parallel clip renders | `--workers`, auto-sized to half your cores |
| Whisper model cached per process | a server pays the load cost once, not once a job |
| Transcript cached per model+language | restyling a batch costs seconds |
| `condition_on_previous_text=False` | faster, and stops the phrase-loop on noisy audio |

## What's not built yet

- **Sports mode** — swap face detection for ball/action tracking in `reframe.py`
- **Caption translation** — translate the ASS text layer, keep original audio
- **B-roll / gameplay strip** — `vstack` a second source under the main frame

## Then the rest of the product

1. **Scale out** — more of the compose service against shared storage, and
   Postgres in place of SQLite. The schema ports unchanged.
2. **Storage** — Cloudflare R2 (zero egress fees, which matters a lot for video)
3. **Publishing** — YouTube Data API v3 (**1600 quota units per upload against a
   10,000/day default = 6 uploads/day** until you get an increase; apply early,
   it takes weeks). TikTok Content Posting API requires an app audit. Instagram
   Reels needs a Business account plus Facebook app review.
4. **Scheduling** — delayed jobs *or* a cron sweep, never both without an atomic
   status transition, or you will double-post.

## Before you scale

Source video has to be yours or licensed. Re-uploading other creators' content
is what gets accounts terminated and platform API access revoked — and API
access revocation kills the whole product, not just one account. The
sponsored-campaign model (creator opts in and pays for clips) is the version
of this business with a legal foundation under it.
