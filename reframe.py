"""Step 3: 16:9 -> 9:16 with the subject kept in frame.

The naive approach (static center crop) decapitates anyone standing off-centre,
which is most podcast footage. The naive fix (crop follows the face every frame)
produces seasick jitter.

What works, and what the commercial tools actually do:

  1. Sample faces a few times per second, not every frame.
  2. Smooth the trajectory hard (median -> EMA).
  3. LOCK the crop and only move when the subject drifts past a threshold
     (hysteresis). Then ease across over ~0.5s instead of snapping.

Result: the camera holds still during a shot and repositions on speaker
changes, like a human operator.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float


def probe(video: Path) -> VideoInfo:
    def q(field: str) -> str:
        return subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", f"stream={field}",
             "-of", "default=nw=1:nk=1", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]

    num, den = (q("r_frame_rate").split("/") + ["1"])[:2]
    duration = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    return VideoInfo(
        width=int(q("width")),
        height=int(q("height")),
        fps=float(num) / float(den or 1),
        duration=float(duration),
    )


def _detect_face_centers(
    video: Path, start: float, duration: float, sample_fps: float = 4.0
) -> list[tuple[float, float | None]]:
    """[(t_relative, face_center_x_normalised or None)] — None = no face found."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("[reframe] mediapipe/opencv missing -> static centre crop")
        return []

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    out: list[tuple[float, float | None]] = []
    step = 1.0 / sample_fps
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + t) * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        res = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.detections:
            # largest face wins — that's the speaker, not the background
            best = max(res.detections, key=lambda d: d.location_data.relative_bounding_box.width)
            box = best.location_data.relative_bounding_box
            out.append((t, box.xmin + box.width / 2))
        else:
            out.append((t, None))
        t += step

    cap.release()
    detector.close()
    return out


def _smooth(values: list[float], window: int = 5) -> list[float]:
    """Median filter kills detection outliers; EMA removes the remaining shake."""
    med = []
    for i in range(len(values)):
        lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
        chunk = sorted(values[lo:hi])
        med.append(chunk[len(chunk) // 2])

    ema, alpha = [], 0.25
    acc = med[0] if med else 0.5
    for v in med:
        acc = alpha * v + (1 - alpha) * acc
        ema.append(acc)
    return ema


def build_crop_plan(
    video: Path,
    start: float,
    duration: float,
    info: VideoInfo,
    *,
    move_threshold: float = 0.06,   # fraction of frame width before we bother moving
    ease_seconds: float = 0.5,
    sample_fps: float = 4.0,
) -> tuple[int, list[tuple[float, int]]]:
    """Returns (crop_width_px, [(time, x_px), ...])."""
    crop_w = int(info.height * 9 / 16) // 2 * 2
    crop_w = min(crop_w, info.width)
    max_x = info.width - crop_w

    samples = _detect_face_centers(video, start, duration, sample_fps)
    if not samples:
        return crop_w, [(0.0, max_x // 2)]

    # carry the last known face through frames where detection dropped
    last = 0.5
    filled = []
    for _, c in samples:
        if c is not None:
            last = c
        filled.append(last)
    smoothed = _smooth(filled)

    # hysteresis: hold position until the subject drifts too far
    plan: list[tuple[float, int]] = []
    locked = smoothed[0]
    for (t, _), target in zip(samples, smoothed):
        if abs(target - locked) > move_threshold:
            # ease from locked -> target over ease_seconds
            steps = max(2, int(ease_seconds * sample_fps))
            for i in range(1, steps + 1):
                p = i / steps
                p = p * p * (3 - 2 * p)  # smoothstep
                v = locked + (target - locked) * p
                x = int(max(0, min(max_x, v * info.width - crop_w / 2)))
                plan.append((round(t + (i - 1) / sample_fps, 3), x))
            locked = target
        elif not plan:
            x = int(max(0, min(max_x, locked * info.width - crop_w / 2)))
            plan.append((0.0, x))

    # dedupe identical consecutive positions — keeps the sendcmd file small
    deduped = [plan[0]]
    for t, x in plan[1:]:
        if x != deduped[-1][1] and t > deduped[-1][0]:
            deduped.append((t, x))

    print(f"[reframe] crop_w={crop_w} moves={len(deduped)}")
    return crop_w, deduped


def write_sendcmd(plan: list[tuple[float, int]], path: Path) -> Path:
    """ffmpeg's sendcmd filter re-parameterises crop at given timestamps."""
    lines = [f"{t:.3f} crop x {x};" for t, x in plan[1:]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
