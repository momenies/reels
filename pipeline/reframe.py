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

Sampling reads the clip forward once and drops the frames it does not need.
The obvious implementation — ``cap.set(POS_MSEC)`` before every sample — makes
OpenCV re-seek and re-decode from the preceding keyframe each time, which on a
long-GOP H.264 file costs more than decoding the whole clip. Detection also
runs on a downscaled copy: face position is normalised to frame width, so the
answer is identical and the detector is several times faster.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import MODELS_DIR, settings
from .log import get

log = get("reframe")


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool = True

    @property
    def is_portrait(self) -> bool:
        return self.height >= self.width


class ProbeError(RuntimeError):
    """ffprobe could not read the file — corrupt, truncated, or not a video."""


def probe(video: Path) -> VideoInfo:
    """Read dimensions, frame rate and duration in a single ffprobe call."""
    video = Path(video)
    if not video.exists():
        raise ProbeError(f"{video} does not exist")

    try:
        raw = subprocess.run(
            ["ffprobe", "-v", "error", "-of", "json",
             "-show_entries", "stream=codec_type,width,height,r_frame_rate,duration",
             "-show_entries", "format=duration", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout
    except FileNotFoundError as exc:  # pragma: no cover - environment guard
        raise ProbeError("ffprobe is not on PATH; install ffmpeg") from exc
    except subprocess.CalledProcessError as exc:
        raise ProbeError(f"ffprobe failed on {video.name}: {exc.stderr.strip()}") from exc

    import json as _json

    try:
        data = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned unreadable output for {video.name}") from exc

    streams = data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise ProbeError(f"{video.name} has no video stream")
    v = video_streams[0]

    num, den = (str(v.get("r_frame_rate", "25/1")).split("/") + ["1"])[:2]
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 25.0

    duration = data.get("format", {}).get("duration") or v.get("duration") or 0
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        raise ProbeError(
            f"{video.name} reports no duration — the file is likely truncated"
        )

    return VideoInfo(
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps if fps > 0 else 25.0,
        duration=duration,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


#: BlazeFace weights, committed so a worker never needs egress to reframe.
#: Full range on purpose: the short range model is the selfie one and finds
#: nothing on a normal wide shot. See assets/models/README.md.
MODEL_PATH = MODELS_DIR / "face_detection_full_range_sparse.tflite"


def _open_detector():
    """A face detector with a ``detect(rgb_frame) -> [(cx, width), ...]`` call.

    MediaPipe dropped the ``mp.solutions`` namespace, so the call this used to
    make (``mp.solutions.face_detection.FaceDetection``) now raises
    AttributeError rather than ImportError on any current install — which the
    old ImportError guard let through as a crash. Prefer the Tasks API and
    keep the legacy path only for old pins.
    """
    import mediapipe as mp

    if hasattr(mp, "tasks"):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} is missing; run: python scripts/fetch_models.py"
            )
        det = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                min_detection_confidence=0.5,
            )
        )

        def detect(rgb):
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            h, w = rgb.shape[:2]
            return [
                ((d.bounding_box.origin_x + d.bounding_box.width / 2) / w,
                 d.bounding_box.width / w)
                for d in res.detections
            ]

        return detect, det.close

    legacy = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    def detect(rgb):
        res = legacy.process(rgb)
        return [
            (d.location_data.relative_bounding_box.xmin
             + d.location_data.relative_bounding_box.width / 2,
             d.location_data.relative_bounding_box.width)
            for d in (res.detections or [])
        ]

    return detect, legacy.close


def _detect_face_centers(
    video: Path, start: float, duration: float, sample_fps: float = 4.0
) -> list[tuple[float, float | None]]:
    """[(t_relative, face_center_x_normalised or None)] — None = no face found."""
    try:
        import cv2
        import mediapipe  # noqa: F401
    except ImportError:
        log.info("mediapipe/opencv missing -> static centre crop")
        return []

    try:
        detect, close = _open_detector()
    except Exception as exc:
        # Never fatal: a centre crop is a worse clip, a crash here throws away
        # the transcript the caller has already paid for.
        log.warning("face detector unavailable (%s) -> static centre crop", exc)
        return []

    cfg = settings()
    cap = cv2.VideoCapture(str(video))
    out: list[tuple[float, float | None]] = []
    try:
        if not cap.isOpened():
            log.warning("could not open %s for face sampling", Path(video).name)
            return []

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if src_fps <= 0 or src_fps > 240:
            src_fps = 30.0
        # One seek to the clip start, then a forward read: seeking per sample
        # re-decodes from the preceding keyframe every time.
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

        stride = max(1, int(round(src_fps / max(sample_fps, 0.1))))
        wanted = int(duration * src_fps)
        index = 0
        while index < wanted:
            ok = cap.grab()          # decode-light: advance without converting
            if not ok:
                break
            if index % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = index / src_fps
                small = _downscale(cv2, frame, cfg.detect_width)
                faces = detect(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                # largest face wins — that's the speaker, not the background
                out.append((t, max(faces, key=lambda f: f[1])[0] if faces else None))
            index += 1
    finally:
        cap.release()
        close()

    found = sum(1 for _, c in out if c is not None)
    log.info("sampled %d frame(s), %d with a face", len(out), found)
    return out


def _downscale(cv2, frame, target_width: int):
    """Detection accuracy is scale-invariant here; speed is not."""
    h, w = frame.shape[:2]
    if target_width <= 0 or w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, max(2, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


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


def even_offset(x: float) -> int:
    """Crop offsets must be even: an odd x on 4:2:0 misaligns the chroma
    planes, and some encoders reject the frame outright."""
    return int(x) // 2 * 2


def crop_width_for(info: VideoInfo) -> int:
    """Widest 9:16 window that fits inside the source frame."""
    return max(2, min(int(info.height * 9 / 16) // 2 * 2, info.width // 2 * 2))


def build_crop_plan(
    video: Path,
    start: float,
    duration: float,
    info: VideoInfo,
    *,
    move_threshold: float = 0.06,   # fraction of frame width before we bother moving
    ease_seconds: float = 0.5,
    sample_fps: float | None = None,
) -> tuple[int, list[tuple[float, int]]]:
    """Returns (crop_width_px, [(time, x_px), ...])."""
    sample_fps = sample_fps if sample_fps else settings().sample_fps
    crop_w = crop_width_for(info)
    max_x = (info.width - crop_w) // 2 * 2
    centred = [(0.0, even_offset(max_x / 2))]

    if max_x <= 0:
        # Already 9:16 or narrower — there is nothing to pan across.
        return crop_w, centred

    samples = _detect_face_centers(video, start, duration, sample_fps)
    if not samples:
        return crop_w, centred

    # carry the last known face through frames where detection dropped
    last = 0.5
    filled = []
    for _, c in samples:
        if c is not None:
            last = c
        filled.append(last)
    smoothed = _smooth(filled)

    def to_x(centre: float) -> int:
        return even_offset(max(0, min(max_x, centre * info.width - crop_w / 2)))

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
                plan.append((round(t + (i - 1) / sample_fps, 3), to_x(v)))
            locked = target
        elif not plan:
            plan.append((0.0, to_x(locked)))

    if not plan:
        plan = [(0.0, to_x(locked))]

    # Keep only strictly increasing timestamps with a changed position. Eases
    # that were still in flight when the next move started emit overlapping
    # timestamps, and sendcmd requires them to be ordered.
    deduped = [plan[0]]
    for t, x in plan[1:]:
        if x != deduped[-1][1] and t > deduped[-1][0]:
            deduped.append((t, x))

    log.info("crop_w=%d moves=%d", crop_w, len(deduped))
    return crop_w, deduped


def static_plan(info: VideoInfo) -> tuple[int, list[tuple[float, int]]]:
    """The ``--no-faces`` path: one centred 9:16 window, no panning."""
    crop_w = crop_width_for(info)
    max_x = max(0, info.width - crop_w)
    return crop_w, [(0.0, even_offset(max_x / 2))]


def write_sendcmd(plan: list[tuple[float, int]], path: Path) -> Path:
    """ffmpeg's sendcmd filter re-parameterises crop at given timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{t:.3f} crop x {x};" for t, x in plan[1:]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
