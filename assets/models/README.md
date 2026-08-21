# Bundled models

`face_detection_full_range_sparse.tflite` — MediaPipe BlazeFace, **full range**,
used by `pipeline/reframe.py` to locate the speaker before building a crop plan.

Full range, not short range, and the distinction decides whether the feature
works at all. Short range is the selfie model: it wants a face filling a good
part of the frame. In a two-person podcast wide shot a head is roughly 8% of
frame width, where short range detects nothing and the crop silently falls back
to centre. Measured on a 1920x1080 frame with a head at 8% width: full range
finds it, short range returns zero detections.

Committed rather than downloaded at runtime: a render worker should not need
egress to Google to reframe a clip, and a fetch that fails midway through a
batch costs the transcript that was already paid for.

Source: <https://storage.googleapis.com/mediapipe-assets/face_detection_full_range_sparse.tflite>
Licence: Apache-2.0, © Google LLC. Refresh with `python scripts/fetch_models.py`.
