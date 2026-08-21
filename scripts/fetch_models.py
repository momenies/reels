"""Refresh the bundled face-detection model in assets/models/.

The model is committed, so a clone reframes with no network. Run this only to
pull a newer copy from upstream:

    python scripts/fetch_models.py

Full range is deliberate — see assets/models/README.md. Swapping in the short
range model makes reframing silently do nothing on normal wide-shot footage.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "assets" / "models"
MODELS = {
    "face_detection_full_range_sparse.tflite":
        "https://storage.googleapis.com/mediapipe-assets/"
        "face_detection_full_range_sparse.tflite",
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        path = DEST / name
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
        print(f"  {name}  {path.stat().st_size} bytes")
    print("Done.")


if __name__ == "__main__":
    main()
