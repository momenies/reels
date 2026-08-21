"""Fetch the caption fonts into assets/fonts/.

Run once after cloning:

    python scripts/fetch_fonts.py

Both faces come from google/fonts under the SIL Open Font License. Tajawal
ships a static ExtraBold; Montserrat only ships a variable font, so its
ExtraBold instance is generated here and renamed, because libass matches on
family name and would otherwise fall back to the Regular weight.

The fonts are fetched rather than committed to keep the repo text-only, but
the pipeline still renders from assets/fonts/ alone — see captions.FONTS_DIR
for why system fonts are not trusted.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "assets" / "fonts"
BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl"

DOWNLOADS = {
    "Tajawal-ExtraBold.ttf": f"{BASE}/tajawal/Tajawal-ExtraBold.ttf",
    "OFL-Tajawal.txt": f"{BASE}/tajawal/OFL.txt",
    "OFL-Montserrat.txt": f"{BASE}/montserrat/OFL.txt",
}
MONTSERRAT_VF = f"{BASE}/montserrat/Montserrat%5Bwght%5D.ttf"


def get(url: str, dest: Path) -> Path:
    print(f"  {dest.name}")
    with urllib.request.urlopen(url) as r:
        dest.write_bytes(r.read())
    return dest


def build_montserrat_extrabold(dest: Path) -> None:
    """Pin the variable font at wght=800 and name the result what ASS asks for."""
    try:
        from fontTools import ttLib
        from fontTools.varLib import instancer
    except ImportError:
        sys.exit(
            "Montserrat needs fonttools to instance the variable font:\n"
            "    pip install 'fonttools[woff]'"
        )

    vf = dest.parent / "Montserrat-variable.ttf"
    get(MONTSERRAT_VF, vf)

    font = instancer.instantiateVariableFont(
        ttLib.TTFont(vf), {"wght": 800}, updateFontNames=True
    )
    name = font["name"]
    for rec in list(name.names):
        if rec.nameID in (1, 4):
            name.setName("Montserrat ExtraBold", rec.nameID,
                         rec.platformID, rec.platEncID, rec.langID)
        elif rec.nameID == 2:
            name.setName("Regular", 2, rec.platformID, rec.platEncID, rec.langID)
        elif rec.nameID == 6:
            name.setName("Montserrat-ExtraBold", 6,
                         rec.platformID, rec.platEncID, rec.langID)
    # typographic family/subfamily would otherwise re-introduce plain "Montserrat"
    for nid in (16, 17, 21, 22):
        name.removeNames(nameID=nid)

    font.save(dest)
    vf.unlink()
    print(f"  {dest.name} (instanced at wght=800)")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Fetching fonts into {DEST}")
    for filename, url in DOWNLOADS.items():
        get(url, DEST / filename)
    build_montserrat_extrabold(DEST / "Montserrat-ExtraBold.ttf")
    print("Done.")


if __name__ == "__main__":
    main()
