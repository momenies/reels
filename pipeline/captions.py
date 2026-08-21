"""Step 4: word timings -> burned-in captions (ASS).

The Arabic problem, which every English-first competitor gets wrong:

Arabic is cursive. Letters change shape depending on their neighbours, and
libass runs HarfBuzz shaping + FriBidi reordering over each *rendered line*.
Word-by-word highlighting works (spaces are natural break points), but the
moment you inject override tags INSIDE a word, or mix Latin and Arabic on one
line with per-word colouring, the bidi algorithm can reorder runs and the
highlight lands on the wrong word.

So: word-level pop for LTR, line-level pop for RTL. Safe, and still looks sharp.
Placement: ASS ignores MarginV for the middle alignments (\\an4-6), so a
centred caption with a big MarginV still lands dead centre. Captions are
positioned with an explicit \\pos() instead, which keeps them in the lower
third and clear of the TikTok/Reels UI chrome.
"""

from __future__ import annotations

from pathlib import Path

ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))

PLAY_RES_X = 1080
PLAY_RES_Y = 1920

#: Caption fonts, committed to the repo. System fonts are not reliable in
#: containers, so the pipeline renders from this directory only.
#: ``scripts/fetch_fonts.py`` regenerates them from upstream.
FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_FILES = {
    "Montserrat ExtraBold": "Montserrat-ExtraBold.ttf",
    "Tajawal ExtraBold": "Tajawal-ExtraBold.ttf",
}


class MissingFontError(RuntimeError):
    """Raised when the font a caption needs is not on disk.

    Worth its own exception: libass silently draws nothing when it cannot
    find a face, and ffmpeg still exits 0. The clip renders, the captions
    are simply absent, and nothing in the pipeline notices. Fail here
    instead, before an hour of GPU time goes into blank output.
    """


def require_font(name: str, fonts_dir: Path | None = None) -> Path:
    """Check that ``name`` is available, and return the file backing it."""
    fonts_dir = Path(fonts_dir) if fonts_dir else FONTS_DIR
    filename = FONT_FILES.get(name)
    if filename is None:
        raise MissingFontError(
            f"{name!r} is not a known font. Put the file in {fonts_dir} and "
            f"register it in captions.FONT_FILES."
        )
    path = fonts_dir / filename
    if not path.exists():
        raise MissingFontError(
            f"{name!r} needs {path}, which is missing. libass would render "
            f"empty captions and ffmpeg would still exit 0. "
            f"Restore it with: python scripts/fetch_fonts.py"
        )
    return path


def is_rtl(text: str) -> bool:
    hits = sum(
        any(lo <= ord(ch) <= hi for lo, hi in ARABIC_RANGES) for ch in text if ch.isalpha()
    )
    letters = sum(1 for ch in text if ch.isalpha())
    return letters > 0 and hits / letters > 0.4


def _ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _header(font: str, size: int, primary: str, outline: str) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_RES_X}
PlayResY: {PLAY_RES_Y}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Pop,{font},{size},{primary},&H000000FF,{outline},&H80000000,-1,0,0,0,100,100,0,0,1,7,3,5,80,80,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _chunk(words, max_words: int, max_gap: float, max_dur: float):
    """Group words into caption lines that breathe with the speech."""
    lines, cur = [], []
    for w in words:
        if cur and (
            len(cur) >= max_words
            or w.start - cur[-1].end > max_gap
            or w.end - cur[0].start > max_dur
        ):
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def build_ass(
    words,
    out_path: Path,
    *,
    font_ltr: str = "Montserrat ExtraBold",
    font_rtl: str = "Tajawal ExtraBold",
    size: int = 92,
    primary: str = "&H00FFFFFF",     # white  (ASS is &HAABBGGRR)
    highlight: str = "&H0000E5FF",   # amber
    outline: str = "&H00000000",
    margin_v: int = 520,
    fonts_dir: Path | None = None,
) -> Path:
    """Writes an .ass file. Word timings must be relative to clip start."""
    if not words:
        out_path.write_text(_header(font_ltr, size, primary, outline), encoding="utf-8")
        return out_path

    rtl = is_rtl(" ".join(w.text for w in words))
    font = font_rtl if rtl else font_ltr
    require_font(font, fonts_dir)
    body = _header(font, size, primary, outline)
    pos = f"\\pos({PLAY_RES_X // 2},{PLAY_RES_Y - margin_v})"

    # RTL fits fewer words per line comfortably at this size
    lines = _chunk(words, max_words=3 if rtl else 4, max_gap=0.7, max_dur=2.6)

    for line in lines:
        l_start, l_end = line[0].start, line[-1].end

        if rtl:
            # one event per line, whole line pops in — no intra-line tags
            text = " ".join(w.text for w in line)
            body += (
                f"Dialogue: 0,{_ts(l_start)},{_ts(l_end)},Pop,,0,0,0,,"
                f"{{\\an5{pos}\\fad(80,80)}}{text}\n"
            )
        else:
            # one event per word: the whole line stays up, active word is amber
            for i, w in enumerate(line):
                parts = []
                for j, x in enumerate(line):
                    parts.append(
                        f"{{\\c{highlight}\\fscx108\\fscy108}}{x.text}{{\\c{primary}\\fscx100\\fscy100}}"
                        if i == j
                        else x.text
                    )
                end = line[i + 1].start if i + 1 < len(line) else l_end
                body += (
                    f"Dialogue: 0,{_ts(w.start)},{_ts(end)},Pop,,0,0,0,,"
                    f"{{\\an5{pos}}}{' '.join(parts)}\n"
                )

    out_path.write_text(body, encoding="utf-8")
    print(f"[captions] {out_path.name} lines={len(lines)} rtl={rtl}")
    return out_path


def build_hook(
    title: str,
    duration: float,
    out_path: Path,
    *,
    font: str = "Tajawal ExtraBold",
    top_y: int = 620,
    fonts_dir: Path | None = None,
) -> Path:
    """Optional: a persistent hook title pinned to the top third."""
    require_font(font, fonts_dir)
    body = _header(font, 76, "&H00FFFFFF", "&H00000000")
    body += (
        f"Dialogue: 0,{_ts(0)},{_ts(duration)},Pop,,0,0,0,,"
        f"{{\\an5\\pos({PLAY_RES_X // 2},{top_y})\\fad(200,200)}}{title}\n"
    )
    out_path.write_text(body, encoding="utf-8")
    return out_path
