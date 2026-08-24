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

Transcript text is never trusted as ASS markup: ``escape()`` neutralises the
brace and backslash characters that would otherwise be parsed as override
tags, which is how a single stray ``{`` in a transcript silently eats the rest
of a caption line.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .config import FONTS_DIR
from .log import get

log = get("captions")

ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))

PLAY_RES_X = 1080
PLAY_RES_Y = 1920

#: Shortest caption event we will emit. Below this libass may drop the event
#: entirely, and a negative one (out-of-order word timings do happen) makes
#: ffmpeg discard everything after it.
MIN_EVENT = 0.06

FONT_FILES = {
    "Montserrat ExtraBold": "Montserrat-ExtraBold.ttf",
    "Tajawal ExtraBold": "Tajawal-ExtraBold.ttf",
}


@dataclass(frozen=True)
class Style:
    """A caption look. ASS colours are ``&HAABBGGRR`` — alpha first, then BGR."""

    name: str
    label: str
    size: int = 92
    primary: str = "&H00FFFFFF"      # resting word colour
    highlight: str = "&H0000E5FF"    # active word colour
    outline_colour: str = "&H00000000"
    back_colour: str = "&H80000000"
    outline: int = 7
    shadow: int = 3
    border_style: int = 1            # 1 = outline+shadow, 3 = opaque box
    scale: int = 108                 # active-word pop, percent
    margin_v: int = 520              # px from the bottom of a 1920-tall frame
    uppercase: bool = False


#: The presets a customer picks from in the studio UI.
STYLES: dict[str, Style] = {
    s.name: s
    for s in (
        Style("pop", "Pop — white with an amber active word"),
        Style(
            "beast",
            "Beast — chunky yellow, heavy outline",
            size=100,
            primary="&H0000FFFF",
            highlight="&H00FFFFFF",
            outline=9,
            shadow=4,
            scale=112,
            uppercase=True,
        ),
        Style(
            "clean",
            "Clean — plain white, thin outline",
            size=84,
            highlight="&H00FFFFFF",
            outline=4,
            shadow=2,
            scale=100,
        ),
        Style(
            "neon",
            "Neon — cyan active word on white",
            size=92,
            highlight="&H00F5FF00",
            outline=6,
            shadow=3,
            scale=110,
        ),
        Style(
            "boxed",
            "Boxed — solid backing plate, high contrast",
            size=80,
            highlight="&H0000E5FF",
            back_colour="&HA0000000",
            border_style=3,
            outline=6,
            shadow=0,
            scale=104,
        ),
    )
}

DEFAULT_STYLE = "pop"


def get_style(name: str | Style | None) -> Style:
    if isinstance(name, Style):
        return name
    if not name:
        return STYLES[DEFAULT_STYLE]
    try:
        return STYLES[str(name).lower()]
    except KeyError:
        raise ValueError(
            f"unknown caption style {name!r}; choose from {', '.join(sorted(STYLES))}"
        ) from None


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


def escape(text: str) -> str:
    """Make transcript text safe to drop into an ASS dialogue line.

    ``{`` opens an override block and ``\\`` starts a tag, so either one
    arriving from a transcript can swallow the rest of the line or inject
    styling nobody asked for. Both are replaced rather than escaped: libass
    accepts ``\\{`` inconsistently across versions, and a lookalike character
    is indistinguishable at caption size. Hard line breaks collapse to spaces
    because line breaking is decided by ``_chunk``, not by the transcript.
    """
    return (
        text.replace("\\", "⧵")   # ⧵ reverse solidus operator
        .replace("{", "(")
        .replace("}", ")")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


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


def _header(style: Style, font: str) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_RES_X}
PlayResY: {PLAY_RES_Y}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Pop,{font},{style.size},{style.primary},&H000000FF,{style.outline_colour},{style.back_colour},-1,0,0,0,100,100,0,0,{style.border_style},{style.outline},{style.shadow},5,80,80,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _clean_words(words):
    """Drop empties, escape the text, and force timings to make sense.

    Whisper emits the occasional word whose end precedes its start, or two
    words with identical timestamps. Either produces a zero or negative
    duration event, and libass stops rendering at the first one.
    """
    out = []
    prev_start = 0.0
    for w in words:
        text = escape(getattr(w, "text", "") or "")
        if not text:
            continue
        start = max(0.0, float(w.start))
        end = float(w.end)
        start = max(start, prev_start)            # keep starts monotonic
        end = max(end, start + MIN_EVENT)         # never zero or negative length
        out.append(replace(w, start=start, end=end, text=text)
                   if hasattr(w, "__dataclass_fields__")
                   else type(w)(start=start, end=end, text=text))
        prev_start = start
    return out


def _chunk(words, max_words: int, max_gap: float, max_dur: float, max_chars: int):
    """Group words into caption lines that breathe with the speech."""
    lines, cur = [], []
    for w in words:
        chars = sum(len(x.text) + 1 for x in cur) + len(w.text)
        if cur and (
            len(cur) >= max_words
            or chars > max_chars
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
    style: str | Style | None = None,
    font_ltr: str = "Montserrat ExtraBold",
    font_rtl: str = "Tajawal ExtraBold",
    size: int | None = None,
    primary: str | None = None,
    highlight: str | None = None,
    outline: str | None = None,
    margin_v: int | None = None,
    fonts_dir: Path | None = None,
) -> Path:
    """Writes an .ass file. Word timings must be relative to clip start.

    ``style`` picks a preset from :data:`STYLES`; the individual keyword
    arguments still win over it, so existing callers keep working.
    """
    st = get_style(style)
    if size is not None:
        st = replace(st, size=size)
    if primary is not None:
        st = replace(st, primary=primary)
    if highlight is not None:
        st = replace(st, highlight=highlight)
    if outline is not None:
        st = replace(st, outline_colour=outline)
    if margin_v is not None:
        st = replace(st, margin_v=margin_v)

    words = _clean_words(words)
    if not words:
        # An empty track still needs the font that the header names to exist,
        # or a later restyle silently produces nothing.
        require_font(font_ltr, fonts_dir)
        out_path.write_text(_header(st, font_ltr), encoding="utf-8")
        return out_path

    rtl = is_rtl(" ".join(w.text for w in words))
    font = font_rtl if rtl else font_ltr
    require_font(font, fonts_dir)
    body = _header(st, font)
    pos = f"\\pos({PLAY_RES_X // 2},{PLAY_RES_Y - st.margin_v})"

    # RTL fits fewer words per line comfortably at this size
    lines = _chunk(
        words,
        max_words=3 if rtl else 4,
        max_gap=0.7,
        max_dur=2.6,
        max_chars=26 if rtl else 32,
    )

    def render_text(text: str) -> str:
        return text.upper() if st.uppercase and not rtl else text

    for line in lines:
        l_start, l_end = line[0].start, max(w.end for w in line)

        if rtl:
            # one event per line, whole line pops in — no intra-line tags
            text = render_text(" ".join(w.text for w in line))
            body += (
                f"Dialogue: 0,{_ts(l_start)},{_ts(l_end)},Pop,,0,0,0,,"
                f"{{\\an5{pos}\\fad(80,80)}}{text}\n"
            )
        else:
            # one event per word: the whole line stays up, active word is amber
            for i, w in enumerate(line):
                parts = []
                for j, x in enumerate(line):
                    txt = render_text(x.text)
                    parts.append(
                        f"{{\\c{st.highlight}\\fscx{st.scale}\\fscy{st.scale}}}{txt}"
                        f"{{\\c{st.primary}\\fscx100\\fscy100}}"
                        if i == j
                        else txt
                    )
                end = line[i + 1].start if i + 1 < len(line) else l_end
                end = max(end, w.start + MIN_EVENT)
                body += (
                    f"Dialogue: 0,{_ts(w.start)},{_ts(end)},Pop,,0,0,0,,"
                    f"{{\\an5{pos}}}{' '.join(parts)}\n"
                )

    out_path.write_text(body, encoding="utf-8")
    log.info("%s lines=%d rtl=%s style=%s", out_path.name, len(lines), rtl, st.name)
    return out_path


def build_hook(
    title: str,
    duration: float,
    out_path: Path,
    *,
    font: str | None = None,
    top_y: int = 620,
    size: int = 76,
    max_chars: int = 24,
    fonts_dir: Path | None = None,
) -> Path:
    """A persistent hook title pinned to the top third.

    ``font`` defaults to whichever face matches the title's script, so an
    English title does not get rendered in the Arabic face (and vice versa,
    which draws nothing at all).
    """
    title = escape(title)
    if font is None:
        font = "Tajawal ExtraBold" if is_rtl(title) else "Montserrat ExtraBold"
    require_font(font, fonts_dir)

    # wrap long hooks rather than letting them run off the frame
    lines, cur = [], ""
    for word in title.split():
        candidate = f"{cur} {word}".strip()
        if cur and len(candidate) > max_chars:
            lines.append(cur)
            cur = word
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    text = "\\N".join(lines[:3])

    st = replace(get_style("clean"), size=size)
    body = _header(st, font)
    body += (
        f"Dialogue: 0,{_ts(0)},{_ts(max(duration, MIN_EVENT))},Pop,,0,0,0,,"
        f"{{\\an5\\pos({PLAY_RES_X // 2},{top_y})\\fad(200,200)}}{text}\n"
    )
    out_path.write_text(body, encoding="utf-8")
    return out_path
