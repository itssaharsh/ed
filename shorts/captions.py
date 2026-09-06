"""Stage 9: ASS karaoke captions rendered by libass.

The old pipeline rendered every caption to a PNG with PIL and composited them with moviepy. This
build of ffmpeg has no `drawtext`, and libass is the better tool regardless: real typography,
per-word highlighting, outlines, and scale transforms, all burned in a single pass.

Placement respects the Shorts UI. Top ~150px is the logo/search row, the bottom ~420px carries
the title/channel/subscribe/description, and the right ~140px is the engagement rail. Captions sit
in a band around y=1140 - clear of the chrome, and below the subject's face rather than across it.
"""
from __future__ import annotations

from pathlib import Path

from .config import CAPTION_BAND_Y, HEIGHT, WIDTH, logger

# Words per caption card. 2-3 is the Shorts convention: enough to read in one saccade, few enough
# that the highlight keeps moving.
WORDS_PER_CARD = 3
MIN_CARD_SECONDS = 0.42


def _ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def build_ass(words: list[tuple[float, float, str, bool]], out: Path,
              font: str = "Anton", font_size: int = 118) -> Path:
    """Group word timings into cards with a per-word karaoke highlight."""
    cards: list[list[tuple[float, float, str, bool]]] = []
    cur: list[tuple[float, float, str, bool]] = []
    for w in words:
        cur.append(w)
        # Break early on an emphasised word so the punch word gets its own card and full weight.
        if len(cur) >= WORDS_PER_CARD or (w[3] and len(cur) >= 2):
            cards.append(cur)
            cur = []
    if cur:
        cards.append(cur)

    margin_v = HEIGHT - CAPTION_BAND_Y      # ASS MarginV measures up from the bottom

    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,{font},{font_size},&H00FFFFFF,&H00FFFFFF,&H00101010,&H90000000,-1,0,0,0,100,100,1,0,1,7,3,2,90,150,{margin_v},1
Style: Hit,{font},{font_size},&H0034E5FF,&H0034E5FF,&H00101010,&H90000000,-1,0,0,0,100,100,1,0,1,8,3,2,90,150,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    for card in cards:
        start = card[0][0]
        end = max(card[-1][1], start + MIN_CARD_SECONDS)
        has_emph = any(w[3] for w in card)
        style = "Hit" if has_emph else "Base"

        # \k karaoke units are centiseconds; libass advances SecondaryColour -> PrimaryColour.
        # A short scale-pop on card entry gives the caption a beat of its own.
        pieces = []
        for (ws, we, word, is_emph) in card:
            cs = max(1, int(round((we - ws) * 100)))
            token = _esc(word.upper())
            if is_emph:
                token = "{\\c&H34E5FF&}" + token + "{\\c&HFFFFFF&}"
            pieces.append("{\\k%d}%s" % (cs, token))
        body = " ".join(pieces)

        intro = "{\\fad(70,60)\\t(0,110,\\fscx108\\fscy108)\\t(110,230,\\fscx100\\fscy100)}"
        events.append(
            f"Dialogue: 0,{_ts(start)},{_ts(end)},{style},,0,0,0,,{intro}{body}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + "\n".join(events) + "\n", encoding="utf-8")
    logger.info("stage 9: %d caption cards from %d words -> %s", len(cards), len(words), out.name)
    return out
