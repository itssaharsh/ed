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
# that the highlight keeps moving. This is a *maximum*, not a quota - see _should_break.
WORDS_PER_CARD = 3
MIN_CARD_SECONDS = 0.42

# Punctuation that ends a thought. A card must never span one of these.
_SENTENCE_END = (".", "!", "?", "…")
_CLAUSE_END = (",", ";", ":", "-")


def _ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def _should_break(cur: list[tuple[float, float, str, bool]],
                  w: tuple[float, float, str, bool]) -> bool:
    """Decide whether the card ends after word `w`.

    Grouping used to be a flat every-third-word counter, which regularly produced cards that
    straddled a sentence end - "FRIDGE ITEMS. NOT", "THE TAPE. THE", "DATE. EVERY SINGLE" were all
    real output. On a Short that is worse than untidy: the caption is the *visual* beat, and
    gluing the end of one thought to the start of the next steps on the timing the delivery stage
    worked to create. The audio already pauses there; the caption should too.

    Priority order: a finished sentence always breaks, an emphasised word breaks (so the punch
    word carries its own card), a clause break lands if the card is already readable, and the
    word cap is the last resort rather than the rule.
    """
    text = w[2].rstrip()
    if text.endswith(_SENTENCE_END):
        return True
    if w[3] and len(cur) >= 2:            # emphasised word gets full weight
        return True
    if text.endswith(_CLAUSE_END) and len(cur) >= 2:
        return True
    return len(cur) >= WORDS_PER_CARD


def build_ass(words: list[tuple[float, float, str, bool]], out: Path,
              font: str = "Anton", font_size: int = 118) -> Path:
    """Group word timings into cards with a per-word karaoke highlight."""
    cards: list[list[tuple[float, float, str, bool]]] = []
    cur: list[tuple[float, float, str, bool]] = []
    for w in words:
        cur.append(w)
        if _should_break(cur, w):
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
