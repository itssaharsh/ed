"""Stage 6: directed lines -> shot list bound to script beats.

This replaces the old keyword-roulette approach (8 "visual irony" terms thrown at a stock library).
Every shot here is tied to a `line_index`, so a cut lands on a comic beat rather than near one.
"""
from __future__ import annotations

import random
from typing import Any

from . import prompts
from .config import DEFAULT_STYLE, MECHANISM_STYLE, logger
from .llm import LLM, LLMError
from .write import Line

VALID_MOTION = ("push-in", "pull-out", "drift-left", "drift-right", "static-float")
VALID_SIZE = ("wide", "medium", "close", "extreme-close", "over-shoulder")


def choose_style(mechanism: str, rng: random.Random) -> str:
    """Style follows the joke mechanism so the look matches the humour, rather than being random."""
    for key, style in MECHANISM_STYLE.items():
        if key in (mechanism or "").lower():
            return style
    return DEFAULT_STYLE


def _dedupe_motion(shots: list[dict], rng: random.Random) -> list[dict]:
    """No two adjacent shots may share a camera move — that is what makes it read as a slideshow."""
    prev = None
    for s in shots:
        m = s.get("motion")
        if m not in VALID_MOTION:
            m = rng.choice(VALID_MOTION)
        if m == prev:
            alternatives = [x for x in VALID_MOTION if x != prev]
            m = rng.choice(alternatives)
        s["motion"] = m
        prev = m
    return shots


def build_shot_list(llm: LLM, lines: list[Line], style: str, total_seconds: float,
                    rng: random.Random) -> tuple[list[dict], str, str, str]:
    """Returns (shots, character_sheet, generation_suffix, negative_prompt).

    Note the third value is the SHORT suffix, not the long contract: the long form is art
    direction for the shot-list model, the short form is what the image model actually receives.
    """
    contract, negative = prompts.style_contract(style)
    suffix = prompts.style_suffix(style)
    palette = prompts.style_palette(style)
    contract_block = f"{contract}\n\nPalette: {palette}"

    lines_block = "\n".join(
        f"{l.index}. [{l.role}] {l.text}" for l in lines
    )
    p = prompts.render(
        "06_shotlist",
        LINES=lines_block, STYLE_NAME=style, STYLE_CONTRACT=contract_block,
        TOTAL_SECONDS=round(total_seconds, 1), VOICE="",
    )
    try:
        data = llm.complete_json(p, temperature=0.85)
    except LLMError as exc:
        logger.warning("shot list generation failed, using per-line fallback: %s", exc)
        data = {}

    raw = data.get("shots") if isinstance(data, dict) else data
    character = str(data.get("character_sheet", "")).strip() if isinstance(data, dict) else ""

    shots: list[dict] = []
    valid_indices = {l.index for l in lines}
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        prompt_text = str(item.get("prompt", "")).strip()
        if not prompt_text:
            continue
        try:
            li = int(item.get("line_index", 0))
        except (TypeError, ValueError):
            li = 0
        if li not in valid_indices:
            li = min(valid_indices, key=lambda x: abs(x - li))
        size = str(item.get("shot_size", "")).strip().lower()
        shots.append({
            "line_index": li,
            "shot_size": size if size in VALID_SIZE else "medium",
            "prompt": prompt_text,
            "motion": str(item.get("motion", "")).strip().lower(),
            "why_this_image": str(item.get("why_this_image", "")).strip(),
        })

    if not shots:
        # Fallback: one literal shot per line. Weak, but never leaves the video without frames.
        logger.warning("no usable shots returned; falling back to one shot per line")
        shots = [{
            "line_index": l.index,
            "shot_size": "medium",
            "prompt": f"medium shot illustrating: {l.text}",
            "motion": "",
            "why_this_image": "fallback",
        } for l in lines]

    shots.sort(key=lambda s: s["line_index"])

    # Inject the character sheet so the same person appears across shots rather than a new
    # stranger every cut. Visual incoherence between shots is the biggest cheap-AI tell.
    if character:
        for s in shots:
            if any(w in s["prompt"].lower() for w in ("man", "woman", "person", "he ", "she ", "they ")):
                s["prompt"] = f"{s['prompt']} The person is {character}"

    shots = _dedupe_motion(shots, rng)
    logger.info("stage 6: %d shots, style=%s, character=%s",
                len(shots), style, "yes" if character else "none")
    return shots, character, suffix, negative


def assign_timing(shots: list[dict], lines: list[Line], total: float) -> list[dict]:
    """Give every shot a start/duration derived from the measured audio.

    Timing flows out of the performance, not into it: each line's real spoken duration decides
    how long its shot is held, so cuts land on beats instead of on a fixed grid.
    """
    by_line: dict[int, list[dict]] = {}
    for s in shots:
        by_line.setdefault(s["line_index"], []).append(s)

    timed: list[dict] = []
    carried = 0.0            # time from lines that have no shot of their own
    for line in lines:
        # The pause before a line belongs to the shot that follows it: the silence before the
        # punch should sit on the punch's image, not linger on the previous one.
        start = line.start - (line.pause_before_ms / 1000.0)
        span = line.duration + (line.pause_before_ms / 1000.0)

        group = by_line.get(line.index)
        if not group:
            # No shot for this line: hold the previous image across it rather than dropping the
            # time, which would desynchronise the video from the audio. This is what lets a brief
            # use fewer shots than lines — useful when image generation is the slow step.
            if timed:
                timed[-1]["duration"] += max(0.0, span)
            else:
                carried += max(0.0, span)
            continue

        if span <= 0:
            continue
        span += carried          # absorb any leading uncovered lines
        start -= carried
        carried = 0.0
        each = span / len(group)
        for k, shot in enumerate(group):
            timed.append({**shot, "start": max(0.0, start + k * each), "duration": each})

    if not timed:
        return timed

    timed.sort(key=lambda s: s["start"])
    # Close any gaps so the video never shows black between shots.
    for i in range(len(timed) - 1):
        timed[i]["duration"] = max(0.4, timed[i + 1]["start"] - timed[i]["start"])
    timed[-1]["duration"] = max(0.5, total - timed[-1]["start"])
    return timed
