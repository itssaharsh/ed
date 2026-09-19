"""Stage 6: directed lines -> shot list bound to script beats.

This replaces the old keyword-roulette approach (8 "visual irony" terms thrown at a stock library).
Every shot here is tied to a `line_index`, so a cut lands on a comic beat rather than near one.
"""
from __future__ import annotations

import math
import random
import re
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


# ── People versus objects ───────────────────────────────────────────────────
#
# The first real render made every one of its seven shots a portrait of the same man. Three
# causes, all mechanical: the prompt told the model to copy the character sheet "verbatim into
# every prompt", this module then appended it a second time, and person detection was
# `"man" in prompt` - which also matches "command", "manual" and "human". Weak image models weight
# the opening words, so a prompt that *starts* with seventeen words of character description
# comes back as a portrait whatever the rest of it asks for.

_PERSON_RE = re.compile(
    r"\b(man|men|woman|women|person|people|guy|girl|boy|lady|gentleman|kid|child|"
    r"he|she|they|him|her|his|hers|neighbou?r|colleague|coworker|flatmate|roommate|"
    r"friend|husband|wife|mum|mom|dad|father|mother)\b", re.I)
_HYPHENS = re.compile(r"[\u2010-\u2015\u2212]")
MIN_INSERT_FRACTION = 1 / 3


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", _HYPHENS.sub("-", text)).strip().lower()


def features_person(prompt: str, character: str = "") -> bool:
    """Whole-word match, so "command centre" and "manual" are not people."""
    if character and _norm(character) and _norm(character) in _norm(prompt):
        return True
    return bool(_PERSON_RE.search(prompt))


def _handle(character: str) -> str:
    """Short noun for the recurring person: 'the man', 'the woman', else 'the person'."""
    m = _PERSON_RE.search(character or "")
    noun = m.group(1).lower() if m else "person"
    if noun in {"he", "him", "his", "they", "she", "her", "hers"}:
        noun = "person"
    return f"the {noun}"


def _strip_character(prompt: str, character: str) -> str:
    """Replace an inlined copy of the character sheet with a short handle."""
    if not character:
        return prompt
    norm_char = _norm(character).rstrip(".")
    # re.escape turns each space into "\\ " (it escapes whitespace for VERBOSE-mode safety), so
    # swap those for \\s+ to tolerate the model's spacing. tests/test_units.py guards this.
    pattern = re.compile(re.escape(norm_char).replace(r"\ ", r"\s+"), re.I)
    return pattern.sub(_handle(character), _HYPHENS.sub("-", prompt)).strip()


def _attach_character(prompt: str, character: str) -> str:
    """Put the object/action first and the character last, once."""
    if not character or not features_person(prompt, character):
        return prompt
    body = _strip_character(prompt, character).rstrip(" .")
    return f"{body}. {_handle(character).capitalize()} is {character.strip().rstrip('.')}."


def insert_shortfall(shots: list[dict], character: str = "") -> int:
    """How many more no-person insert shots this list needs to meet MIN_INSERT_FRACTION."""
    if not shots:
        return 0
    need = math.ceil(len(shots) * MIN_INSERT_FRACTION)
    have = sum(1 for s in shots if not features_person(s["prompt"], character))
    return max(0, need - have)


def _parse_shots(raw: Any, lines: list[Line]) -> list[dict]:
    """Validate the model's shot list into clean dicts. Tolerates junk; never raises."""
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
    def ask(repair: str) -> dict:
        p = prompts.render(
            "06_shotlist",
            LINES=lines_block, STYLE_NAME=style, STYLE_CONTRACT=contract_block,
            TOTAL_SECONDS=round(total_seconds, 1), VOICE="", REPAIR=repair,
        )
        try:
            out = llm.complete_json(p, temperature=0.85)
        except LLMError as exc:
            logger.warning("shot list generation failed: %s", exc)
            return {}
        return out if isinstance(out, dict) else {"shots": out}

    data = ask("")
    character = str(data.get("character_sheet", "")).strip()
    shots = _parse_shots(data.get("shots"), lines)

    # Enforced in code, not left to the prompt: the prompt already asked for inserts and the
    # first real render still came back as seven portraits. One targeted re-ask names the
    # measured shortfall; if that is no better, keep whichever list is closer and carry on - a
    # shot-composition miss is not worth failing the run over.
    short = insert_shortfall(shots, character)
    if shots and short:
        have = len(shots) - sum(1 for s in shots if features_person(s["prompt"], character))
        logger.warning("stage 6: only %d of %d shots are object inserts; asking again", have,
                       len(shots))
        retry = ask(
            f"CORRECTION: your previous shot list had {have} insert shot(s) out of "
            f"{len(shots)}. At least {have + short} must have no person in frame at all - "
            f"close-ups of the specific objects the lines name. Return the full list again."
        )
        retry_shots = _parse_shots(retry.get("shots"), lines)
        retry_char = str(retry.get("character_sheet", "")).strip() or character
        if retry_shots and insert_shortfall(retry_shots, retry_char) < short:
            shots, character = retry_shots, retry_char
        else:
            logger.warning("stage 6: re-ask did not add inserts; keeping the original list")

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
            s["prompt"] = _attach_character(s["prompt"], character)

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
