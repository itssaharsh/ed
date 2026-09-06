"""Stages 1-5: premise -> script -> punch-up -> vocal direction.

The whole design rests on two research findings (docs/RESEARCH.md section 4):
  * LLMs write good setups but rarely good punchlines, so we generate many and select hard.
  * Absolute humour scoring is noise, so every selection is a pairwise tournament.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, asdict, field
from typing import Any

from . import prompts
from .config import (
    CATEGORIES, MAX_DIRECTIONS, MAX_NONVERBALS, MAX_PAUSE_MS, N_PREMISES, N_PUNCHLINES,
    ORPHEUS_CHAR_LIMIT, TARGET_SECONDS, TARGET_WORDS, TOURNAMENT_ROUNDS, Category, logger,
)
from .llm import LLM, LLMError
from .tournament import run_tournament

BEAT_ORDER = ("hook", "setup", "escalate", "turn", "punch", "tag")


@dataclass
class Premise:
    situation: str
    turn: str
    detail: str
    mechanism: str
    target: str
    has_screen: bool = False
    has_other_people: bool = False

    def text(self) -> str:
        return f"{self.situation} {self.turn}"

    def prompt_block(self) -> str:
        return (
            f"Situation: {self.situation}\n"
            f"Where it goes: {self.turn}\n"
            f"The specific detail: {self.detail}\n"
            f"Mechanism: {self.mechanism}"
        )


@dataclass
class Line:
    index: int
    role: str
    text: str
    direction: str | None = None
    pause_before_ms: int = 0
    emphasis: list[str] = field(default_factory=list)
    # filled in by the voice stage
    audio_path: str | None = None
    duration: float = 0.0
    start: float = 0.0


def pick_category(rng: random.Random) -> Category:
    return rng.choices(CATEGORIES, weights=[c.weight for c in CATEGORIES], k=1)[0]


def _judge_factory(llm: LLM, item_kind: str, context: str):
    """Build a pairwise judge callable for the tournament.

    Note it renders 02_tournament.md WITHOUT the comedian persona: a judge carrying the writing
    persona reliably prefers its own voice, which collapses the tournament to noise.
    """
    def judge(a: str, b: str) -> dict:
        p = prompts.render("02_tournament", ITEM_KIND=item_kind, CONTEXT=context, A=a, B=b, VOICE="")
        out = llm.complete_json(p, temperature=0.25)
        if not isinstance(out, dict) or "winner" not in out:
            raise LLMError(f"judge returned malformed verdict: {str(out)[:160]}")
        return out
    return judge


# ── Stage 1-2: premises ─────────────────────────────────────────────────────

def generate_premises(llm: LLM, category: Category, recent: list[str]) -> list[Premise]:
    recent_block = "\n".join(f"- {r}" for r in recent[:40]) or "- (nothing yet)"
    p = prompts.render(
        "01_ideate",
        CATEGORY=category.id, CATEGORY_BRIEF=category.brief,
        RECENT_PREMISES=recent_block, N=N_PREMISES,
    )
    data = llm.complete_json(p, temperature=1.0)
    raw = data.get("premises") if isinstance(data, dict) else data
    if not isinstance(raw, list) or not raw:
        raise LLMError("ideation returned no premises")

    out: list[Premise] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        situation = str(item.get("situation", "")).strip()
        turn = str(item.get("turn", "")).strip()
        if not situation or not turn:
            continue
        out.append(Premise(
            situation=situation,
            turn=turn,
            detail=str(item.get("detail", "")).strip(),
            mechanism=str(item.get("mechanism", "")).strip().lower(),
            target=str(item.get("target", "")).strip().lower(),
            has_screen=bool(item.get("has_screen", False)),
            has_other_people=bool(item.get("has_other_people", False)),
        ))
    if not out:
        raise LLMError("ideation produced no usable premises")
    logger.info("stage 1: %d premises generated", len(out))
    return out


def select_premise(llm: LLM, premises: list[Premise], rng: random.Random) -> tuple[Premise, list]:
    judge = _judge_factory(
        llm, "comedy premises for a 40-second spoken bit",
        "Each candidate is a premise: the situation, where it escalates to, and its specific detail. "
        "You are judging which would make the better finished bit, not which is better written.",
    )
    texts = [p.prompt_block() for p in premises]
    winner, standings, bouts = run_tournament(texts, judge, rounds=TOURNAMENT_ROUNDS, rng=rng)
    logger.info("stage 2: premise %d wins after %d bouts", winner, len(bouts))
    return premises[winner], bouts


# ── Stage 3: script ─────────────────────────────────────────────────────────

def _clean_spoken(text: str) -> str:
    """Strip anything a TTS engine would read aloud as literal characters."""
    t = re.sub(r"\[[^\]]*\]", " ", text)          # stray stage directions
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"[*_#`]", "", t)
    t = t.replace("—", ", ").replace("–", ", ")
    t = re.sub(r"\.{2,}", ".", t)
    return re.sub(r"\s+", " ", t).strip()


def draft_script(llm: LLM, premise: Premise) -> tuple[list[dict], str]:
    p = prompts.render(
        "03_script",
        PREMISE=premise.prompt_block(),
        TARGET_SECONDS=TARGET_SECONDS, TARGET_WORDS=TARGET_WORDS,
    )
    data = llm.complete_json(p, temperature=0.95)
    beats = data.get("beats") if isinstance(data, dict) else data
    if not isinstance(beats, list) or len(beats) < 4:
        raise LLMError(f"script draft returned {len(beats) if isinstance(beats, list) else 0} beats")

    cleaned: list[dict] = []
    for b in beats:
        if not isinstance(b, dict):
            continue
        text = _clean_spoken(str(b.get("text", "")))
        role = str(b.get("role", "escalate")).strip().lower()
        if not text:
            continue
        if role not in BEAT_ORDER:
            role = "escalate"
        cleaned.append({"role": role, "text": text})

    if not any(b["role"] == "punch" for b in cleaned):
        cleaned[-1]["role"] = "punch"          # the last line is the punch by definition
    the_joke = str(data.get("the_joke", "")).strip() if isinstance(data, dict) else ""
    logger.info("stage 3: %d beats, %d words", len(cleaned), sum(len(b["text"].split()) for b in cleaned))
    return cleaned, the_joke


# ── Stage 4: punch-up ───────────────────────────────────────────────────────

def punch_up(llm: LLM, beats: list[dict], the_joke: str, rng: random.Random) -> list[dict]:
    """Generate alternative punchlines and tournament them against the incumbent."""
    punch_idx = max(i for i, b in enumerate(beats) if b["role"] == "punch")
    script_text = "\n".join(f"[{b['role']}] {b['text']}" for b in beats)

    p = prompts.render("04_punchup", SCRIPT=script_text, THE_JOKE=the_joke, N=N_PUNCHLINES)
    try:
        data = llm.complete_json(p, temperature=1.0)
    except LLMError as exc:
        logger.warning("punch-up generation failed, keeping original ending: %s", exc)
        return beats

    cands = data.get("candidates") if isinstance(data, dict) else data
    options = [beats[punch_idx]["text"]]           # incumbent competes
    for c in cands or []:
        if isinstance(c, dict):
            t = _clean_spoken(str(c.get("text", "")))
            if t and t.lower() not in {o.lower() for o in options}:
                options.append(t)

    if len(options) < 2:
        return beats

    setup = "\n".join(f"[{b['role']}] {b['text']}" for b in beats[:punch_idx])
    judge = _judge_factory(
        llm, "final punchlines for the same comedy bit",
        f"Both candidates are the LAST line of this bit. Everything before them is identical:\n\n"
        f"{setup}\n\nJudge only which ending lands harder.",
    )
    winner, _, bouts = run_tournament(options, judge, rounds=TOURNAMENT_ROUNDS, rng=rng)
    if winner != 0:
        logger.info("stage 4: punchline replaced after %d bouts", len(bouts))
        beats = [dict(b) for b in beats]
        beats[punch_idx]["text"] = options[winner]
    else:
        logger.info("stage 4: original punchline survived %d bouts", len(bouts))
    return beats


# ── Stage 5: vocal direction ────────────────────────────────────────────────

_NONVERBAL = re.compile(r"<(laugh|sigh|giggle|groan|chuckle|gasp|cough|sniff|yawn)>", re.I)


def direct(llm: LLM, beats: list[dict], voice_name: str) -> list[Line]:
    beats_block = "\n".join(f"{i}. [{b['role']}] {b['text']}" for i, b in enumerate(beats))
    p = prompts.render("05_delivery", BEATS=beats_block, VOICE_NAME=voice_name)
    try:
        data = llm.complete_json(p, temperature=0.5)
        raw = data.get("lines") if isinstance(data, dict) else data
    except LLMError as exc:
        logger.warning("direction failed, falling back to undirected read: %s", exc)
        raw = None

    if not isinstance(raw, list) or not raw:
        logger.warning("stage 5: no usable direction returned; using default timing "
                       "(delivery will be flatter than intended)")
        return _default_direction(beats)

    by_index = {}
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            by_index[item["index"]] = item

    lines = _default_direction(beats)
    directed = 0
    nonverbals = 0
    for ln in lines:
        item = by_index.get(ln.index)
        if not item:
            continue

        # The model directs; it does not rewrite. Any text edit is discarded.
        proposed = _clean_spoken(str(item.get("text", "")))
        if proposed and proposed.lower() != ln.text.lower():
            logger.warning("line %d: direction stage altered text, keeping original", ln.index)

        d = item.get("direction")
        d = str(d).strip().strip("[]").lower() if d else None
        if d and directed < MAX_DIRECTIONS:
            ln.direction = d
            directed += 1

        try:
            pause = int(item.get("pause_before_ms", ln.pause_before_ms))
        except (TypeError, ValueError):
            pause = ln.pause_before_ms
        ln.pause_before_ms = max(0, min(pause, MAX_PAUSE_MS))
        if ln.index == 0:
            ln.pause_before_ms = 0            # never make the viewer wait at the start

        emph = item.get("emphasis") or []
        if isinstance(emph, list):
            words = {w.strip(".,!?").upper() for w in ln.text.split()}
            ln.emphasis = [str(e).strip().upper() for e in emph
                           if str(e).strip().upper() in words][:3]

        found = _NONVERBAL.findall(ln.text)
        if found:
            nonverbals += len(found)
            if nonverbals > MAX_NONVERBALS:
                ln.text = _NONVERBAL.sub("", ln.text).strip()

    for ln in lines:
        if len(ln.text) > ORPHEUS_CHAR_LIMIT:
            logger.warning("line %d exceeds %d chars; it will be split for synthesis",
                           ln.index, ORPHEUS_CHAR_LIMIT)
    logger.info("stage 5: %d/%d lines directed, pauses %s",
                directed, len(lines), [l.pause_before_ms for l in lines])
    return lines


def _default_direction(beats: list[dict]) -> list[Line]:
    """Sensible timing with no model involvement — also the fallback path."""
    defaults = {"hook": 0, "setup": 120, "escalate": 110, "turn": 260, "punch": 520, "tag": 320}
    lines: list[Line] = []
    for i, b in enumerate(beats):
        lines.append(Line(
            index=i, role=b["role"], text=b["text"],
            pause_before_ms=0 if i == 0 else defaults.get(b["role"], 120),
        ))
    return lines


def script_text(lines: list[Line]) -> str:
    return " ".join(l.text for l in lines)
