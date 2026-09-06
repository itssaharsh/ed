"""Hand-authored briefs — run the pipeline with a human (or Claude) as the writer.

Stages 1-6 exist to make an LLM produce a premise, a script, a performance direction and a shot
list. A brief supplies all four directly, so the pipeline can render without any LLM API key at
all. That makes this useful in three situations:

  * No key, or the key's daily quota is spent.
  * You want to write the comedy yourself and keep the machinery.
  * Claude (or any assistant) is acting as the writer and hands over finished JSON.

A brief is validated strictly on load. A malformed brief fails loudly here rather than producing
a subtly broken video six stages later.

Schema (see briefs/ for worked examples):

    {
      "premise":   {"situation", "turn", "detail", "mechanism", "target"},
      "style":     "flat_absurd" | "grain_docu" | "neon_late",
      "character_sheet": "...",            # optional; appended to shots featuring a person
      "beats":     [{"role", "text"}],     # role: hook|setup|escalate|turn|punch|tag
      "direction": [{"index", "direction", "pause_before_ms", "emphasis"}],   # optional
      "shots":     [{"line_index", "shot_size", "prompt", "motion"}],
      "metadata":  {"title", "description_hook", "hashtags", "tags"}
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import prompts
from .config import MAX_DURATION, MAX_PAUSE_MS, MIN_DURATION, ORPHEUS_CHAR_LIMIT, logger
from .visuals import VALID_MOTION, VALID_SIZE
from .write import BEAT_ORDER, Line, Premise, _clean_spoken


class BriefError(ValueError):
    pass


@dataclass
class Brief:
    path: Path
    premise: Premise
    style: str
    character_sheet: str
    lines: list[Line]
    shots: list[dict]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def script(self) -> str:
        return " ".join(l.text for l in self.lines)

    @property
    def word_count(self) -> int:
        return len(self.script.split())


def _require(d: dict, key: str, where: str) -> Any:
    if key not in d:
        raise BriefError(f"{where}: missing required key {key!r}")
    return d[key]


def load(path: Path) -> Brief:
    """Parse and validate a brief. Raises BriefError with a specific reason on any problem."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BriefError(f"{path.name}: not valid JSON - {exc}") from exc
    if not isinstance(data, dict):
        raise BriefError(f"{path.name}: top level must be an object")

    # ── premise ─────────────────────────────────────────────────────────────
    p = _require(data, "premise", path.name)
    if not isinstance(p, dict):
        raise BriefError(f"{path.name}: 'premise' must be an object")
    premise = Premise(
        situation=str(_require(p, "situation", "premise")).strip(),
        turn=str(_require(p, "turn", "premise")).strip(),
        detail=str(p.get("detail", "")).strip(),
        mechanism=str(p.get("mechanism", "")).strip().lower(),
        target=str(p.get("target", "")).strip().lower(),
    )
    if not premise.situation or not premise.turn:
        raise BriefError(f"{path.name}: premise.situation and premise.turn must be non-empty")

    # ── style ───────────────────────────────────────────────────────────────
    style = str(data.get("style", "")).strip() or "flat_absurd"
    available = prompts.available_styles()
    if style not in available:
        raise BriefError(f"{path.name}: unknown style {style!r} (have: {', '.join(available)})")

    # ── beats ───────────────────────────────────────────────────────────────
    beats = _require(data, "beats", path.name)
    if not isinstance(beats, list) or len(beats) < 4:
        raise BriefError(f"{path.name}: need at least 4 beats, got {len(beats) if isinstance(beats, list) else 0}")

    cleaned: list[dict] = []
    for i, b in enumerate(beats):
        if not isinstance(b, dict):
            raise BriefError(f"{path.name}: beat {i} is not an object")
        text = _clean_spoken(str(_require(b, "text", f"beat {i}")))
        if not text:
            raise BriefError(f"{path.name}: beat {i} has empty text after cleaning")
        if len(text) > ORPHEUS_CHAR_LIMIT:
            logger.warning("%s: beat %d is %d chars and will be split for synthesis",
                           path.name, i, len(text))
        role = str(b.get("role", "escalate")).strip().lower()
        if role not in BEAT_ORDER:
            raise BriefError(f"{path.name}: beat {i} has unknown role {role!r} "
                             f"(have: {', '.join(BEAT_ORDER)})")
        cleaned.append({"role": role, "text": text})

    # Estimate spoken length before anything expensive happens. Image generation can take ten
    # minutes on a free tier, and spending that only for the gate to reject the video for being
    # four seconds short is maddening. Calibrated against a real render: 84 words plus 1.38s of
    # designed pauses produced 21.7s, so ~0.235s per word plus the pause budget.
    #
    # A 1s tolerance keeps the check honest about its own error bars - it should only stop briefs
    # that clearly cannot pass, and warn about the ones near the line.
    PAUSE_DEFAULTS = {"hook": 0, "setup": 130, "escalate": 110, "turn": 280,
                      "punch": 540, "tag": 320}
    words = sum(len(b["text"].split()) for b in cleaned)
    pause_total = sum(PAUSE_DEFAULTS.get(b["role"], 120) for b in cleaned[1:]) / 1000.0
    est = words * 0.235 + pause_total
    TOLERANCE = 1.0

    if est < MIN_DURATION - TOLERANCE:
        short_by = MIN_DURATION - est
        raise BriefError(
            f"{path.name}: ~{words} words is about {est:.0f}s of speech, below the "
            f"{MIN_DURATION:.0f}s minimum the quality gate enforces. Add roughly "
            f"{int(short_by / 0.235) + 5} more words - another beat, not padding."
        )
    if est > MAX_DURATION + TOLERANCE:
        raise BriefError(
            f"{path.name}: ~{words} words is about {est:.0f}s, above the {MAX_DURATION:.0f}s "
            f"maximum. Cut a beat."
        )
    if est < MIN_DURATION + TOLERANCE:
        logger.warning("%s: ~%.0fs is close to the %.0fs floor; if the voice comes out fast this "
                       "will be rejected", path.name, est, MIN_DURATION)

    roles = [b["role"] for b in cleaned]
    if "hook" not in roles:
        raise BriefError(f"{path.name}: no 'hook' beat - the first line must land the situation")
    if "punch" not in roles:
        raise BriefError(f"{path.name}: no 'punch' beat")
    if roles[-1] not in ("punch", "tag"):
        raise BriefError(f"{path.name}: last beat is {roles[-1]!r}; it must be 'punch' or 'tag'")

    # ── direction ───────────────────────────────────────────────────────────
    defaults = {"hook": 0, "setup": 130, "escalate": 110, "turn": 280, "punch": 540, "tag": 320}
    lines = [
        Line(index=i, role=b["role"], text=b["text"],
             pause_before_ms=0 if i == 0 else defaults.get(b["role"], 120))
        for i, b in enumerate(cleaned)
    ]

    for item in data.get("direction") or []:
        if not isinstance(item, dict) or "index" not in item:
            continue
        try:
            idx = int(item["index"])
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < len(lines):
            raise BriefError(f"{path.name}: direction index {idx} has no matching beat")
        ln = lines[idx]
        d = item.get("direction")
        ln.direction = str(d).strip().strip("[]").lower() or None if d else None
        if "pause_before_ms" in item:
            try:
                ln.pause_before_ms = max(0, min(int(item["pause_before_ms"]), MAX_PAUSE_MS))
            except (TypeError, ValueError):
                pass
        emph = item.get("emphasis") or []
        if isinstance(emph, list):
            words = {w.strip(".,!?;:").upper() for w in ln.text.split()}
            missing = [str(e).upper() for e in emph if str(e).strip().upper() not in words]
            if missing:
                raise BriefError(f"{path.name}: line {idx} emphasis {missing} not present in the "
                                 f"line text - captions highlight by exact word match")
            ln.emphasis = [str(e).strip().upper() for e in emph][:3]
    lines[0].pause_before_ms = 0        # never make the viewer wait at the start

    # ── shots ───────────────────────────────────────────────────────────────
    raw_shots = _require(data, "shots", path.name)
    if not isinstance(raw_shots, list) or len(raw_shots) < 4:
        raise BriefError(f"{path.name}: need at least 4 shots, "
                         f"got {len(raw_shots) if isinstance(raw_shots, list) else 0}")

    shots: list[dict] = []
    for i, s in enumerate(raw_shots):
        if not isinstance(s, dict):
            raise BriefError(f"{path.name}: shot {i} is not an object")
        try:
            li = int(_require(s, "line_index", f"shot {i}"))
        except (TypeError, ValueError) as exc:
            raise BriefError(f"{path.name}: shot {i} has a non-numeric line_index") from exc
        if not 0 <= li < len(lines):
            raise BriefError(f"{path.name}: shot {i} points at line {li}, "
                             f"but there are only {len(lines)} lines")
        motion = str(s.get("motion", "")).strip().lower()
        if motion and motion not in VALID_MOTION:
            raise BriefError(f"{path.name}: shot {i} has unknown motion {motion!r} "
                             f"(have: {', '.join(VALID_MOTION)})")
        size = str(s.get("shot_size", "medium")).strip().lower()
        if size not in VALID_SIZE:
            size = "medium"
        shots.append({
            "line_index": li,
            "shot_size": size,
            "prompt": str(_require(s, "prompt", f"shot {i}")).strip(),
            "motion": motion or "push-in",
            "why_this_image": str(s.get("why_this_image", "")).strip(),
        })

    # A shot may span several lines: an uncovered line holds the previous image (see
    # visuals.assign_timing). Fewer shots means fewer image generations, which matters a lot on a
    # rate-limited free tier. The hook and the punch always need their own frame though — the
    # first shot is the scroll-stopper, and revealing the punch's image early kills the joke.
    covered = {s["line_index"] for s in shots}
    if lines[0].index not in covered:
        raise BriefError(f"{path.name}: the hook (line 0) has no shot - it is the first frame "
                         f"and decides whether anyone watches")
    punch = max((l.index for l in lines if l.role == "punch"), default=None)
    if punch is not None and punch not in covered:
        raise BriefError(f"{path.name}: the punch (line {punch}) has no shot of its own - "
                         f"the visual gag must land on the punch, not before it")

    # No two adjacent shots may share a camera move, or it reads as a slideshow.
    shots.sort(key=lambda s: s["line_index"])
    for a, b in zip(shots, shots[1:]):
        if a["motion"] == b["motion"]:
            alt = [m for m in VALID_MOTION if m != a["motion"]]
            b["motion"] = alt[shots.index(b) % len(alt)]

    character = str(data.get("character_sheet", "")).strip()
    if character:
        for s in shots:
            low = s["prompt"].lower()
            if any(w in low for w in ("man", "woman", "person", "he ", "she ", "they ")):
                s["prompt"] = f"{s['prompt']} The person is {character}"

    meta = data.get("metadata") or {}
    if not isinstance(meta, dict):
        raise BriefError(f"{path.name}: 'metadata' must be an object")

    brief = Brief(path=path, premise=premise, style=style, character_sheet=character,
                  lines=lines, shots=shots, metadata=meta)
    logger.info("brief %s: %d beats (%d words), %d shots, style=%s",
                path.name, len(lines), brief.word_count, len(shots), style)
    return brief


def load_all(directory: Path) -> list[Brief]:
    """Load every brief in a directory, skipping (loudly) any that fail validation."""
    out: list[Brief] = []
    for path in sorted(directory.glob("*.json")):
        try:
            out.append(load(path))
        except BriefError as exc:
            logger.error("skipping %s: %s", path.name, exc)
    return out
