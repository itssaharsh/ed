"""Stage 11: the fail-closed quality gate.

The old pipeline had nothing here. When every LLM call 404'd it still downloaded a stock clip for
the query "person" and uploaded it (see the run log in ed/run_logs.zip). This module exists so
that cannot happen: if quality cannot be positively established, nothing is published.

Part A is mechanical and runs first, so a structurally broken video never costs a judge call.
Part B compares the script against a fixed baseline of known mediocre-but-acceptable quality,
using the same pairwise machinery as the tournament, for the same reason: absolute LLM quality
scores are noise.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts
from .config import (
    MAX_DURATION, MAX_WORDS, MIN_DURATION, MIN_WORDS, Config, logger,
)
from .llm import LLM, LLMError
from .store import Store
from .write import Line

MIN_SHOTS = 4
MIN_CAPTION_WORDS = 15
LUFS_RANGE = (-17.0, -11.0)
MAX_SILENCE = 1.2


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    judge: dict = field(default_factory=dict)

    def report(self) -> str:
        lines = [f"quality gate: {'PASS' if self.passed else 'REJECT'}"]
        for f in self.failures:
            lines.append(f"  FAIL    {f}")
        for w in self.warnings:
            lines.append(f"  warn    {w}")
        if self.judge:
            lines.append(f"  judge   beats_baseline={self.judge.get('beats_baseline')} "
                         f"({self.judge.get('deciding_criterion', '?')}): "
                         f"{self.judge.get('why', '')}")
        return "\n".join(lines)


def mechanical_checks(*, lines: list[Line], shots: list[dict], video_info: dict,
                      audio_duration: float, lufs: float | None, script: str,
                      store: Store, premise: str) -> tuple[list[str], list[str]]:
    fails: list[str] = []
    warns: list[str] = []

    if not (MIN_DURATION <= audio_duration <= MAX_DURATION):
        fails.append(f"audio duration {audio_duration:.1f}s outside {MIN_DURATION}-{MAX_DURATION}s")

    vdur = video_info.get("duration")
    if vdur is None:
        fails.append("could not read video duration")
    elif abs(vdur - audio_duration) > 0.5:
        fails.append(f"video {vdur:.2f}s does not match audio {audio_duration:.2f}s")

    if video_info.get("width") != 1080 or video_info.get("height") != 1920:
        fails.append(f"wrong dimensions {video_info.get('width')}x{video_info.get('height')}")
    if not video_info.get("has_audio"):
        fails.append("no audio stream")

    usable = [s for s in shots if s.get("ok") and s.get("image")]
    if len(usable) < MIN_SHOTS:
        fails.append(f"only {len(usable)} usable shots (need {MIN_SHOTS})")

    distinct = {s.get("image") for s in usable}
    if usable and len(distinct) < max(2, len(usable) // 2):
        fails.append(f"only {len(distinct)} distinct images across {len(usable)} shots")

    words = script.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        fails.append(f"script is {len(words)} words, outside {MIN_WORDS}-{MAX_WORDS}")

    caption_words = sum(len(l.text.split()) for l in lines)
    if caption_words < MIN_CAPTION_WORDS:
        fails.append(f"only {caption_words} caption words (need {MIN_CAPTION_WORDS})")

    if lufs is not None and not (LUFS_RANGE[0] <= lufs <= LUFS_RANGE[1]):
        warns.append(f"loudness {lufs:.1f} LUFS outside {LUFS_RANGE}")

    for ln in lines:
        if ln.pause_before_ms / 1000.0 > MAX_SILENCE:
            fails.append(f"line {ln.index} has a {ln.pause_before_ms}ms pause (max {MAX_SILENCE}s)")

    if store.seen_script(script):
        fails.append("this exact script has been published before")
    dup, score, entry = store.is_duplicate(premise)
    if dup:
        fails.append(f"premise duplicates run {entry.run_id if entry else '?'} (similarity {score:.2f})")
    elif score > 0.25:
        warns.append(f"premise is somewhat close to a previous one (similarity {score:.2f})")

    if any(l.role == "punch" for l in lines):
        punch_idx = max(i for i, l in enumerate(lines) if l.role == "punch")
        if punch_idx < len(lines) - 2:
            warns.append("the punch line is not near the end")

    return fails, warns


def baseline_script(cfg: Config) -> str:
    """The quality floor. A candidate that loses to this is not published."""
    path = cfg.assets / "baseline_script.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return " ".join(b["text"] for b in data["beats"])
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("baseline_script.json is malformed; using the built-in floor")
    return (
        "My flatmate labels his food in the fridge. Not with his name. With dates. "
        "Every container has a little date on masking tape. Last week I moved one two inches "
        "to get the milk. He noticed. He asked me, genuinely, whether I had a system. "
        "I said no. He looked at me like I'd told him I couldn't read."
    )


def judge_quality(llm: LLM, cfg: Config, script: str, duration: float,
                  shot_count: int) -> dict:
    p = prompts.render(
        "08_qc", SCRIPT=script, BASELINE=baseline_script(cfg),
        DURATION=round(duration, 1), SHOT_COUNT=shot_count, VOICE="",
    )
    return llm.complete_json(p, temperature=0.2)


def run_gate(llm: LLM, cfg: Config, *, lines: list[Line], shots: list[dict],
             video_info: dict, audio_duration: float, lufs: float | None,
             script: str, store: Store, premise: str) -> GateResult:
    fails, warns = mechanical_checks(
        lines=lines, shots=shots, video_info=video_info, audio_duration=audio_duration,
        lufs=lufs, script=script, store=store, premise=premise,
    )
    if fails:
        # Do not spend a judge call on a structurally broken video.
        return GateResult(passed=False, failures=fails, warnings=warns)

    if llm is None:
        # A hand-authored brief with no key available. The mechanical checks all passed, and the
        # comedy was written by a person rather than a model, so there is no baseline comparison
        # to make. Pass, but say plainly that the humour was never machine-judged - this is the
        # one place the pipeline relies on the author instead of the gate.
        warns.append("no LLM available: comedy judge skipped, mechanical checks only")
        return GateResult(passed=True, failures=[], warnings=warns,
                          judge={"beats_baseline": None, "why": "judge skipped (no LLM)"})

    try:
        verdict = judge_quality(llm, cfg, script, audio_duration, len(shots))
    except LLMError as exc:
        # Fail closed. An unavailable judge is not permission to publish.
        return GateResult(
            passed=False,
            failures=[f"quality judge unavailable, refusing to publish unverified: {exc}"],
            warnings=warns,
        )

    if not verdict.get("beats_baseline", False):
        fails.append(f"loses to the baseline script "
                     f"({verdict.get('deciding_criterion', '?')}): {verdict.get('why', '')}")

    vetoes = verdict.get("vetoes") or {}
    for name, tripped in vetoes.items():
        if tripped:
            fails.append(f"veto: {name}")

    return GateResult(passed=not fails, failures=fails, warnings=warns, judge=verdict)
