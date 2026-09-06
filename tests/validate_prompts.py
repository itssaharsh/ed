"""Round-trip every prompt through the real LLM and check the shapes the code depends on.

    .venv/bin/python tests/validate_prompts.py

The stub in tests/stub_llm.py always returns perfectly-shaped JSON, so the offline test cannot
catch a prompt that a real model answers in the wrong shape. This can. Run it once after adding
an API key, and again whenever you edit a prompt file.

Costs ~6 LLM calls.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shorts.config import CATEGORIES, Config, logger
from shorts.llm import LLM, LLMError
from shorts.write import (
    Premise, _judge_factory, direct, draft_script, generate_premises,
)

FAILURES: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    cfg = Config()
    if not cfg.has_llm():
        print("no LLM key set - nothing to validate. See docs/SETUP.md.")
        return 4

    llm = LLM(cfg)
    caps = cfg.capabilities()
    print(f"validating prompts against: {caps['llm']}\n")

    # ── 01_ideate ───────────────────────────────────────────────────────────
    print("01_ideate")
    premises: list[Premise] = []
    try:
        t = time.time()
        premises = generate_premises(llm, CATEGORIES[0], ["a man labels his fridge food with dates"])
        report("returns premises", len(premises) >= 4, f"{len(premises)} in {time.time()-t:.0f}s")
        report("every premise has situation and turn",
               all(p.situation and p.turn for p in premises))
        report("mechanisms are populated",
               sum(1 for p in premises if p.mechanism) >= len(premises) // 2,
               f"{sum(1 for p in premises if p.mechanism)}/{len(premises)}")
        for p in premises[:3]:
            print(f"        [{p.mechanism or '?'}] {p.situation[:78]}")
    except LLMError as exc:
        report("01_ideate round-trip", False, str(exc)[:160])

    # ── 02_tournament ───────────────────────────────────────────────────────
    print("\n02_tournament")
    try:
        judge = _judge_factory(llm, "comedy premises", "Judging which makes the better bit.")
        v = judge("A man alphabetises his spice rack by scientific name.",
                  "A woman narrates her own cooking to an empty kitchen.")
        report("verdict has a valid winner", str(v.get("winner", "")).upper() in ("A", "B"),
               json.dumps(v)[:120])
        report("verdict names a criterion", bool(v.get("deciding_criterion")))
    except LLMError as exc:
        report("02_tournament round-trip", False, str(exc)[:160])

    # ── 03_script ───────────────────────────────────────────────────────────
    print("\n03_script")
    beats: list[dict] = []
    try:
        seed = premises[0] if premises else Premise(
            "A man alphabetises his spice rack", "He relabels them by scientific name",
            "latin labels", "bad system", "friend")
        beats, joke = draft_script(llm, seed)
        roles = [b["role"] for b in beats]
        words = sum(len(b["text"].split()) for b in beats)
        report("returns beats", len(beats) >= 4, f"{len(beats)} beats, {words} words")
        report("has a hook", "hook" in roles)
        report("has a punch", "punch" in roles)
        report("punch is at the end", roles[-1] in ("punch", "tag"), f"roles={roles}")
        report("length is usable", 50 <= words <= 160, f"{words} words")
        report("explains its own mechanism", bool(joke), joke[:80])
    except LLMError as exc:
        report("03_script round-trip", False, str(exc)[:160])

    # ── 05_delivery ─────────────────────────────────────────────────────────
    print("\n05_delivery")
    try:
        src = beats or [{"role": "hook", "text": "A man alphabetises his spice rack."},
                        {"role": "punch", "text": "By scientific name."}]
        lines = direct(llm, src, "austin")
        report("one line per beat", len(lines) == len(src), f"{len(lines)} vs {len(src)}")
        report("text is never rewritten",
               all(l.text == b["text"] for l, b in zip(lines, src)))
        report("hook does not wait", lines[0].pause_before_ms == 0)
        report("at most 3 directions",
               sum(1 for l in lines if l.direction) <= 3,
               f"{sum(1 for l in lines if l.direction)} directed")
        report("punch gets the longest pause",
               max(lines, key=lambda l: l.pause_before_ms).role in ("punch", "tag"),
               str([l.pause_before_ms for l in lines]))
        report("every line fits Orpheus's 200-char cap",
               all(len(l.text) <= 200 for l in lines))
    except LLMError as exc:
        report("05_delivery round-trip", False, str(exc)[:160])

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall prompt round-trips valid")
    if FAILURES:
        print("  " + "\n  ".join(FAILURES))
    print(f"used {llm.calls} LLM calls")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
