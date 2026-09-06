"""Unit tests for the parts where a silent regression would be expensive.

Run:  .venv/bin/python tests/test_units.py
"""
from __future__ import annotations

import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shorts import prompts
from shorts.llm import LLMError, extract_json
from shorts.qc import mechanical_checks
from shorts.store import Entry, Store, similarity
from shorts.tournament import bradley_terry, run_tournament
from shorts.voice import _split_for_limit, estimate_word_times
from shorts.write import Line, _clean_spoken, _default_direction

FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        FAILS.append(label)


def test_json_extraction() -> None:
    print("\njson extraction")
    check("clean", extract_json('{"a":1}') == {"a": 1})
    check("fenced", extract_json('```json\n{"a":2}\n```') == {"a": 2})
    check("prose-wrapped", extract_json('Here:\n{"a":3}\nthanks') == {"a": 3})
    check("array", extract_json("[1,2,3]") == [1, 2, 3])
    for bad in ("", "no json at all"):
        try:
            extract_json(bad)
            check(f"raises on {bad!r}", False)
        except LLMError:
            check(f"raises on {bad!r}", True)


def test_prompts() -> None:
    print("\nprompt rendering")
    cases = {
        "01_ideate": dict(CATEGORY="x", CATEGORY_BRIEF="y", RECENT_PREMISES="- a", N=12),
        "02_tournament": dict(ITEM_KIND="k", CONTEXT="c", A="a", B="b", VOICE=""),
        "03_script": dict(PREMISE="p", TARGET_SECONDS=40, TARGET_WORDS=105),
        "04_punchup": dict(SCRIPT="s", THE_JOKE="j", N=6),
        "05_delivery": dict(BEATS="b", VOICE_NAME="austin"),
        "06_shotlist": dict(LINES="l", STYLE_NAME="flat_absurd", STYLE_CONTRACT="c",
                            TOTAL_SECONDS=40, VOICE=""),
        "08_qc": dict(SCRIPT="s", BASELINE="b", DURATION=40, SHOT_COUNT=8, VOICE=""),
        "09_metadata": dict(SCRIPT="s", PREMISE="p", CATEGORY_TAGS="#t", VOICE=""),
    }
    for name, kw in cases.items():
        out = prompts.render(name, **kw)
        check(f"{name} renders with no leftover variables", "{{" not in out)
    try:
        prompts.render("03_script", PREMISE="p")
        check("missing variable raises", False)
    except prompts.PromptError:
        check("missing variable raises", True)

    for style in prompts.available_styles():
        contract, negative = prompts.style_contract(style)
        check(f"style {style} has a contract", len(contract) > 60)
        check(f"style {style} inherits the universal negative", "jpeg artifacts" in negative)

    # Judges must never be given the writing persona - it makes them prefer their own voice.
    judge = prompts.render("02_tournament", ITEM_KIND="k", CONTEXT="c", A="a", B="b", VOICE="")
    check("judge prompt carries no persona", "You write for a single narrator" not in judge)


def test_tournament() -> None:
    print("\ntournament")
    check("no bouts is neutral", bradley_terry(3, []) == [0.0, 0.0, 0.0])

    def trial(seed: int, noise: float) -> tuple[bool, float]:
        rng = random.Random(seed)
        truth = rng.sample(range(1, 21), 8)
        cands = [f"c{v}" for v in truth]

        def judge(a: str, b: str) -> dict:
            va, vb = int(a[1:]), int(b[1:])
            better = "A" if va > vb else "B"
            worse = "B" if better == "A" else "A"
            return {"winner": better if rng.random() > noise else worse,
                    "deciding_criterion": "x", "why": "y", "confidence": "high"}

        w, _, _ = run_tournament(cands, judge, rounds=3, rng=rng)
        picked = int(cands[w][1:])
        return picked == max(truth), (max(truth) - picked) / max(truth)

    for noise, min_top1, max_regret in ((0.0, 0.95, 0.02), (0.15, 0.45, 0.16)):
        res = [trial(s, noise) for s in range(120)]
        top1 = sum(r[0] for r in res) / len(res)
        regret = statistics.mean(r[1] for r in res)
        check(f"judge noise {noise:.0%}: picks best {top1:.0%} (>={min_top1:.0%})", top1 >= min_top1,
              f"got {top1:.2f}")
        check(f"judge noise {noise:.0%}: regret {regret:.1%} (<={max_regret:.0%})",
              regret <= max_regret, f"got {regret:.3f}")


def test_dedup() -> None:
    print("\npremise dedup")
    dupes = [
        ("my friend tried to microwave a salad and set off the fire alarm",
         "my buddy microwaved a salad and the fire alarm went off"),
        ("a man has a spreadsheet for his coffee budget and a 400 dollar crossbow",
         "a guy tracks every coffee in a spreadsheet but owns an expensive crossbow"),
    ]
    unrelated = [
        ("my friend tried to microwave a salad and set off the fire alarm",
         "a woman refuses to use the office lift and takes the stairs to floor 11"),
        ("a man keeps a spreadsheet ranking his dating matches",
         "someone brings a full mechanical keyboard to a coffee shop"),
    ]
    from shorts.store import SIMILARITY_THRESHOLD as T
    for a, b in dupes:
        s = similarity(a, b)
        check(f"reworded duplicate caught ({s:.2f} >= {T})", s >= T)
    for a, b in unrelated:
        s = similarity(a, b)
        check(f"unrelated premise kept ({s:.2f} < {T})", s < T)

    store = Store(Path(tempfile.mkdtemp()) / "s.jsonl")
    store.append(Entry("r1", time.time(), "c", dupes[0][0], "script text", "t"))
    dup, score, _ = store.is_duplicate(dupes[0][1])
    check("store flags the duplicate", dup, f"score {score:.2f}")
    check("store keeps the unrelated one", not store.is_duplicate(unrelated[0][1])[0])


def test_voice_timing() -> None:
    print("\nvoice timing")
    long_line = ("This is a sentence that goes on. And here is another one that also goes on "
                 "for a while. And a third to push it past the limit for sure.")
    parts = _split_for_limit(long_line, 80)
    check("splits respect the character limit", all(len(p) <= 80 for p in parts))
    check("split loses no words",
          sorted(" ".join(parts).split()) == sorted(long_line.split()))

    lines = [Line(0, "hook", "He has a SPREADSHEET for coffee", duration=2.0, start=0.0,
                  emphasis=["SPREADSHEET"])]
    words = estimate_word_times(lines)
    check("one timing per word", len(words) == len(lines[0].text.split()))
    check("timings stay inside the line", words[-1][1] <= 2.0001)
    check("timings are monotonic", all(words[i][1] <= words[i + 1][0] + 1e-6
                                       for i in range(len(words) - 1)))
    check("emphasis word is flagged", any(w[3] for w in words))


def test_script_cleaning() -> None:
    print("\nscript cleaning")
    out = _clean_spoken("He said [PAUSE] *loudly* - it was ... fine (obviously)")
    for artefact in ("[", "]", "*", "(", ")"):
        check(f"strips {artefact!r} so TTS cannot read it aloud", artefact not in out)

    lines = _default_direction([{"role": r, "text": "x"} for r in
                                ("hook", "setup", "escalate", "turn", "punch")])
    check("hook never waits", lines[0].pause_before_ms == 0)
    check("punch gets the longest pause",
          lines[-1].pause_before_ms == max(l.pause_before_ms for l in lines))


def test_sparse_shot_timing() -> None:
    """A shot may cover several lines. The uncovered ones must extend the previous shot, not
    vanish - dropping their time would desynchronise the video from the audio."""
    print("\nsparse shot timing")
    from shorts.visuals import assign_timing

    lines = []
    cursor = 0.0
    for i in range(7):
        pause = 0.0 if i == 0 else 0.2
        cursor += pause
        ln = Line(i, "punch" if i == 6 else "escalate", "word " * 10, pause_before_ms=int(pause * 1000))
        ln.start, ln.duration = cursor, 2.4
        cursor += ln.duration
        lines.append(ln)
    audio = cursor

    for shot_lines in ([0, 2, 4, 6], [0, 6], [0, 1, 2, 3, 4, 5, 6]):
        shots = [{"line_index": i, "prompt": "p", "motion": "push-in"} for i in shot_lines]
        timed = assign_timing(shots, lines, audio)
        covered = sum(s["duration"] for s in timed)
        starts = [s["start"] for s in timed]
        label = f"{len(shot_lines)} shots over 7 lines"
        check(f"{label}: covers the full audio", abs(covered - audio) < 0.05,
              f"covered {covered:.2f} vs {audio:.2f}")
        check(f"{label}: starts at zero", abs(starts[0]) < 0.01)
        check(f"{label}: shots are in order", starts == sorted(starts))
        check(f"{label}: no zero-length shot", all(s["duration"] > 0.3 for s in timed))


def test_briefs_valid() -> None:
    """Every shipped brief must load. A broken one is a broken example."""
    print("\nshipped briefs")
    from shorts.brief import BriefError, load

    briefs = sorted((Path(__file__).resolve().parent.parent / "briefs").glob("*.json"))
    check("briefs exist", bool(briefs), f"{len(briefs)} found")
    for b in briefs:
        try:
            loaded = load(b)
            ok = 60 <= loaded.word_count <= 140 and len(loaded.shots) >= 4
            check(f"{b.stem} loads and is sane", ok,
                  f"{loaded.word_count}w {len(loaded.shots)}sh")
        except BriefError as exc:
            check(f"{b.stem} loads", False, str(exc)[:110])


def test_image_breaker() -> None:
    print("\nimage circuit breaker")
    import tempfile
    import shorts.images as im
    from shorts.config import Config

    orig = im.generate_one
    im.generate_one = lambda *a, **k: (_ for _ in ()).throw(im.ImageError("provider down"))
    try:
        shots = [{"prompt": f"shot {i}", "motion": "push-in"} for i in range(8)]
        res = im.generate_all(Config(), shots, "style", "neg", Path(tempfile.mkdtemp()), 1)
    finally:
        im.generate_one = orig

    attempted = sum(1 for r in res if not r.get("skipped"))
    check("gives up on a dead provider", attempted <= 5, f"attempted {attempted}/8")
    check("still returns a row per shot", len(res) == 8)
    check("no shot is falsely marked ok", all(not r["ok"] for r in res))


def test_gate() -> None:
    print("\nquality gate")
    store = Store(Path(tempfile.mkdtemp()) / "s.jsonl")
    lines = [Line(i, r, "word " * 12, duration=4.0, start=i * 4.0)
             for i, r in enumerate(["hook", "setup", "escalate", "turn", "punch"])]
    shots = [{"ok": True, "image": f"/tmp/s{i}.png"} for i in range(6)]
    info = {"duration": 20.0, "width": 1080, "height": 1920, "has_audio": True, "has_video": True}
    base = dict(lines=lines, shots=shots, video_info=info, audio_duration=20.0, lufs=-14.0,
                script="word " * 100, store=store, premise="a unique premise about lifts")

    def fails(**over):
        kw = {**base, **over}
        return mechanical_checks(**kw)[0]

    check("healthy video passes", not fails())
    check("rejects too short", fails(audio_duration=8.0, video_info={**info, "duration": 8.0}))
    check("rejects too long", fails(audio_duration=75.0, video_info={**info, "duration": 75.0}))
    check("rejects av desync", fails(video_info={**info, "duration": 26.0}))
    check("rejects landscape", fails(video_info={**info, "width": 1920, "height": 1080}))
    check("rejects missing audio", fails(video_info={**info, "has_audio": False}))
    check("rejects too few shots", fails(shots=[{"ok": True, "image": "/tmp/a.png"}] * 2))
    check("rejects one repeated image",
          fails(shots=[{"ok": True, "image": "/tmp/same.png"} for _ in range(6)]))
    check("rejects a stub script", fails(script="tiny script"))
    check("rejects an over-long pause",
          fails(lines=[Line(0, "hook", "a b c", duration=4.0, pause_before_ms=2000)] + lines[1:]))


def main() -> int:
    for fn in (test_json_extraction, test_prompts, test_tournament, test_dedup,
               test_voice_timing, test_script_cleaning, test_sparse_shot_timing,
               test_briefs_valid, test_image_breaker, test_gate):
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
