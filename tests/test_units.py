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


def test_personas() -> None:
    """The writers' room must actually contain different writers.

    The point of the craft/persona split is variance in the *shape of the ending*: every script
    before it ended on a flat declarative sentence, which is both the funniness ceiling and what
    YouTube's Inauthentic Content policy penalises. A persona that sounds different but ends the
    same way buys nothing, so punch shapes are asserted distinct here.
    """
    from shorts import prompts as P
    print("\nwriters' room")

    names = P.available_personas()
    check("at least three personas", len(names) >= 3, f"{len(names)}: {names}")
    check("the incumbent voice survives", P.DEFAULT_PERSONA in names)

    shapes, bodies = {}, {}
    for n in names:
        b = P.persona_block(n)
        bodies[n] = b
        line = next((l for l in b.splitlines() if l.startswith("**Punch shape")), "")
        shapes[n] = line
        check(f"{n} declares a punch shape", bool(line))
        check(f"{n} declares a forbidden punch shape", "**Forbidden punch shape" in b)

    check("punch shapes are all distinct", len(set(shapes.values())) == len(names),
          f"{len(set(shapes.values()))} distinct of {len(names)}")
    check("persona bodies are all distinct", len(set(bodies.values())) == len(names))

    for n in names:
        v = P.voice_block(n)
        check(f"{n} VOICE carries the craft block",
              "specific + true + escalating" in v and "Banned" in v)
        check(f"{n} VOICE carries its own persona", shapes[n] in v)

    try:
        P.persona_block("no_such_voice")
        check("unknown persona raises", False)
    except P.PromptError as exc:
        check("unknown persona raises with the list", "available" in str(exc))


def test_caption_cards() -> None:
    """Cards must break where the sentence breaks.

    The old flat every-third-word counter produced real output like "FRIDGE ITEMS. NOT" and
    "DATE. EVERY SINGLE" - the end of one thought glued to the start of the next. On a Short the
    caption is the visual beat, so that steps on the timing the delivery stage designed.
    """
    from shorts.captions import _should_break
    print("\ncaption cards")

    def split(text, emph=()):
        cards, cur = [], []
        for w in text.split():
            item = (0.0, 0.0, w, w.strip(".,!?").upper() in emph)
            cur.append(item)
            if _should_break(cur, item):
                cards.append(" ".join(x[2] for x in cur)); cur = []
        if cur:
            cards.append(" ".join(x[2] for x in cur))
        return cards

    cards = split("There is a man in my office who puts dates on the fridge items. "
                  "Not his name on the tape. The date.")
    straddles = [c for c in cards if any(m in c[:-1] for m in (".", "!", "?"))]
    check("no card straddles a sentence end", not straddles, str(straddles))
    check("a short sentence gets its own card", "The date." in cards, str(cards))
    check("cards never exceed the word cap", all(len(c.split()) <= 3 for c in cards))

    cards = split("He looked at me like I had offered him a DEBT.", emph={"DEBT"})
    check("an emphasised word still breaks the card", cards[-1].split()[-1].startswith("DEBT"))

    # A clause break lands only once the card is already readable (>=2 words), so a comma is
    # never traded for a choppy one-word card. A sentence end always breaks regardless.
    cards = split("Every single container, in the same handwriting.")
    check("a clause break lands when the card is readable",
          "Every single container," in cards, str(cards))
    check("the sentence still closes its own card",
          cards[-1].endswith("handwriting."), str(cards))


def test_coherence() -> None:
    """The style-drift metric must separate the runs that visibly read as one production."""
    from shorts.images import coherence
    from shorts.config import MAX_SATURATION_RANGE
    from pathlib import Path as _P
    print("\nstyle coherence")

    check("empty input is safe", coherence([])["saturation_range"] == 0.0)
    check("a single shot cannot drift", coherence([_P("nope.png")])["saturation_range"] == 0.0)

    runs = {r.name: sorted((r / "images").glob("shot_*.png"))
            for r in _P("work").iterdir() if (r / "images").is_dir()}
    runs = {k: v for k, v in runs.items() if len(v) >= 2}
    if not runs:
        check("work/ has renders to calibrate against", False, "no runs on disk")
        return

    scored = {k: coherence(v)["saturation_range"] for k, v in runs.items()}
    tight = [k for k, v in scored.items() if v <= MAX_SATURATION_RANGE]
    loose = [k for k, v in scored.items() if v > MAX_SATURATION_RANGE]
    check("the metric separates runs", bool(tight) and bool(loose),
          f"{len(tight)} under / {len(loose)} over threshold {MAX_SATURATION_RANGE}")
    check("the known-coherent neon_late run passes",
          scored.get("20260822-232109-8017", 1.0) <= MAX_SATURATION_RANGE,
          f"{scored.get('20260822-232109-8017')}")
    check("the known-incoherent offline-11 run is flagged",
          scored.get("offline-11", 0.0) > MAX_SATURATION_RANGE, f"{scored.get('offline-11')}")


def test_subject_check() -> None:
    """is_usable() cannot tell whether an image shows the requested subject; this can.

    Measured on work/offline-11: 6 of 7 shots did not depict their subject (a corridor for "a man
    in front of an open office fridge", an abstract blob for "masking tape on a container") and
    all 7 passed is_usable.
    """
    import dataclasses
    from shorts.images import subject_of, _parse_verdict, depicts_subject
    from shorts.config import Config
    print("\nsubject verification")

    check("strips the shot size",
          subject_of("wide shot of a man at an open office fridge, flat stare, strip light")
          == "a man at an open office fridge")
    check("handles extreme close-up",
          subject_of("extreme close up of a strip of masking tape on a tub, hard light")
          == "a strip of masking tape on a tub")
    check("survives an unexpected shape",
          subject_of("a jar of mustard on a shelf") == "a jar of mustard on a shelf")

    check("yes parses true", _parse_verdict("Yes")[0] is True)
    check("no parses false", _parse_verdict("no, an empty corridor")[0] is False)
    check("prefixed yes parses true", _parse_verdict("YES - clearly a mustard jar")[0] is True)
    check("a non-answer is unknown, not a rejection", _parse_verdict("I think maybe")[0] is None)
    check("empty is unknown, not a rejection", _parse_verdict("")[0] is None)

    # The keyless path must be untouched: no key means no check, and no check means keep the
    # image. A vision outage must never halt image generation - unlike the QC gate, which fails
    # closed, this one fails open by design.
    cfg = dataclasses.replace(Config(), gemini_key=None)
    verdict, _ = depicts_subject(cfg, b"not-an-image", "wide shot of a fridge")
    check("no key means no check (fails open)", verdict is None)


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
                script="word " * 100, store=store, premise="a unique premise about lifts",
                voice_peak_dbfs=-4.2, final_peak_dbfs=-1.3)

    def fails(**over):
        kw = {**base, **over}
        return mechanical_checks(**kw)[0]

    check("healthy video passes", not fails())

    # Silence rails. These exist because a silent render used to pass every check here and
    # publish: ffmpeg reports "-inf LUFS" for silence, the old measure_loudness regex could not
    # match "inf" and returned None, and the loudness check was guarded by `if lufs is not None`
    # - so it was skipped exactly when it mattered. Measured on this build: a real render peaks
    # at -1.3 dBFS, digital silence at -91.0.
    check("rejects a silent render", fails(final_peak_dbfs=-91.0))
    check("rejects silent narration", fails(voice_peak_dbfs=-120.0))
    check("rejects an unmeasurable peak", fails(final_peak_dbfs=None))
    check("rejects unmeasurable loudness", fails(lufs=None))
    check("rejects audio at the silence floor", fails(final_peak_dbfs=-45.0))
    check("allows quiet but real audio", not fails(final_peak_dbfs=-44.0))
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
    for fn in (test_json_extraction, test_prompts, test_personas,
               test_caption_cards, test_coherence, test_subject_check, test_tournament, test_dedup,
               test_voice_timing, test_script_cleaning, test_sparse_shot_timing,
               test_briefs_valid, test_image_breaker, test_gate):
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
