# Unit: selftest — the pipeline tests itself, end to end, with no keys

Owns: new `shorts/selftest.py`, new `selftest.py` (root entry), new
`tests/test_selftest_*.py`, new `docs/SELFTEST.md`. Do NOT edit `run.py`
(the `stages` unit owns it) — hand off a one-line `--selftest` flag wiring
in your results file for the `ci` lead.

Saharsh's instruction was "tell it to test itself". Build exactly that:

1. `python -m shorts.selftest` (and `python selftest.py`) renders a brief
   (`briefs/fridge_baseline.json`) through the REAL `run._finish` path with:
   a stub image provider (deterministic generated PNGs — solid subjects with
   real directionality so the structural validator passes), a stub voice
   (a synthesised tone track with word timings, no network), the real
   ffmpeg/libass render, the real mechanical QC, and a stub judge that
   returns "pass". No network, no keys.
2. Then it VERIFIES the output itself: `ffprobe` the mp4 (1080×1920, duration
   within the brief's expected range, video+audio streams, loudness in
   range), captions file parses and its cue count equals the word count,
   shot count matches the brief, every stage checkpoint JSON exists and
   validates, and the run took less than a stated ceiling. Prints a table
   and exits non-zero on any failure.
3. Add a second mode `--negative` that deliberately breaks one invariant
   (e.g. drops a shot image) and asserts the self-test FAILS — the self-test
   must be able to fail (mutation-check applied to itself).
4. `tests/test_selftest_*.py` runs both modes under pytest (mark `slow`).
5. `docs/SELFTEST.md`: what it proves, what it cannot prove (prompt
   regressions — that needs `validate_prompts.py` with a key), runtime.
6. Run it here; paste the table. If the ffmpeg download is blocked →
   `BLOCKED-NETWORK`.

End with `FLEET-RESULT`.
