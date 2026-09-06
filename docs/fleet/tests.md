# Unit: tests — a real test suite with a real entry point

Owns: `tests/test_units.py`, `tests/stub_llm.py`, `tests/run_offline.py`,
`tests/validate_prompts.py`, new `tests/conftest.py`, new `tests/test_*.py`
that are NOT prefixed by another unit (`stages`, `providers`, `render`,
`selftest`), `Makefile`, `pyproject.toml`, `requirements-dev.txt`.

1. Convert `tests/test_units.py` (10 hand-rolled tests, `main()` runner) into
   pytest: `pytest` discovers everything under `tests/`, the old
   `python tests/test_units.py` entry keeps working (thin shim), `pyproject.toml`
   carries the pytest config, `requirements-dev.txt` pins pytest.
2. Pin the CLAUDE.md "Calibrated constants" with tests that use the recorded
   evidence as independent reference values: `SIMILARITY_THRESHOLD = 0.32`
   (a reworded duplicate ≈0.39–0.49 must be caught, unrelated <0.02 must
   pass), `MIN_DIRECTIONALITY = 0.18` (an abstract gradient at 0.063 fails,
   a real shot at 0.535 passes — build synthetic images), `MAX_DIRECTIONS = 3`.
   Each test must fail if the constant drifts.
3. Pin the "Things that will bite you": judge prompts get `VOICE=""`
   (assert the rendered judge prompt has no persona), any judging function
   is pairwise (no scale), model ids appear only in `shorts/config.py`
   (a grep test with a positive control), `zoompan` and `drawtext` absent
   from `render.py`, image requests serial (the 16 s gap).
4. `tests/stub_llm.py`: a missing branch must raise loudly (it does today —
   keep it); add a test that every `prompts/*.md` schema marker has a stub
   branch.
5. `Makefile`: `make test` (pytest, no keys, no network), `make offline`
   (`run_offline.py`), `make selftest` (calls the `selftest` unit's entry
   if present — `python -m shorts.selftest`), `make doctor`.
6. Mutation-check every test you write; mutation table in the results file.
   Run `pytest -q` and paste the count.

End with `FLEET-RESULT`.
