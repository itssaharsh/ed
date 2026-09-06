# Unit: review-tests — adversarial review lens: tests and the self-test

Owns: `docs/fleet/reviews/tests.md` only. You change no code.

Review the CURRENT `main` (the pre-fleet code) as a sceptic on the opposite
model to the `tests` and `selftest` workers. Goal: which promises in
`CLAUDE.md` have NO test; which existing tests in `tests/test_units.py`
cannot fail (apply mutants to the code they claim to pin and show the suite
stays green — throwaway edits, restored after, never committed); whether
`tests/run_offline.py` proves anything about render/captions/qc or only that
the stub returns well-shaped JSON; what a self-test must assert about the
produced mp4 to be worth trusting. For every finding: file:line, the promise
it leaves unpinned, the mutant that survives, and the test that would kill
it. Rank by user harm. Refute your own weakest findings first. Then read
`docs/fleet/results/` for the `tests` and `selftest` units if they exist and
mark each finding `fixed there` / `not fixed` / `fixed wrongly`.

End with `FLEET-RESULT: BUILT … — findings=N surviving_mutants=M`.
