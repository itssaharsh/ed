# Unit: review-stages — adversarial review lens: stages, briefs, batch

Owns: `docs/fleet/reviews/stages.md` only. You change no code.

Review the CURRENT `main` (the pre-fleet code) as a sceptic on the opposite
model to the `stages` worker. Goal: find what breaks the product's promises
(see `CLAUDE.md`) in `run.py`, `batch.py`, `shorts/{write,tournament,visuals,
store,brief,prompts}.py`, `prompts/*.md`, `briefs/*.json`. For every finding:
file:line, the promise it breaks, a concrete input that triggers it, and how
you verified (run `tests/run_offline.py` and craft inputs; write throwaway
tests under `/tmp`, not in the repo). Rank by user harm. Refute your own
weakest findings before writing them down. Then read `docs/fleet/results/`
for the `stages` unit if it already exists and mark each of your findings
`fixed there` / `not fixed` / `fixed wrongly`.

End with `FLEET-RESULT: BUILT … — findings=N confirmed_by_run=M`.
