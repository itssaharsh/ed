# Unit: review-providers — adversarial review lens: providers, publish, doctor, gate

Owns: `docs/fleet/reviews/providers.md` only. You change no code.

Review the CURRENT `main` (the pre-fleet code) as a sceptic on the opposite
model to the `providers` and `render` workers. Goal: find what breaks the
product's promises (see `CLAUDE.md`) on the money paths in
`shorts/{config,llm,voice,images,doctor,publish,captions,render,qc,fonts}.py`:
missing keys, quotas, 429/5xx handling, the 16 s anonymous-image gap, the
fail-closed gate, upload privacy defaults, token refresh, model ids outside
`config.py`, `zoompan`/`drawtext` creeping back. For every finding:
file:line, the promise it breaks, a concrete input that triggers it, and how
you verified (fake HTTP responses in throwaway scripts under `/tmp`, not in
the repo). Rank by user harm (an accidental public upload of a broken video
ranks first). Refute your own weakest findings before writing them down.
Then read `docs/fleet/results/` for the `providers` and `render` units if
they exist and mark each finding `fixed there` / `not fixed` / `fixed wrongly`.

End with `FLEET-RESULT: BUILT … — findings=N confirmed_by_run=M`.
