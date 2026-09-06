# Unit: docs — every sentence true against the code

Owns: `README.md`, `docs/ARCHITECTURE.md`, `briefs/README.md`,
`prompts/README.md`, `CLAUDE.md`.

1. Read each owned doc and check every factual claim against the code
   (numbers, file names, commands, provider names, free-tier limits with
   their dates, the layout table, the testing section). A claim you cannot
   verify by reading code or running a command becomes "as of <date>,
   unverified" — never silently kept.
2. `README.md`: the quickstart must work on a fresh clone as written (run
   it here: venv, `pip install -r requirements.txt`, `--doctor`, the brief
   dry run). Add the fleet's new entry points only if they exist on `main`
   at the time you write (the `ci` lead will reconcile with the other units
   — leave a `## Handoff to ci` list of doc lines that depend on their work).
3. `CLAUDE.md`: keep the binding sections; add a short "Fleet" paragraph
   pointing at `docs/fleet/README.md`; fix the Testing table to match the
   real commands.
4. `docs/ARCHITECTURE.md`: the "what was wrong with v1" section must cite
   `ed/` files by path; check each citation exists.
5. Do not touch any `.py` file. Results file lists every claim you changed
   with the evidence that justified it.

End with `FLEET-RESULT`.
