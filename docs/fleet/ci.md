# Unit: ci — GitHub Actions, integration lead, the real self-test with real keys

Owns: `.github/workflows/*.yml`. As the LAST unit of the squad (the controller
launches you after every other pipeline unit and the three review lenses are
merged into `main`) you are also the integrator: you may edit any file to
make the merged whole coherent, and you apply every `## Handoff to ci` and
`## Handoff to <unit>` item the other units' results files recorded.

1. `git log --oneline origin/main -40` and read every `docs/fleet/results/*.md`
   and `docs/fleet/reviews/*.md`. Apply the handoffs. Resolve the reviews'
   findings that no unit fixed. Run `make test`, `make offline`,
   `make selftest`, `run.py --doctor`, the brief dry run; all must be green
   here before you touch CI. Paste counts.
2. `.github/workflows/ci.yml`: on `push` and `pull_request`: Python 3.12,
   `pip install -r requirements.txt -r requirements-dev.txt`, `make test`,
   `make offline`, `make selftest`, upload the self-test mp4 + table as an
   artifact. No secrets, no network beyond pip and the ffmpeg binary.
3. `.github/workflows/shorts.yml` (exists — review it line by line): the
   6/day schedule, `workflow_dispatch` with `dry_run` and `privacy` inputs,
   `--doctor` as a preflight step that fails the job if the LLM key is
   missing, the generate step, the exit-2 gate semantics, artefact upload,
   token refresh persistence, premise-store commit. Uploads must remain
   opt-in (`private` default). Keep the concurrency group.
4. The old v1 workflow is disabled and lives under `ed/.github/` (inactive —
   GitHub only reads the root `.github/`). Leave it there.
5. **The real self-test.** The scheduled workflow is currently DISABLED by
   the controller. Push your branch, then `gh workflow run ci.yml --ref
   <branch>` and `gh workflow run shorts.yml --ref <branch> -f dry_run=true`
   (the repo's secrets give this run the real Gemini/Groq keys; it uploads
   nothing). Wait for both (`gh run watch`), read the logs (`gh run view
   --log`), fix what fails, repeat. Paste the final run URLs and
   conclusions. Then, and only if both are green: `gh workflow enable
   shorts.yml`. If `gh` cannot reach the Actions API from the sandbox,
   leave it disabled, say so, and end `BLOCKED-NETWORK`.
6. Results file: integration changes, the run URLs, what is enabled.

End with `FLEET-RESULT`.
