# Fleet rulebook for this repo — read first, every unit

You are one unit of a cloud fleet fixing Saharsh's AI comedy Shorts pipeline.
Ten sibling sessions work in this repo at the same time, each on its own unit
and branch. There is no user; never ask. Your routine prompt gave you `ROLE`,
`UNIT`, `BRANCH`, `UNIT_ID`, `MODEL_TAG`, `EXTRA`. After this file read
`docs/fleet/<UNIT>.md` and do exactly that one unit, then stop.

## 1. The repo

- Root = the v2 pipeline (`run.py`, `batch.py`, `shorts/`, `prompts/`,
  `briefs/`, `tests/`, `docs/`). `CLAUDE.md` is the constitution: its
  "Things that will bite you" and "Calibrated constants" sections are
  binding — a change to any of them needs evidence, recorded in
  `docs/ARCHITECTURE.md` or the results file, never taste.
- `ed/` = the OLD pipeline (v1), kept as reference. Never edit it, never
  import from it, never run it.
- Secrets: none exist in this sandbox and none may be written into the tree.
  `GEMINI_API_KEY`, `GROQ_API_KEY` and the YouTube OAuth JSON live only as
  GitHub Actions secrets; the `ci` unit exercises them through
  `workflow_dispatch`. Grep your diff for `AIza`, `gsk_`, `refresh_token`
  values and `client_secret` before every commit.
- Nothing you run may upload. `--dry-run` always; `--publish` / `--privacy`
  never. Uploads happen only from the gated GitHub workflow.

## 2. What can run here

`python -m venv .venv && .venv/bin/pip install -r requirements.txt` then:

| command | needs | what it proves |
|---|---|---|
| `.venv/bin/python tests/test_units.py` | nothing | pure logic (10 tests today) |
| `.venv/bin/python tests/run_offline.py` | ffmpeg via imageio-ffmpeg | whole pipeline with a stubbed LLM |
| `.venv/bin/python run.py --brief briefs/fridge_baseline.json --dry-run --seed 7` | Full network (Pollinations images keyless, edge-tts voice keyless), no LLM key | a REAL render end to end |
| `.venv/bin/python run.py --doctor` | keys for a full report | shows exactly which providers are missing here |
| `tests/validate_prompts.py`, `run.py` without `--brief` | `GEMINI_API_KEY` | **cannot run here** — say so, do not fake it |

A `403` from the proxy (no `x-deny-reason` header is returned) means the
sandbox network policy, not the product. Probe 2026-09-06: pypi, npm and
`generativelanguage.googleapis.com` are reachable; `image.pollinations.ai`,
`speech.platform.bing.com` and `api.groq.com` are 403 until Saharsh sets the
environment to Full network. So the keyless real render may not be possible
here yet: run everything that is offline, record the 403 verbatim, and end
with `BLOCKED-NETWORK` only if your unit's deliverable itself needs that host.
The controller sends `/effort max` to your session after it starts.

## 3. Branch protocol and file ownership

```sh
git fetch origin
git checkout -B "$BRANCH" "origin/$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" origin/main
```

Commit after every green step, scoped `git add <paths>` (never `-A`), push
`git push -u origin "$BRANCH"` (`git pull --rebase origin "$BRANCH"` on
rejection). Never push to `main`. Messages: `fleet(<unit>): <what>`.

Each unit OWNS a disjoint set of files (listed in its role file). Two units
never edit the same file, so the controller can merge every branch without
conflicts. If your fix genuinely needs a file another unit owns, do not
edit it: describe the exact change in your results file under
`## Handoff to <unit>` and let the `ci` lead apply it during integration.
New test files are always yours to add if they carry your unit's prefix
(`tests/test_<unit>_*.py`).

## 4. Standards

- **A claim requires a run** you did in this session; paste the trimmed
  real output. Test COUNTS, not exit codes.
- **A test that cannot fail is not a test.** For every test you add or
  change: apply a mutant to the code it pins, show the test fail, restore,
  show it pass. Record a mutation table in your results file.
- **Never score humour on a scale, never give a judge the persona, the gate
  fails closed, model ids only in `shorts/config.py`, no `zoompan`, no PIL
  captions, serial image generation.** These are the rules the v2 rebuild
  exists for; a change that weakens one is a `FAILED` unit.
- One unit. Finish early → better evidence, not more scope.

## 5. How every unit ends

Write `docs/fleet/results/<UNIT_ID>.md` (what you did, commands + output,
what you did not do and why, commits, handoffs), commit and push it, and end
your final message with exactly one line:

```
FLEET-RESULT: <STATUS> unit=<UNIT_ID> slug=ed branch=<BRANCH> commits=<N> — <one sentence>
```

`<STATUS>` ∈ `BUILT` | `PARTIAL` | `BLOCKED-NETWORK` | `BLOCKED` | `FAILED`.
