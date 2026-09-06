# Working on this repo

A $0, fully-automated AI comedy Shorts pipeline. Read `docs/ARCHITECTURE.md` before changing
anything structural, and `docs/RESEARCH.md` before swapping any provider.

`ed/` is the **previous** pipeline, kept as a reference. Do not edit it; do not import from it.

## Things that will bite you

**`zoompan` is unusable.** Measured at 109 s to render 3 s of 1080x1920. It rescales the whole
input every output frame. Camera moves use `scale:eval=frame` + time-varying `crop` instead
(4.3 s per 4 s clip). If you are tempted to "simplify" `shorts/render.py:_move_chain` back to
`zoompan`, don't.

**This ffmpeg build has no `drawtext`.** Captions go through libass (`ass` filter). That is also
the better tool — do not reintroduce PIL-rendered caption PNGs.

**Never score humour on a scale.** Absolute LLM joke ratings collapse (88.5% identical scores in
the research this is built on). Every selection is a pairwise tournament — `shorts/tournament.py`.
Any new judging prompt must be pairwise.

**Judging prompts must not get the comedian persona.** `prompts.render(..., VOICE="")` for judges.
A judge carrying the writing persona prefers its own voice and the tournament becomes noise.

**The gate fails closed.** If the quality judge is unreachable, the run must not publish. This
is deliberate — the old pipeline's defining failure was uploading a video built from a placeholder
after every LLM call 404'd. Do not add a "publish anyway" path.

**An LLM key is mandatory.** There is no working keyless text tier: anonymous Pollinations allows
a handful of requests then returns 401 regardless of pacing, and a video needs ~14 calls. Images
*do* work keyless; text does not.

**Model ids belong in `shorts/config.py`.** The old pipeline hardcoded `gemini-1.5-flash` inline;
when Google retired it, every call 404'd. Keep them in one place so a retirement is a one-line fix.

**Free image tiers rate-limit per IP — render serially.** Three parallel renders produced one
image in seven minutes (the rest in 429 backoff); one paced render produced one every 60–180s.
`shorts/images.py` enforces a process-wide 16s gap between anonymous requests, and `batch.py` runs
briefs one at a time. Do not "optimise" either into parallelism.

**Briefs bypass the LLM entirely.** `--brief briefs/x.json` supplies stages 1–6, so the pipeline
runs with no API key. Both paths converge in `run._finish`, so there is one render pipeline, not
two. A brief may use fewer shots than lines (an uncovered line holds the previous image) but the
hook and the punch always need their own frame.

## Layout

| | |
|---|---|
| `prompts/*.md` | Every LLM instruction. Behaviour changes go here, not in Python. |
| `shorts/write.py` | Stages 1-5: premises → tournament → script → punch-up → direction |
| `shorts/visuals.py` | Stage 6: shot list bound to script beats |
| `shorts/images.py` | Stage 7: image generation + the structural validator |
| `shorts/voice.py` | Stage 8: per-line speech, designed silences, word timings |
| `shorts/captions.py` | Stage 9: ASS karaoke |
| `shorts/render.py` | Stage 10: the ffmpeg graph |
| `shorts/qc.py` | Stage 11: the fail-closed gate |
| `shorts/store.py` | Premise memory + dedup |
| `shorts/brief.py` | Hand-authored briefs: stages 1-6 as JSON, no LLM needed |
| `shorts/doctor.py` | `--doctor` provider preflight |
| `briefs/*.json` | Written comedy. `briefs/README.md` has the schema and the craft rules. |
| `batch.py` | Render every brief, serially |

## Testing

```bash
.venv/bin/python run.py --doctor             # probe every provider, seconds
.venv/bin/python tests/test_units.py         # pure logic, no keys, instant
.venv/bin/python tests/run_offline.py        # whole pipeline, stubbed LLM, no keys
.venv/bin/python tests/validate_prompts.py   # real LLM round-trip, ~6 calls
.venv/bin/python run.py --dry-run --seed 7   # real run, reproducible, no upload
```

**The stub cannot catch prompt regressions.** `tests/stub_llm.py` always returns perfectly-shaped
JSON, so `run_offline.py` passing says nothing about whether a real model answers your prompt in
the right shape. After editing anything in `prompts/`, run `tests/validate_prompts.py`.

`tests/stub_llm.py` dispatches on markers unique to each prompt file's JSON schema block. If you
add a prompt, add a branch — it raises rather than returning `{}`, so a missed branch is loud.

Read `work/<run_id>/` after any run. Every stage checkpoints there.

## Calibrated constants — change with evidence, not taste

- `shorts/store.py:SIMILARITY_THRESHOLD = 0.32` — reworded duplicates score 0.39-0.49, unrelated
  premises below 0.02.
- `shorts/images.py:MIN_DIRECTIONALITY = 0.18` — a real abstract-gradient failure scored 0.063;
  four usable shots scored 0.535-0.809. Edge density does **not** work here (stripe boundaries
  score higher than real subjects).
- `shorts/config.py:MAX_DIRECTIONS = 3` — directing every line sounds like a cartoon.
