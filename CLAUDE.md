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

**`is_usable()` does not mean "the image is right".** It proves a frame is not a flat colour field
and not a 1-D gradient. It says nothing about whether the image shows what was asked for, and on
the keyless tier that is the dominant failure. Measured on a real 7-shot render (2026-09-13):
"a man standing in front of an open office fridge" came back as an empty corridor, "a strip of
masking tape on a plastic container" as an abstract beige blob, and only the jar of mustard was
the requested subject — **6 of 7 wrong, all 7 passing `is_usable`**. `images.depicts_subject()`
now asks Gemini's free vision tier a yes/no and re-rolls the seed on a miss. It **fails open**
(unlike the QC gate) because a vision outage must not halt image generation.

**Style drift is measurable, and it is a saturation problem.** `docs/RESEARCH.md` called style
adherence "the unsolved quality problem" without ever measuring it. Across the eight renders in
`work/`, luminance stayed within 26–79 points however incoherent the video looked, while mean
saturation separated cleanly: coherent videos clustered at a ~0.13 spread, incoherent ones at
0.39–0.58. `images.coherence()` measures it and `MAX_SATURATION_RANGE = 0.35` warns on it. Six of
eight samples trip it — that is honest signal, not a bad threshold, since all eight came from the
keyless `sana` tier that ignores the style contract.

**Caption cards break on thought, not on a word count.** Grouping was a flat every-third-word
counter, which produced real cards like `FRIDGE ITEMS. NOT` and `DATE. EVERY SINGLE` — the end of
one thought glued to the start of the next. The caption is the *visual* beat, so that steps on the
timing stage 5 worked to build. A sentence end always breaks the card now, an emphasised word
breaks it, and a clause break lands once the card is already readable.

**Never score humour on a scale.** Absolute LLM joke ratings collapse (88.5% identical scores in
the research this is built on). Every selection is a pairwise tournament — `shorts/tournament.py`.
Any new judging prompt must be pairwise.

**Judging prompts must not get the comedian persona.** `prompts.render(..., VOICE="")` for judges.
A judge carrying the writing persona prefers its own voice and the tournament becomes noise.

**The gate fails closed.** If the quality judge is unreachable, the run must not publish. This
is deliberate — the old pipeline's defining failure was uploading a video built from a placeholder
after every LLM call 404'd. Do not add a "publish anyway" path.

**An LLM key is mandatory.** There is no working keyless text tier: anonymous Pollinations returns
401 on the *first* request (re-probed 2026-09-12 — it is not "a handful then 401"). The rung is no
longer in the ladder at all unless `POLLINATIONS_TOKEN` is set, because two 120s timeouts per
failure path bought nothing. Images and voice *do* work keyless; text does not.

A video needs **~36 LLM calls**, not the ~14 this file and `config.py` used to claim: ideate 1 +
premise tournament <=19 + script 1 + punch-up gen 1 + punch-up tournament <=10 + direct 1 +
shotlist 1 + qc 1 + metadata 1.

**Silence must fail the gate, and it nearly didn't.** ffmpeg prints `Input Integrated: -inf LUFS`
for a silent file; `measure_loudness`'s regex could not match `inf` so it returned `None`; and
`qc.py` guarded the check with `if lufs is not None`. The loudness check was therefore skipped
*exactly* when it mattered, and a fully silent video passed every mechanical check and uploaded.
`probe()`'s `has_audio` does not help — a silent stream is still a stream.

This matters because **edge-tts returns valid, correctly-sized, entirely silent audio** when
Microsoft's Sec-MS-GEC anti-abuse check rejects the caller, which is routine from datacenter IPs
— i.e. every GitHub Actions runner. A byte-length check cannot see it. `voice.py` now measures
peak dBFS per chunk and on the assembled track, and `qc.py` hard-fails below
`MIN_VOICE_PEAK_DBFS = -45.0`. Measured on this build: a real render peaks at -1.3 dBFS, digital
silence at -91.0. **Do not soften these to warnings.**

Corollary, which inverts the usual framing: **in CI, Orpheus is the safe path and edge-tts is the
risky one.** Orpheus is a keyed API and works fine from datacenter IPs. Keep `edge-tts` pinned
with `>=`, never `==`: Microsoft rotates the token algorithm and only current releases track it.

**Upload quota is no longer the constraint.** `videos.insert` cost 1,600 units of a shared
10,000/day pool — 6 uploads/day — and the whole schedule was built around that. Since 2026-06-01
it is **1 unit against a dedicated 100 calls/day bucket**. The binding constraint is now Groq
Orpheus at 100 RPD (~7 calls/video), then per-run image wall-clock. Anything in this repo still
asserting 6/day is stale.

**Rate-limit every provider, not just images.** `images.py` had a process-wide 16s gate from day
one; `voice.py` had nothing despite documenting Orpheus's 10 RPM in its own docstring, so a
7-beat script fired 7 calls back to back. Both are gated now. If you add a provider, pace it.

**Model ids belong in `shorts/config.py`.** The old pipeline hardcoded `gemini-1.5-flash` inline;
when Google retired it, every call 404'd. Keep them in one place so a retirement is a one-line fix.

That fix is only one line if you *check*. On 2026-09-12, three of the four rungs were already dead
and nobody had noticed, because the ladder degrades quietly: `moonshotai/kimi-k2.6:free` (the
`:free` endpoint is gone — the model went paid), both `llama-3.*` Groq ids (off the free tier),
and keyless Pollinations. Only Gemini worked. **Re-verify every id against the provider's live
model list before trusting this file.**

Also: Groq's free tier is 8,000 TPM and its limiter counts `prompt + max_tokens`. The old
`max_tokens: 8192` made every Groq call 413 no matter which model id was set — a correct id is
necessary but not sufficient. Hence `LLM_MAX_OUTPUT_TOKENS = 4096`.

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
