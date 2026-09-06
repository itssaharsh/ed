# Architecture

## Why the old pipeline produced bad videos

Read `ed/` for the original. Five root causes, each mapped to a fix.

| # | Root cause | Where | Fix |
|---|---|---|---|
| 1 | **Silent quality collapse.** Every Gemini model 404'd (`gemini-1.5-flash` etc. are dead ids); the run continued, downloaded a stock clip for the query `"person"`, and uploaded it. | `ed/gemini.py`, `ed/main.py` | Hard **fail-closed** gate. No brief → no render → no upload, ever. |
| 2 | **Footage chosen by keyword roulette.** 8 "visual irony" keywords → Pexels → whatever it had. Nothing tied a clip to a moment in the script. | `ed/assets.py:_extract_visual_keywords` | Kill stock entirely. Generate a **shot list bound to script beats**, then generate an image per shot. |
| 3 | **Scripts padded with canned filler.** `_stretch_script_to_target` appends from a list of 7 fixed lines until the word count hits 75. Every video ends with the same sentences and the timing of the joke is destroyed. | `ed/audio.py:_stretch_script_to_target` | **Delete.** Length is controlled at write time; pacing comes from measured audio + designed silences. |
| 4 | **Flat delivery.** `edge-tts` at a global `+18%` rate. `[PAUSE]` was replaced by a full stop. No emphasis, no beat, no attitude. | `ed/audio.py` | **Orpheus** with per-line vocal direction + **explicit silence** between beats. |
| 5 | **No memory.** Nothing recorded what was made, so premises repeat forever. | — | Persistent premise store with n-gram + token-overlap dedup. |

Also fixed: `moviepy` compositing replaced by a single ffmpeg graph; PIL-rendered caption PNGs
replaced by libass; the same file is no longer uploaded to two channels (duplicate-content risk).

---

## Pipeline

```
                                    ┌──────────────────────────┐
                                    │  premise store (JSONL)   │  ← dedup, all stages read
                                    └───────────┬──────────────┘
                                                │
 1 IDEATE      12 premises, divergent ──────────┤
 2 TOURNAMENT  pairwise Swiss + Bradley-Terry ──┤   humour is judged by COMPARISON,
 3 DRAFT       beat-structured script ──────────┤   never by an absolute 1-10 score
 4 PUNCH-UP    N alt punchlines → tournament ───┤
 5 DIRECT      per-line Orpheus delivery tags ──┤
 6 SHOT LIST   beats → shots → image prompts ───┘
                       │
        ┌──────────────┴───────────────┐
        │                              │
 7 IMAGES  Cloudflare FLUX-schnell   8 VOICE  Groq Orpheus, per line
           → Pollinations (token)             → Gemini TTS → edge-tts
           → Pollinations (keyless)           measured durations, designed silences
        │                              │
        └──────────────┬───────────────┘
                       │
 9 CAPTIONS   ASS karaoke, word-level, safe-zone aware
10 RENDER     one ffmpeg graph: motion → cuts → captions → mix → loudnorm
11 QC GATE    hard checks + pairwise judge vs. a known-good baseline
12 PUBLISH    YouTube (fail-closed) + write premise to store
```

Every stage is a pure function `state -> state`, checkpointed to `work/<run_id>/`. A failed upload
never costs a re-render.

---

## The three quality levers

### (a) Comedy — volume plus ruthless pairwise selection

Grounded in two findings (see `docs/RESEARCH.md` §4):

- **Absolute humour scoring does not work.** 88.5% of LLM 0–100 joke scores collapse to identical
  values. Pairwise comparison reaches cross-judge τ=0.889.
- **LLMs write setups, not punchlines.** The mechanical substitute for human instinct is to
  generate many candidates and select hard.

So: 12 premises → tournament → 1 winner. Then 6 candidate punchlines → tournament → 1 winner.
The judge only ever answers *"which of these two is funnier, and by what mechanism?"*

The prompts also carry an explicit **ban list** of AI-comedy tells ("it's not X, it's Y",
rule-of-three, explaining the joke, "little did they know"). Research caveat honoured: any single
tell has innocent human causes — it is *convergence* that reads as machine-written, so the check
counts them rather than banning one occurrence.

### (b) Visuals — shots bound to beats, one coherent world

Free text-to-video does not exist (§1). So: **AI stills + real camera language**.

- The shot list is derived from **script beats**, so a cut lands *on* the joke, not near it.
- A **style contract** — one style suffix string, one palette, one lens vocabulary, one character
  sheet — is injected into every image prompt in a video, so 10 shots look like one world. Visual
  incoherence between shots is the single biggest tell of cheap AI shorts.
- Motion is a real camera move per shot (push-in / pull-out / drift + handheld float), varied by
  shot so no two adjacent shots move the same way.
- **Deliberately stylised, never photoreal.** This is both taste and policy: YouTube's
  "Modified or Synthetic" label applies to content that could be mistaken for real footage;
  illustration and cartoon styling are exempt, and photoreal synthetic media has been
  auto-detected since May 2026.

### (c) Voice — direction and silence

- **Orpheus** (`canopylabs/orpheus-v1-english`) takes bracketed direction — `[deadpan]`,
  `[sarcastic]`, `[whisper]`, `[dramatic]` — and inline non-verbals `<laugh>`, `<sigh>`, `<giggle>`.
- Its **200-character cap forces per-line synthesis**, which is what makes precise comic timing
  possible: each line is rendered separately, then silences of *designed* length are inserted
  between beats (a punchline gets a longer pre-beat than a setup line).
- Timing flows *out* of the audio, not into it. Each line's real duration is measured, so captions
  and cuts are locked to the actual performance.

---

## Rendering

`zoompan` is unusable — measured at **109 s to render 3 s** of 1080x1920. It rescales the entire
input on every output frame.

The replacement, measured at **4.3 s per 4 s clip** (~30x faster), gives zoom + pan + handheld in
one chain:

```
scale=w='2*floor(W0*(1+Z*t/D)/2)':h=-2:eval=frame,     ← zoom (per-frame swscale)
crop=1080:1920:x='(in_w-1080)/2 + Ax*sin(t*Fx)'        ← pan + handheld drift
              :y='(in_h-1920)/2 + Ay*sin(t*Fy)'
```

Captions are **libass**, not PIL. This build of ffmpeg has no `drawtext`, and libass is better
regardless: real typography, per-word karaoke highlighting, outlines, and `\t()` transforms for the
scale-pop on each word — all in a single pass.

**Caption placement** respects the Shorts UI: top 150px (logo/search), bottom 420px from y≈1500
(title/subscribe/description), right 140px from x≈940 (engagement rail). Captions sit in a band at
**y ≈ 1050–1250** — clear of the chrome, below the subject's face.

---

## Two ways in: generated, or hand-authored

Stages 1–6 exist to make a model produce a premise, a script, a performance and a shot list. A
**brief** supplies all four as JSON, so the pipeline runs with no LLM key at all.

```
  generated path                       brief path
  ──────────────                       ──────────
  1 ideate      ─┐                     briefs/x.json
  2 tournament   │                          │
  3 draft        ├─ needs an LLM key         │  needs nothing
  4 punch-up     │                          │
  5 direct       │                          │
  6 shot list   ─┘                          │
        └──────────────┬────────────────────┘
                       ▼
              run._finish()  — stages 6-timing, 7, 10, 9, 11, 12
```

Both converge in `run._finish`, so there is **one** render pipeline, not two that drift apart.

The brief path exists for three real situations: the daily quota is spent; you want to write the
jokes yourself; or an assistant is acting as the writer and hands over finished JSON. Briefs are
validated strictly on load — spoken length estimated against the gate's duration floor *before*
any image is generated, hook and punch present, emphasis words actually in their lines, no two
adjacent shots sharing a camera move.

One honest caveat: with no key the pairwise comedy judge cannot run, so the gate falls back to
mechanical checks and says so. The humour then rests entirely on the author.

## Rate limits shape the schedule, not just the budget

Free image tiers limit **per IP**, and this changes the architecture rather than being a footnote.
Measured on the keyless tier:

| approach | throughput |
|---|---|
| 3 renders in parallel | 1 image in 7 minutes; the rest sat in 429 backoff |
| 1 render, paced 16s apart | 1 image every 60–180s |

Parallelism was roughly **five times slower**. So `shorts/images.py` holds a process-wide gate
between anonymous requests, backs off past the full window on a 429, and trips a circuit breaker
after N consecutive failures rather than grinding through a dead provider for the rest of the job.
`batch.py` renders briefs one at a time for the same reason.

The corollary for authoring: **fewer shots renders faster.** A shot may span several lines (an
uncovered line holds the previous image), so four shots across seven lines cuts image generation
by 43% with no real loss of pacing. Only the hook and the punch strictly need their own frame.

## Fallback ladders

Each stage degrades instead of failing, and **every ladder ends in something that needs no API key**,
so the pipeline can always produce output.

| Stage | Primary | Then | Then | Keyless floor |
|---|---|---|---|---|
| LLM | Gemini `gemini-2.5-flash` (500 RPD) | OpenRouter `moonshotai/kimi-k2.6:free` | Groq | Pollinations `openai-fast` |
| Images | Cloudflare `@cf/black-forest-labs/flux-1-schnell` (~145/day) | Pollinations + token | — | Pollinations anonymous (`sana`) |
| Voice | Groq Orpheus (100 RPD) | Gemini TTS (15 RPD) | — | `edge-tts` |

The one thing that never degrades is the **QC gate**. If quality cannot be established, the run
publishes nothing.

---

## Daily budget at 6 videos/day

| Resource | Used | Free cap | Headroom |
|---|---|---|---|
| Cloudflare neurons | ~10 imgs x 69 x 6 = **4,140** | 10,000/day | 2.4x |
| Groq Orpheus calls | ~5 x 6 = **30** | 100/day | 3.3x |
| Gemini text calls | ~14 x 6 = **84** | 500/day | 5.9x |
| YouTube upload units | 1600 x 6 = **9,600** | 10,000/day | **1.04x ← tightest** |

The binding constraint is the YouTube API quota, exactly as before: **6 uploads/day is a hard
ceiling** per Google Cloud project.
