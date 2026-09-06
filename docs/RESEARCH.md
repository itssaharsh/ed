# Free-tier research findings — August 2026

Every claim here was checked against the live web (259 searches) or probed directly from this
machine. **Free tiers churn fast.** Re-verify anything older than ~3 months before trusting it.
Entries marked `PROBED` were tested with a real HTTP request from this repo.

---

## 1. Verdict: free AI *video* generation is not viable at 6 videos/day

This was the decisive question, because it determines the entire visual architecture.

| Route | Status | Evidence |
|---|---|---|
| Google Veo via Gemini API | **Paid only.** No free tier. `veo-3.1-generate-preview` = $0.40/sec (720p/1080p), $0.60/sec (4K). Legacy `veo-3.0-generate-001` shut down 2026-06-30. | ai.google.dev pricing |
| HF Inference Providers (t2v) | Free monthly credits cover **<10 clips/month**. Queues. | huggingface.co/docs/inference-providers/pricing |
| HF Spaces ZeroGPU | 25 min H200/day is a **PRO ($9/mo)** benefit. Free accounts get a small quota (forum reports ~180s). Also intended for hosting *your own* Space, not batch jobs. | hub-docs spaces-zerogpu |
| Pollinations `/video/` | Video models exist (`veo`, `seedance-*`, `wan`, `wan-fast`, `grok-video-*`, `minimax-h3`). Gated by the **Pollen** credit economy; many are `paidOnly` so free "Quest Pollen" cannot pay for them. | APIDOCS.md |
| Pollinations `/video/` anonymous | `PROBED` — **the path is ignored.** `GET /video/{prompt}` returned `content-type: image/jpeg`, `x-model-used: sana`. It treated `video/...` as part of the image prompt. | probe, 2026-08-22 |
| CPU-only local t2v | No current model produces usable 512x896 video on 4 CPU cores in minutes. | — |
| Free GPU CI runners | None. GitHub Actions has no free GPU runner. | — |

**Consequence:** the visual strategy must be **AI-generated stills + genuine procedural motion**,
not text-to-video. This is not a compromise for its own sake — done properly (depth parallax,
real camera language, cut-on-the-beat) it is *more* controllable than 4-second t2v clips, and it
is the only route that costs $0.

---

## 2. Images — the visual backbone

Need ≈10–20 stills per video → ≈120/day at 6 videos/day, 9:16 portrait.

| Service | Free tier | Verdict |
|---|---|---|
| **Cloudflare Workers AI** `@cf/black-forest-labs/flux-1-schnell` | **10,000 neurons/day free.** Billing: $0.000053 per 512x512 tile + $0.00011 per step; neurons are $0.011/1000. A 768x1344 image (6 tiles, 4 steps) ≈ **69 neurons → ~145 images/day**. FLUX.1-schnell weights are Apache-2.0 (commercial OK). | **PRIMARY** |
| Pollinations (free "Seed" token) | Free registration. 1 req/5s, "standard models". Unlocks flux + seed control. | FALLBACK 1 |
| Pollinations anonymous | `PROBED` — keyless, works, **but degraded**: only `sana`, resolution capped to **580x1015** (asked for 768x1344), seed forced to 42, 1 req/15s. Faces are mangled at close range; wide/environment shots are decent. | FALLBACK 2 (zero-setup safety net) |
| Google Gemini image API | **No free tier at all.** Every image model (Gemini 3.1 Flash Image, 3 Pro Image, 2.5 Flash Image, Imagen 4) reads "Not available" on Free. `gemini-2.5-flash-image-preview` shut down 2026-01-15. | REJECT |
| Together.ai FLUX.1-schnell-Free | "3 months free access" = trial, not permanent. | REJECT |

**License trap:** FLUX.1-**schnell** is Apache-2.0 → fine for monetised video. FLUX.1-**dev** is
non-commercial → never use it here. Cloudflare also hosts FLUX.2 [dev] and FLUX.2 [klein] 4B;
check their licence terms before switching.

Note: Cloudflare's `flux-1-schnell` schema docs are inconsistent about `width`/`height`
(one page documents 256–1920, the model schema page omits them). Handled in code: the request asks
for 9:16, and a 400 triggers a retry at the model default followed by a cover-crop.

---

## 3. Voice — the fix for "bad voice"

The old pipeline used `edge-tts` `en-US-GuyNeural` at a flat `+18%` rate. That is the ceiling of
what edge-tts can do, and it is why the delivery is flat.

| Option | Free tier | Expressive control | Verdict |
|---|---|---|---|
| **Groq `canopylabs/orpheus-v1-english`** | **10 RPM / 100 RPD free.** | **Bracketed vocal direction** (`[deadpan]`, `[sarcastic]`, `[whisper]`, `[dramatic]`) + inline non-verbals (`<laugh>`, `<sigh>`, `<giggle>`). 6 voices: autumn, diana, hannah, austin, daniel, troy. **200-char cap per request.** | **PRIMARY** |
| Gemini TTS (`gemini-2.5-flash-preview-tts`) | 3 RPM / **15 RPD** — very tight. | Natural-language style direction. Send the whole script in 1 call → 6 calls/day fits. | FALLBACK 1 |
| edge-tts | Free, keyless, unlimited. | **Custom SSML was removed** by Microsoft. Only `rate`/`volume`/`pitch` on a single `<prosody>`. No pauses, no emphasis, no non-verbals. GPL-3.0 client. | FALLBACK 2 (safety net) |

The Orpheus **200-character cap is a feature, not a limitation**: it forces per-sentence synthesis,
which is exactly what lets us place silences between comedy beats with frame accuracy instead of
hoping the TTS engine pauses in the right place.

Budget check: ~110-word script ≈ 620 chars ≈ 4–5 Orpheus calls. 6 videos/day ≈ **30 calls** vs the
100 RPD cap. Comfortable.

---

## 4. Script — the fix for "bad script / bad humour"

Two research results drive the whole design:

**(a) Pairwise beats absolute scoring for judging humour.** From HumorRank (GTVH-grounded
tournament evaluation): absolute 0–100 rubrics collapse — **88.5% of structured scores came out
identical**, with only a 20.6-point spread across genuinely diverse candidates. Pairwise
comparison achieved cross-judge **τ = 0.889**, and human↔LLM agreement on hard pairs matched
human↔human agreement (Krippendorff α = 0.446 both).

→ **Never ask an LLM to rate a joke 1–10.** Run a pairwise tournament (Swiss pairing +
Bradley–Terry) and take the winner.

**(b) Multi-stage planning beats single-shot generation.** HumorPlanSearch / HuCoT (plan →
select → refine) reported **+18% perceived funniness**, +23% coherence, +31% context relevance
over baseline prompting.

**(c) LLMs write setups, not punchlines.** Studies with professional comedians found LLMs useful
for setup/structure but that humans supplied the punchline. The mechanical substitute is
**volume + ruthless selection**: generate many candidate punchlines, then tournament them.

**Humour leaderboards** (HTB / SemEval-2026 MWAHAHA, Bradley–Terry ratings):
GPT-5 (1314) > **Kimi K2 (1242)** > HumorGen-7B (1097) > Gemini 2.5 Pro (1115 on MWAHAHA).
Kimi K2 is the best humour writer reachable free.

### Free LLM access

| Service | Model | Free limits |
|---|---|---|
| Google Gemini | `gemini-2.5-flash` | 10 RPM / 250k TPM / **500 RPD** |
| OpenRouter | `moonshotai/kimi-k2.6:free` | Zero token cost, rate-limited endpoint |
| Groq | llama / gpt-oss / kimi variants | generous free RPD |
| Pollinations text | `openai-fast` (GPT-OSS 20B) | `PROBED` keyless, 1 req/15s anonymous |

`google-generativeai` is **dead** (support ended 2025-08-31). Use `google-genai`.
The old repo's hardcoded `gemini-1.5-flash` / `gemini-1.5-pro` ids **404** — this is what caused
the observed silent failure in `run_logs.zip`.

### The AI-comedy tells to ban explicitly
"It's not X, it's Y" negative parallelism · tidy rule-of-three lists · em-dash pile-ups ·
explaining the joke after telling it · "little did they know" · "the audacity" · over-signposting ·
safe, sanded-down edges. The research caveat matters: any one of these has innocent human causes —
it is **convergence** of three or four in a short passage that reads as machine-written.

---

## 5. Platform mechanics

**Safe zones on a 1080x1920 Short** (UI chrome overlays the video):
- Top **~150px**: YouTube logo, search, cast.
- Bottom **~420px** (from y≈1500): title, channel name, subscribe, description.
- Right **~140px** (from x≈940): like / dislike / comment / share rail.
- Conservative safe content box: **x 60→920, y 380→1480**.

Captions therefore belong in a band around **y ≈ 1050–1250** — below the subject's face, above the
bottom UI. Dead-centre (y=960) is safe but fights the visual subject.

**AI disclosure policy (this shapes art direction):** YouTube's "Modified or Synthetic" label is
required for content that *appears realistic* — where a viewer could mistake it for real footage of
a real person/place/event. **AI-generated illustration, cartoon styling, and obviously stylised
imagery are exempt.** AI voiceover alone does not trigger disclosure unless it clones a specific
real person. Since May 2026 YouTube auto-detects and labels photoreal synthetic media.

→ **Choose a stylised look, not a photoreal one.** It dodges the label and the auto-detector, and
it reads as an intentional art style rather than as failed realism.

**Upload quota:** YouTube Data API = 10,000 units/day; `videos.insert` = 1600 units → **6 uploads/day
hard cap** per project. Confirmed as the reason the old cron targeted 6/day.

**Duplicate-content risk:** the old `main.py` uploads the *identical file* to two channels. That is
a genuine risk under the mass-produced / repetitive-content policy. Either render a per-channel
variant or publish to one channel.

---

## 5b. What the keyless image tier actually does (measured, 2026-08-22)

Three videos rendered end to end on Pollinations' anonymous tier. Numbers, not impressions:

| | |
|---|---|
| Per image | **60–180s**, occasionally 3s when the prompt is cached |
| 3 renders in parallel | **1 image in 7 minutes**; the rest sat in 429 backoff |
| 1 render, paced 16s apart | steady, no 429s across 15 images |
| A 21.8s / 4-shot video | **~7 minutes** end to end |
| A 21.7s / 7-shot video | **~11 minutes** end to end |

Two conclusions baked into the code: pace anonymous requests process-wide (`shorts/images.py`), and
render briefs serially (`batch.py`). Parallelism was ~5x slower.

**The unsolved quality problem: style adherence.** `sana` largely ignores the style contract. In
one 7-shot video the frames ranged from photoreal to flat cartoon, so the video does not read as
one production. Two things *do* work on this tier and are already in the code:

- **Subject-first prompt ordering.** Leading with the long style paragraph made the model drop the
  subject entirely — a request for "a man at an office fridge" returned a portrait of a stranger.
  Subject first, short style tag after, fixed it.
- **The character sheet.** Injecting one sentence describing the recurring person into every shot
  featuring them does hold the same character across shots (visible in the reply-all video).

Untested idea for the next pass: repeat the medium at both ends of the prompt
("Flat vector illustration. …subject… . flat vector, bold outlines") — prompt-weighting by
repetition. Not tried here because every probe steals a request slot from a running render.

With Cloudflare FLUX-schnell (~5s/image, honours negative prompts and seeds) none of this applies;
it is a limitation of the free-with-no-key tier specifically.

## 6. Local environment (probed on this machine)

- `imageio-ffmpeg` ships a **static ffmpeg 7.0.2** — no `sudo`, no apt. 494 filters.
- Available and load-bearing: `ass`, `subtitles`, `remap`, `displace`, `zoompan`, `xfade`,
  `minterpolate`, `rubberband`, `loudnorm`, `atempo`, `adelay`, `apad`, `amix`, `vignette`,
  `unsharp`, `noise`, `curves`, `chromashift`, `perspective`.
- **`drawtext` is NOT in this build** → caption rendering must go through **libass** (`ass` filter),
  which is better anyway: real typography, per-word karaoke, outlines, transforms, one pass.
- **`zoompan` is pathologically slow**: `PROBED` at **109 s to render 3 s** of 1080x1920 from a
  2160x3840 source. It rescales the full input every output frame. **Do not use it.**
  Use `crop` with time-varying expressions + one fixed `scale` instead.
