# Setup

Everything here is free. The whole point of the stack is that no step needs a card.

## The one thing you must have

**An LLM API key.** The pipeline makes ~14 model calls per video and there is no usable keyless
option — the anonymous Pollinations text tier allows a handful of requests and then returns
`401 A valid API key is required` regardless of pacing (probed, 2026-08-22).

Pick one:

| Provider | Free tier | Get a key |
|---|---|---|
| **Google Gemini** *(recommended)* | 500 requests/day on `gemini-2.5-flash` | <https://aistudio.google.com/apikey> |
| OpenRouter | `moonshotai/kimi-k2.6:free` — zero token cost, rate-limited | <https://openrouter.ai/keys> |
| Groq | generous daily allowance | <https://console.groq.com/keys> |

```bash
export GEMINI_API_KEY="..."
```

Gemini's 500 RPD covers ~35 videos/day of generation, far above the 6/day upload ceiling.

---

## Strongly recommended

### Groq — for the voice (this is the fix for "bad voice")

Unlocks **Orpheus** (`canopylabs/orpheus-v1-english`): bracketed vocal direction (`[deadpan]`,
`[sarcastic]`), inline non-verbals (`<sigh>`, `<laugh>`), six voice personas. Free tier is
**10 requests/minute, 100/day**; a video uses about 5.

Without it the pipeline falls back to `edge-tts`, which Microsoft has stripped of SSML — only
global rate/volume/pitch, no pauses, no emphasis. Functional, but flat. This is the single
biggest quality difference per unit of setup effort.

<https://console.groq.com/keys> → `export GROQ_API_KEY="..."`

### Cloudflare — for the images (this is the fix for the visuals)

Unlocks **FLUX.1-schnell** (`@cf/black-forest-labs/flux-1-schnell`), Apache-2.0 so it is fine for
monetised video. Free tier is **10,000 neurons/day** ≈ **145 portrait images/day**; six videos use
about 4,100.

Without it the pipeline falls back to Pollinations' anonymous tier, which serves only `sana` at
580x1015, ignores the seed, and mangles faces at close range. It works, but it looks cheap — and
it is **slow**: measured at roughly 1–2 minutes per image against about 5 seconds on Cloudflare,
so a six-shot video spends ~10 minutes in image generation alone.

1. <https://dash.cloudflare.com/sign-up> (free, no card)
2. Copy your **Account ID** from the dashboard sidebar
3. **My Profile → API Tokens → Create Token → Workers AI (Read)**

```bash
export CLOUDFLARE_ACCOUNT_ID="..."
export CLOUDFLARE_API_TOKEN="..."
```

> Cloudflare's docs disagree about whether `flux-1-schnell` accepts `width`/`height` (one page
> documents 256–1920, the model schema page omits them). The code asks for 768x1344, and if the
> request is rejected for those parameters it retries at the model default and cover-crops to
> 9:16. Either behaviour works, so this needs no action from you.

---

## Optional

- `POLLINATIONS_TOKEN` — <https://enter.pollinations.ai> upgrades the image fallback from `sana`
  to flux with working seed control.
- `YOUTUBE_PRIVACY` — `private` (default), `unlisted`, or `public`.

---

## YouTube upload

1. Google Cloud Console → new project → enable **YouTube Data API v3**.
2. **OAuth consent screen** → set it to **Production**, not Testing.
   In Testing, refresh tokens expire after **7 days** and the pipeline dies with `invalid_grant`.
   This is the single most common way this setup breaks.
3. Credentials → **OAuth client ID** → *Desktop app* → download as `client_secrets.json`.
4. Run the flow once locally to mint `credentials.json`:

```bash
python -c "
from google_auth_oauthlib.flow import InstalledAppFlow
f = InstalledAppFlow.from_client_secrets_file(
    'client_secrets.json', ['https://www.googleapis.com/auth/youtube.upload'])
open('credentials.json','w').write(f.run_local_server(port=0).to_json())
"
```

**Quota:** 10,000 units/day, `videos.insert` costs 1,600 → **6 uploads/day, hard ceiling.**

---

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python run.py --doctor           # check every provider first
.venv/bin/python run.py --dry-run          # build everything, upload nothing
.venv/bin/python run.py --dry-run --seed 7 # reproducible
.venv/bin/python run.py                    # build, gate, upload as private
.venv/bin/python run.py --privacy public   # opt in to publishing
```

`ffmpeg` is **not** a system dependency — `imageio-ffmpeg` ships a static build, so there is no
`apt install` and no `sudo`.

Artefacts land in `work/<run_id>/`: every stage's JSON, the generated images, per-line audio,
the ASS captions, and `final.mp4`. When something looks wrong, read those before re-running.

---

## GitHub Actions

Add each key as a repository secret (**Settings → Secrets and variables → Actions**):

`GEMINI_API_KEY`, `GROQ_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`,
`YOUTUBE_CLIENT_SECRETS_JSON`, `YOUTUBE_CREDENTIALS_JSON`, and `GH_PAT` (so the workflow can
write refreshed OAuth tokens back into the secret).

The workflow commits `assets/premise_store.jsonl` back to the repo after each run. That file is
the pipeline's memory of what it has already made — **without it, premises repeat forever.**
