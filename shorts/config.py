"""Configuration, content categories, and hard constants."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("shorts")

# ── Frame geometry ──────────────────────────────────────────────────────────
WIDTH, HEIGHT, FPS = 1080, 1920, 30

# YouTube Shorts UI chrome, measured in source pixels on a 1080x1920 frame.
# Top: logo/search. Bottom: title, channel, subscribe, description. Right: engagement rail.
UI_TOP = 150
UI_BOTTOM = 420          # chrome occupies y >= HEIGHT - UI_BOTTOM (1500)
UI_RIGHT = 140           # chrome occupies x >= WIDTH - UI_RIGHT (940)
SAFE_TOP, SAFE_BOTTOM = 380, 1480
SAFE_LEFT, SAFE_RIGHT = 60, 920

# Captions sit below the subject's face and above the bottom chrome.
CAPTION_BAND_Y = 1140

# ── Pacing ──────────────────────────────────────────────────────────────────
TARGET_SECONDS = 40
TARGET_WORDS = 105
MIN_DURATION, MAX_DURATION = 20.0, 60.0
MIN_WORDS, MAX_WORDS = 60, 140

# ── Generation volume (the "quantity then ruthless selection" strategy) ─────
N_PREMISES = 12
N_PUNCHLINES = 6
TOURNAMENT_ROUNDS = 3

# ── Categories ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Category:
    id: str
    weight: int
    brief: str
    tags: str


CATEGORIES: tuple[Category, ...] = (
    Category(
        "domestic_absurd", 26,
        "the small insane things people do in their own homes — systems nobody asked for, "
        "objects used wrong, routines defended with total sincerity",
        "#shorts #comedy #relatable #funny #standup",
    ),
    Category(
        "corporate_satire", 22,
        "office culture as observed anthropology — meetings that could be emails, the language "
        "people adopt at work, the theatre of looking busy",
        "#shorts #comedy #officehumor #worklife #corporate",
    ),
    Category(
        "social_rules", 20,
        "the rules everyone follows that nobody agreed to — queue etiquette, lift behaviour, "
        "the choreography of passing someone on a narrow pavement",
        "#shorts #comedy #relatable #socialanxiety #funny",
    ),
    Category(
        "misplaced_confidence", 18,
        "people who are certain and wrong — the friend who insists he knows a shortcut, "
        "confident incompetence pursued to its natural conclusion",
        "#shorts #comedy #funny #friends #standup",
    ),
    Category(
        "modern_dread", 14,
        "3am decisions, doomscrolling, the specific dread of adult admin — funny because it is "
        "recognisable, not because it is bleak",
        "#shorts #comedy #darkhumor #relatable #adulting",
    ),
)

# Style is chosen by joke mechanism, not at random, so the look matches the joke.
MECHANISM_STYLE = {
    "bad system": "flat_absurd",
    "unspoken rule": "flat_absurd",
    "sincere wrong effort": "grain_docu",
    "escalating commitment": "grain_docu",
    "misplaced confidence": "neon_late",
}
DEFAULT_STYLE = "flat_absurd"

# ── Voice ───────────────────────────────────────────────────────────────────
ORPHEUS_MODEL = "canopylabs/orpheus-v1-english"
ORPHEUS_VOICES = ("austin", "daniel", "troy", "autumn", "diana", "hannah")
ORPHEUS_CHAR_LIMIT = 200          # hard cap per request
EDGE_FALLBACK_VOICE = "en-US-AndrewNeural"

# Groq's Orpheus free tier is 10 RPM. images.py has paced its provider since day one; this
# module did not, so a 7-beat script fired 7 calls back to back and tripped the limit on the
# first real run with a key. 6.0s is the arithmetic floor; 7.0 leaves margin for clock skew.
GROQ_TTS_MIN_INTERVAL = 7.0

# The silence floor, in dBFS peak. Measured on this build: a real render peaks at -1.3 dB,
# digital silence at -91.0 dB. Anything at or below this is not speech.
#
# This exists because edge-tts returns a *valid, well-formed, entirely silent* MP3 when
# Microsoft's Sec-MS-GEC anti-abuse check rejects the caller - which is what happens from
# datacenter IPs, i.e. every GitHub Actions runner. A byte-length check does not catch it.
MIN_VOICE_PEAK_DBFS = -45.0

MAX_DIRECTIONS = 3                # at most 3 directed lines per script
MAX_NONVERBALS = 1
MAX_PAUSE_MS = 650

# ── Image generation ────────────────────────────────────────────────────────
CF_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
IMAGE_W, IMAGE_H = 768, 1344      # 9:16; upscaled to 1296x2304 for camera-move headroom
IMAGE_STEPS = 4                   # flux-schnell is a 4-step distilled model
MASTER_W, MASTER_H = 1296, 2304   # 1.2x of 1080x1920 — headroom for zoom/pan

# Style coherence: the largest mean-saturation spread across a video's shots before it stops
# reading as one production. docs/RESEARCH.md called style adherence "the unsolved quality
# problem" but never measured it; this is the measurement.
#
# Calibrated over the 8 real renders in work/ (2026-09-13):
#   0.124  neon_late,   4 shots  - per-shot 0.61 0.52 0.64 0.63, unmistakably one look
#   0.130  flat_absurd, 4 shots  - 0.27 0.40 0.30 0.38, coherent
#   0.387 - 0.453                - visibly drifting
#   0.562  grain_docu,  7 shots  - 0.15 0.59 0.23 0.31 0.28 0.03 0.07, photoreal next to
#                                  near-greyscale next to a warm abstract blob
#   0.579  flat_absurd, 7 shots  - worst observed
# Coherent runs cluster at ~0.13 and incoherent ones at ~0.39+, so 0.35 sits in the gap.
#
# WARNING ONLY, deliberately. Six of the eight samples trip it - which is the honest signal, not
# an over-sensitive threshold: every one of those renders came from the keyless `sana` tier that
# ignores the style contract. Expect this to go quiet once CLOUDFLARE_* is configured and
# flux-1-schnell honours the contract. Promote it to a hard failure only with renders from a
# provider that can actually hold a style, and with more than eight samples.
MAX_SATURATION_RANGE = 0.35

# ── LLM models, in fallback order ───────────────────────────────────────────
#
# Verified against each provider's live model list on 2026-09-12. Re-verify before trusting
# any of these: free tiers churn, and a stale id here is precisely what killed v1 - every
# gemini-1.5-* call 404'd, the run continued, and it published a video built from a placeholder.
GEMINI_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite")

# Was "moonshotai/kimi-k2.6:free". The model is still on OpenRouter, but the :free endpoint is
# gone - Kimi went paid, so RESEARCH.md's "the best humour writer reachable free" is no longer
# true. Of the 19 remaining :free ids, nemotron-3-ultra is much the largest.
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

# Was ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"). Neither is on Groq's free tier any
# more. These three are, at 30 RPM / 1K RPD / 8K TPM / 200K TPD.
GROQ_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b")

POLLINATIONS_TEXT_MODEL = "openai-fast"

# Groq's free gpt-oss tier is 8,000 TPM, and its limiter counts prompt + max_tokens together.
# The largest prompt this repo renders is 01_ideate carrying 40 recent premises, ~2.6K tokens.
# At the old max_tokens of 8192 that is ~10.8K against an 8K ceiling, so every Groq call
# returned 413 regardless of which model id was set. 4096 leaves comfortable headroom, and the
# largest output the pipeline ever asks for (12 premises x 7 fields) is ~2.5K.
LLM_MAX_OUTPUT_TOKENS = 4096

# Gemini gets its own, larger budget - and the reason is the first live CI run (2026-09-19).
# gemini-2.5-flash is a *thinking* model: its reasoning tokens are drawn from max_output_tokens.
# Applying the Groq-motivated 4096 cap to Gemini let thinking consume the budget, and the JSON
# premise list was truncated mid-string on all three attempts - the run died at stage 1 on a
# 200 OK. Gemini's free tier is not TPM-shaped like Groq's (250K TPM), so a large ceiling costs
# nothing; the thinking budget is capped separately so it can never again starve the answer.
GEMINI_MAX_OUTPUT_TOKENS = 16384

# ── Role routing ────────────────────────────────────────────────────────────
# Measured in CI on 2026-09-19: Gemini's free tier refused everything after ~20 successful
# requests in a day, even at 1.3 requests/minute - a daily cap, 25x lower than the 500/day this
# pipeline was planned around. The same run spent that allowance on the *premise tournament*
# (~19 judge calls) inside its first minute, so the script, the punch-up and the direction were
# all written by the fallback model. Judging is ~80% of all calls and needs a verdict, not
# prose; writing is ~6 calls a video and is where model quality actually shows.
#
# So: writing goes to Gemini first, judging goes to Groq first, and Gemini is judging's last
# resort. Each provider's quota is per model, and every provider tries its next model on a
# rate limit before giving up.
ROLE_ORDER = {
    "generate": ("gemini", "groq", "openrouter", "pollinations"),
    "judge":    ("groq", "openrouter", "gemini", "pollinations"),
}
# A judge answers with ~150 tokens of JSON. Groq's 8K TPM limiter counts prompt + max_tokens,
# so the 4096 writing budget let only one judge call through per minute - which is why the CI
# run waited on Groq rate limits three times in a row.
JUDGE_MAX_TOKENS = 1024
GEMINI_THINKING_BUDGET = 2048

# ── Publishing ──────────────────────────────────────────────────────────────
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
# Public by default. This was "private", and the scheduled workflow relied on that default
# (it passes `inputs.privacy || 'private'`, and inputs are null on a schedule event) - so every
# cron run would have uploaded privately and the channel would have stayed empty however many
# times the pipeline succeeded. Override per-run with --privacy or $YOUTUBE_PRIVACY.
DEFAULT_PRIVACY = os.environ.get("YOUTUBE_PRIVACY", "public")


def _env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v and v.strip():
            return v.strip()
    return None


@dataclass
class Config:
    root: Path = ROOT
    work: Path = field(default_factory=lambda: ROOT / "work")
    prompts_dir: Path = field(default_factory=lambda: ROOT / "prompts")
    assets: Path = field(default_factory=lambda: ROOT / "assets")
    store_path: Path = field(default_factory=lambda: ROOT / "assets" / "premise_store.jsonl")

    gemini_key: str | None = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    groq_key: str | None = field(default_factory=lambda: _env("GROQ_API_KEY"))
    openrouter_key: str | None = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    pollinations_token: str | None = field(default_factory=lambda: _env("POLLINATIONS_TOKEN"))
    cf_account: str | None = field(default_factory=lambda: _env("CLOUDFLARE_ACCOUNT_ID"))
    cf_token: str | None = field(default_factory=lambda: _env("CLOUDFLARE_API_TOKEN"))

    privacy: str = DEFAULT_PRIVACY
    dry_run: bool = False

    def has_llm(self) -> bool:
        """An LLM key is mandatory.

        The keyless Pollinations text tier looked like a viable floor but is not: the anonymous
        allowance is a handful of requests, after which it returns 401 "A valid API key is
        required" regardless of pacing. This pipeline makes ~14 LLM calls per video, so it needs
        a real key. Gemini's free tier (500 requests/day) is the easiest to get.
        """
        return bool(self.gemini_key or self.openrouter_key or self.groq_key)

    def capabilities(self) -> dict[str, str]:
        """What this run can actually do, given the keys present."""
        if self.gemini_key:
            llm = "gemini"
        elif self.openrouter_key:
            llm = "openrouter"
        elif self.groq_key:
            llm = "groq"
        else:
            llm = "NONE - pipeline cannot run"

        if self.cf_account and self.cf_token:
            img = "cloudflare flux-schnell"
        elif self.pollinations_token:
            img = "pollinations (token)"
        else:
            img = "pollinations anonymous (sana, 580x1015, degraded)"

        voice = "groq orpheus" if self.groq_key else "edge-tts (flat, degraded)"
        return {"llm": llm, "images": img, "voice": voice}
