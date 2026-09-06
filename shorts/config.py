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

MAX_DIRECTIONS = 3                # at most 3 directed lines per script
MAX_NONVERBALS = 1
MAX_PAUSE_MS = 650

# ── Image generation ────────────────────────────────────────────────────────
CF_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
IMAGE_W, IMAGE_H = 768, 1344      # 9:16; upscaled to 1296x2304 for camera-move headroom
IMAGE_STEPS = 4                   # flux-schnell is a 4-step distilled model
MASTER_W, MASTER_H = 1296, 2304   # 1.2x of 1080x1920 — headroom for zoom/pan

# ── LLM models, in fallback order ───────────────────────────────────────────
GEMINI_MODELS = ("gemini-2.5-flash",)
OPENROUTER_MODEL = "moonshotai/kimi-k2.6:free"   # #2 on humour leaderboards, free endpoint
GROQ_MODELS = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant")
POLLINATIONS_TEXT_MODEL = "openai-fast"

# ── Publishing ──────────────────────────────────────────────────────────────
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
DEFAULT_PRIVACY = os.environ.get("YOUTUBE_PRIVACY", "private")


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
