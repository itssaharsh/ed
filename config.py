import os
import logging
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
# en-US-GuyNeural: crisp, articulate, conversational American male voice
# Perfect for the "roasting a friend" stand-up storytelling style of the reference video.
DEFAULT_VOICE = "en-US-GuyNeural"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_PRIVACY_STATUS = "public"
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"


# Categories: (id, weight, description, video_search_terms, hashtags)
# All categories are comedy-first — chaos, absurdity, and unhinged takes only.
_CONTENT_CATEGORIES: list[tuple[str, int, str, list[str], str]] = [
    (
        "relatable_comedy",
        28,
        "everyday absurdities, awkward social interactions, unspoken rules of society, stand-up observational humor — the weirder and more niche the better",
        ["coffee", "office", "crowd", "awkward", "city", "smartphone"],
        "#shorts #comedy #relatable #funny #standup #humor #introvert",
    ),
    (
        "dark_humor",
        27,
        "cynical takes on modern life, existential dread masked as comedy, sarcastic observations about adulthood and the crushing weight of existence",
        ["rain", "commute", "clock", "boredom", "shadow", "empty street"],
        "#shorts #darkhumor #cynical #funny #sarcasm #adulthood #relatable",
    ),
    (
        "corporate_satire",
        20,
        "mocking office culture, passive-aggressive emails, pointless meetings, corporate jargon, the absurdity of synergizing your bandwidth",
        ["office", "laptop", "meeting", "shaking hands", "business", "coffee"],
        "#shorts #corporate #officehumor #worklife #comedy #satire #relatable",
    ),
    (
        "modern_dating",
        12,
        "the horrors of dating apps, awful first dates, texting anxiety, cynical romance — love is a scam and we have receipts",
        ["restaurant", "smartphone", "couple", "nightclub", "coffee shop"],
        "#shorts #dating #datinghumor #relationships #comedy #funny #datingapps",
    ),
    (
        "internet_culture",
        13,
        "doomscrolling, social media addiction, influencer absurdity, chronically online behavior — your brain is now 80% TikTok sound effects",
        ["smartphone", "scrolling", "neon", "typing", "computer", "glowing screen"],
        "#shorts #internet #socialmedia #doomscrolling #comedy #relatable #satire",
    ),
]

@dataclass(frozen=True)
class PipelineConfig:
    workspace: Path
    gemini_api_key: str | None
    pexels_api_key: str | None
    groq_api_key: str | None
    youtube_client_secrets: Path
    youtube_credentials: Path
    output_audio: Path
    output_srt: Path
    output_background: Path
    output_final: Path
    gemini_model: str = DEFAULT_GEMINI_MODEL
    voice: str = DEFAULT_VOICE
    privacy_status: str = DEFAULT_PRIVACY_STATUS

@dataclass(frozen=True)
class ContentBrief:
    title: str
    description: str
    script: str
    search_query: str
    category: str = "space"
    # Tuple of 8 stock-footage search terms — one per 3-4 second video cut.
    # Using tuple (not list) so the frozen dataclass remains hashable.
    video_keywords: tuple[str, ...] = ()
    # 3-5 most ridiculous words to be highlighted in red in captions.
    punchline_words: tuple[str, ...] = ()

@dataclass(frozen=True)
class SrtCue:
    start: float
    end: float
    text: str
    # Set to True when this burst contains a punchline word — renders in red.
    is_emphasis: bool = False

def build_config() -> PipelineConfig:
    workspace = Path(__file__).resolve().parent
    return PipelineConfig(
        workspace=workspace,
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        pexels_api_key=os.environ.get("PEXELS_API_KEY"),
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        youtube_client_secrets=Path(os.environ.get("YOUTUBE_CLIENT_SECRETS_PATH", workspace / "client_secrets.json")),
        youtube_credentials=Path(os.environ.get("YOUTUBE_CREDENTIALS_PATH", workspace / "credentials.json")),
        output_audio=workspace / "audio.mp3",
        output_srt=workspace / "captions.srt",
        output_background=workspace / "background.mp4",
        output_final=workspace / "final_short.mp4",
    )
