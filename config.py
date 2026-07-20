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
DEFAULT_VOICE = "en-US-BrianNeural"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_PRIVACY_STATUS = "public"
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"


_CONTENT_CATEGORIES: list[tuple[str, int, str, list[str], str]] = [
    (
        "relatable_comedy",
        25,
        "everyday absurdities, awkward social interactions, unspoken rules of society, stand-up observational humor",
        ["coffee", "office", "crowd", "awkward", "city", "smartphone"],
        "#shorts #comedy #relatable #funny #standup #humor #introvert",
    ),
    (
        "dark_humor",
        25,
        "cynical takes on modern life, existential dread masked as comedy, sarcastic observations about adulthood",
        ["rain", "commute", "clock", "boredom", "shadow", "empty street"],
        "#shorts #darkhumor #cynical #funny #sarcasm #adulthood #relatable",
    ),
    (
        "corporate_satire",
        20,
        "mocking office culture, passive-aggressive emails, pointless meetings, corporate jargon",
        ["office", "laptop", "meeting", "shaking hands", "business", "coffee"],
        "#shorts #corporate #officehumor #worklife #comedy #satire #relatable",
    ),
    (
        "modern_dating",
        15,
        "the horrors of dating apps, awful first dates, texting anxiety, cynical romance",
        ["restaurant", "smartphone", "couple", "nightclub", "coffee shop"],
        "#shorts #dating #datinghumor #relationships #comedy #funny #datingapps",
    ),
    (
        "internet_culture",
        15,
        "doomscrolling, social media addiction, influencer absurdity, chronically online behavior",
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

@dataclass(frozen=True)
class SrtCue:
    start: float
    end: float
    text: str

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
