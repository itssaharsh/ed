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
        "finance",
        28,
        "personal finance, wealth psychology, banking secrets, investing mindset",
        ["stock market", "money", "trading", "bank", "investment", "financial"],
        "#shorts #finance #money #wealth #investing #facts #mindblowing #moneytips",
    ),
    (
        "ai_tech",
        25,
        "AI surveillance, algorithmic manipulation, tech company secrets, machine learning dangers",
        ["artificial intelligence", "server room", "coding", "technology", "digital", "cyber"],
        "#shorts #ai #tech #technology #artificialintelligence #facts #scary #future",
    ),
    (
        "dark_psychology",
        20,
        "dark psychology, manipulation tactics, cognitive biases exploited by corporations, persuasion science",
        ["shadow", "silhouette", "crowd", "person thinking", "abstract mind", "psychology"],
        "#shorts #psychology #psychologyfacts #behavior #facts #manipulation #mindblown",
    ),
    (
        "dark_history",
        15,
        "forgotten atrocities, government cover-ups, disturbing historical experiments, suppressed history",
        ["abandoned", "ruins", "vintage", "old building", "historical", "archive"],
        "#shorts #history #darkhistory #facts #scary #conspiracy #mindblown #educational",
    ),
    (
        "nature_horror",
        12,
        "deep sea creatures, parasites, extreme predator behavior, disturbing animal biology",
        ["ocean", "underwater", "deep sea", "wildlife", "jungle", "nature"],
        "#shorts #nature #science #deepocean #scary #wildlife #facts #mindblowing",
    ),
]

@dataclass(frozen=True)
class PipelineConfig:
    workspace: Path
    gemini_api_key: str | None
    pexels_api_key: str | None
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
        youtube_client_secrets=Path(os.environ.get("YOUTUBE_CLIENT_SECRETS_PATH", workspace / "client_secrets.json")),
        youtube_credentials=Path(os.environ.get("YOUTUBE_CREDENTIALS_PATH", workspace / "credentials.json")),
        output_audio=workspace / "audio.mp3",
        output_srt=workspace / "captions.srt",
        output_background=workspace / "background.mp4",
        output_final=workspace / "final_short.mp4",
    )
