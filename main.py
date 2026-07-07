from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import edge_tts

# Handle Google GenAI SDK deprecation gracefully by trying the new SDK first
try:
    from google import genai
    from google.genai import types
    USE_NEW_GEMINI_SDK = True
except ImportError:
    import google.generativeai as genai
    USE_NEW_GEMINI_SDK = False

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    CompositeAudioClip,
    ImageClip,
    VideoFileClip,
    vfx,
)
from moviepy.audio.AudioClip import AudioClip, AudioArrayClip
import moviepy.audio.fx.all as afx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Pillow renamed ANTIALIAS to Resampling.LANCZOS in newer versions; provide
# a compatibility alias to avoid runtime errors in downstream libraries.
if not hasattr(Image, "ANTIALIAS"):
    try:
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    except Exception:
        # Fallback for very old/new variants
        setattr(Image, "ANTIALIAS", getattr(Image, "LANCZOS", 1))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
DEFAULT_VOICE = "en-US-GuyNeural"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"  # Updated to newer default
DEFAULT_PRIVACY_STATUS = "public"
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"
# NASA Image and Video Library: free, no API key, actual space/science footage
NASA_VIDEO_SEARCH_URL = "https://images-api.nasa.gov/search"

# When Pexels returns nothing for a niche term, try these broader fallbacks in order.
# Keys are lowercase substrings to match against the search_query.
PEXELS_QUERY_FALLBACKS: dict[str, list[str]] = {
    "magnetar":      ["neutron star", "deep space", "galaxy"],
    "vacuum":        ["cosmos", "deep space", "universe"],
    "bootes":        ["deep space", "galaxy", "stars"],
    "void":          ["deep space", "galaxy", "stars"],
    "spaghett":      ["black hole", "space", "galaxy"],
    "attractor":     ["galaxy", "cosmos", "universe"],
    "cannibali":     ["galaxy", "cosmos", "stars"],
    "heat death":    ["cosmos", "universe", "stars"],
    "cosmic string": ["cosmos", "space", "galaxy"],
    "rogue planet":  ["space", "galaxy", "stars"],
    "neutron":       ["deep space", "galaxy", "stars"],
    "sagittarius":   ["black hole", "milky way", "galaxy"],
    "dark matter":   ["galaxy", "cosmos", "universe"],
    "pulsar":        ["neutron star", "space", "galaxy"],
    "quasar":        ["galaxy", "deep space", "cosmos"],
    "wormhole":      ["space", "galaxy", "cosmos"],
}
# Generic Pexels fallback chain if nothing matches above
PEXELS_DEFAULT_FALLBACKS = ["deep space", "galaxy", "stars", "cosmos", "space"]


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
    category: str = "space"  # one of: finance, ai_tech, dark_psychology, dark_history, nature_horror, space


@dataclass(frozen=True)
class SrtCue:
    start: float
    end: float
    text: str


class PipelineError(RuntimeError):
    pass


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


def main() -> int:
    config = build_config()
    try:
        brief = generate_content_brief(config)
        if brief is None:
            return 0

        audio_path, srt_path = generate_voice_and_captions(config, brief.script)
        if audio_path is None or srt_path is None:
            return 0

        background_path = download_background_video(config, brief.search_query)
        if background_path is None:
            return 0

        final_video_path = assemble_video(config, background_path, audio_path, srt_path)
        if final_video_path is None:
            return 0

        upload_ok = upload_to_youtube(config, final_video_path, brief.title, brief.description)
        return 0 if upload_ok else 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled pipeline error: %s", exc)
        return 0


def _call_gemini(api_key: str, model_name: str, prompt: str, temperature: float) -> str:
    """Helper to route Gemini requests through either the new SDK or the old deprecated SDK."""
    if USE_NEW_GEMINI_SDK:
        client = genai.Client(api_key=api_key)
        safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95 if temperature == 1.0 else 0.9,
                max_output_tokens=2048,  # Increased from 512 to prevent truncated JSON
                response_mime_type="application/json",
                safety_settings=safety_settings
            )
        )
        return response.text or ""
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95 if temperature == 1.0 else 0.9,
                "max_output_tokens": 2048,  # Increased from 512 to prevent truncated JSON
                "response_mime_type": "application/json",
            },
        )
        return _response_text(response)


# ─────────────────────────────────────────────────────────────────────────────
# Content category definitions (research-backed niche selection)
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (category_id, weight, description, video_search_terms, hashtags)
# Weights are proportional probabilities; higher = more frequent.
# Finance/AI get higher weight for RPM; psychology/history for virality.
_CONTENT_CATEGORIES: list[tuple[str, int, str, list[str], str]] = [
    (
        "finance",
        28,  # highest RPM ($10-25), evergreen
        "personal finance, wealth psychology, banking secrets, investing mindset",
        ["stock market", "money", "trading", "bank", "investment", "financial"],
        "#shorts #finance #money #wealth #investing #facts #mindblowing #moneytips",
    ),
    (
        "ai_tech",
        25,  # high RPM ($10-20) + very viral
        "AI surveillance, algorithmic manipulation, tech company secrets, machine learning dangers",
        ["artificial intelligence", "server room", "coding", "technology", "digital", "cyber"],
        "#shorts #ai #tech #technology #artificialintelligence #facts #scary #future",
    ),
    (
        "dark_psychology",
        20,  # very viral, decent RPM
        "dark psychology, manipulation tactics, cognitive biases exploited by corporations, persuasion science",
        ["shadow", "silhouette", "crowd", "person thinking", "abstract mind", "psychology"],
        "#shorts #psychology #darkpsychology #mindcontrol #facts #manipulation #mindblown",
    ),
    (
        "dark_history",
        15,  # very viral, moderate RPM
        "forgotten atrocities, government cover-ups, disturbing historical experiments, suppressed history",
        ["abandoned", "ruins", "vintage", "old building", "historical", "archive"],
        "#shorts #history #darkhistory #facts #scary #conspiracy #mindblown #educational",
    ),
    (
        "nature_horror",
        12,  # viral, moderate RPM
        "deep sea creatures, parasites, extreme predator behavior, disturbing animal biology",
        ["ocean", "underwater", "deep sea", "wildlife", "jungle", "nature"],
        "#shorts #nature #science #deepocean #scary #wildlife #facts #mindblowing",
    ),
]


def _pick_category() -> tuple[str, str, list[str], str]:
    """Weighted-random category selection.
    Returns (category_id, description, video_search_terms, hashtags).
    """
    ids     = [c[0] for c in _CONTENT_CATEGORIES]
    weights = [c[1] for c in _CONTENT_CATEGORIES]
    descs   = [c[2] for c in _CONTENT_CATEGORIES]
    terms   = [c[3] for c in _CONTENT_CATEGORIES]
    tags    = [c[4] for c in _CONTENT_CATEGORIES]
    chosen  = random.choices(range(len(ids)), weights=weights, k=1)[0]
    return ids[chosen], descs[chosen], terms[chosen], tags[chosen]


def _build_prompt(category_id: str, category_desc: str, hashtags: str) -> str:
    """Build a category-specific Gemini prompt using the research-backed viral formula:

    4-Beat structure (proven for 70%+ completion rate):
      1. HOOK   — Contradicts common belief or states shocking fact in <1.5s of audio
      2. CONTEXT — Why you should care (no fluff, 1-2 sentences)
      3. PAYOFF  — The disturbing/surprising depth of the fact
      4. LOOP    — Final sentence loops back to the opening for replay bait

    Target: 55-70 words = ~22-28 seconds at 150 WPM (the proven Shorts sweet spot).
    """
    category_instructions = {
        "finance": (
            "Topic: Pick ONE shocking personal finance or wealth psychology fact. "
            "Examples: how banks profit from your savings account, a cognitive bias that keeps people poor, "
            "a tax trick only the wealthy know, how inflation secretly transfers wealth upward, "
            "why the stock market is rigged against retail investors, or how compound interest works against debt holders. "
            "Hook style: Start with 'The bank is...' or 'Right now...' or 'Every time you...' — immediate, personal, alarming. "
            "Punchline: End with a cynical observation about who benefits from the system. "
            "Loop: Last sentence should echo or contradict the first sentence so it loops naturally."
        ),
        "ai_tech": (
            "Topic: Pick ONE genuinely disturbing AI or tech fact. "
            "Examples: how recommendation algorithms detect depression before you do, "
            "how your phone microphone data is sold, how AI can predict your vote, "
            "how facial recognition is being used without your consent, "
            "how large language models encode bias, or how social media maximizes addiction not connection. "
            "Hook style: Start with 'Right now, AI...' or 'Your phone already knows...' or 'The algorithm...' "
            "Punchline: End with what this means for human autonomy or privacy. "
            "Loop: Last sentence should mirror the opening to encourage replays."
        ),
        "dark_psychology": (
            "Topic: Pick ONE specific, real psychological manipulation technique. "
            "Examples: the foot-in-the-door technique used by subscription services, "
            "how casinos use variable reward schedules to create addiction, "
            "how dark patterns in app design exploit loss aversion, "
            "how social proof is manufactured, or how anchoring is used in pricing. "
            "Hook style: Start with 'This trick is being used on you right now.' or 'You have already fallen for this today.' "
            "Punchline: Name the industry or company exploiting this technique. "
            "Loop: End with a line that makes the viewer want to watch again to catch what they missed."
        ),
        "dark_history": (
            "Topic: Pick ONE genuinely disturbing, lesser-known historical fact. "
            "Examples: a forgotten atrocity, a government experiment on citizens, "
            "a covered-up disaster, a historical figure's hidden crimes, "
            "or a suppressed invention/discovery. "
            "Hook style: Start with 'This actually happened.' or 'In [year]...' or 'A government once...' "
            "Punchline: Connect it to something that still affects us today, or note it was never taught in school. "
            "Loop: End with a question or statement that mirrors the opening."
        ),
        "nature_horror": (
            "Topic: Pick ONE genuinely disturbing fact about a real animal, parasite, or biological phenomenon. "
            "Examples: a parasite that controls its host's brain, a deep sea creature that hunts with bioluminescence, "
            "an animal that digests its prey alive, how cordyceps fungi work, "
            "or the extreme conditions life can survive. "
            "Hook style: Start with the disturbing fact immediately, no preamble. Deadpan tone. "
            "Punchline: End with a twist about how this relates to human biology or everyday life. "
            "Loop: Final sentence creates urgency to rewatch."
        ),
    }

    cat_instruction = category_instructions.get(category_id, category_instructions["dark_psychology"])

    return (
        "OUTPUT ONLY A SINGLE RAW JSON OBJECT. "
        "DO NOT wrap in markdown code fences. DO NOT add any text before or after the JSON. "
        "The very first character MUST be '{' and the very last MUST be '}'. "
        "Use exactly these four string keys: title, description, script, search_query.\n\n"

        f"CATEGORY: {category_desc}\n\n"

        "FIELD RULES:\n"
        "- title: Under 70 chars. Starts with a hook question or alarming statement. "
        "Must create a curiosity gap. Include '#shorts' at the end.\n"
        f"- description: One punchy sarcastic sentence, then on a new line: {hashtags}\n"
        "- script: EXACTLY 55-70 words of spoken narration using the 4-BEAT VIRAL FORMULA:\n"
        "  BEAT 1 (Hook, ~10 words): Contradicts common belief OR states shocking fact. MUST be "
        "completable in under 1.5 seconds of audio. No 'Hey guys' or slow intros.\n"
        "  BEAT 2 (Context, ~15 words): Why this matters to the viewer personally. One sentence.\n"
        "  BEAT 3 (Payoff, ~30 words): The full disturbing depth. Short punchy sentences. Deadpan tone.\n"
        "  BEAT 4 (Loop, ~10 words): Final sentence mirrors or contradicts Beat 1 to encourage replays.\n"
        "  NO filler words. NO 'And so...' or 'In conclusion'. Deliver like a documentarian who "
        "has completely given up on humanity.\n"
        "- search_query: 1-2 English words describing a VISUAL that exists in stock video libraries. "
        "Use concrete, filmable nouns (e.g. 'money', 'server room', 'ocean', 'crowd', 'ruins'). "
        "NOT abstract concepts like 'freedom' or 'fear'.\n\n"

        f"SPECIFIC INSTRUCTIONS:\n{cat_instruction}\n\n"

        "EXAMPLE OUTPUT (exact JSON structure required):\n"
        '{"title": "Your Savings Account Is a Lie #shorts", '
        '"description": "The bank thanks you for your generous donation.\\n'
        '#shorts #finance #money #wealth #facts #banking #mindblown #moneytips", '
        '"script": "Your savings account is not saving you. The average savings rate is 0.4 percent. '
        'Inflation runs at 3 percent. Every year you leave money in that account, you are paying the bank '
        'to hold it. They lend it out at 20 percent interest and give you 0.4. '
        'Your savings account is not saving you.", '
        '"search_query": "money"}'
    )


def generate_content_brief(config: PipelineConfig) -> ContentBrief | None:
    if not config.gemini_api_key:
        logger.error("GEMINI_API_KEY is missing. Skipping content generation.")
        return None

    category_id, category_desc, video_terms, hashtags = _pick_category()
    logger.info("Selected content category: %s", category_id)

    try:
        prompt = _build_prompt(category_id, category_desc, hashtags)

        # De-duplicate candidate models to prevent burning API quota on retries
        base_models = [
            config.gemini_model,
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
        candidate_models = []
        for m in base_models:
            if m and m not in candidate_models:
                candidate_models.append(m)

        last_exc: Exception | None = None
        for candidate in candidate_models:
            try:
                logger.info("Attempting Gemini model: %s", candidate)

                raw_text = _call_gemini(config.gemini_api_key, candidate, prompt, temperature=1.0)
                brief = _parse_brief_json(raw_text, category_id)
                if brief is not None:
                    return brief

                logger.warning("Gemini response from model %s was not valid JSON.", candidate)
                time.sleep(2)

                # Retry once with a stricter instruction
                strict_prompt = prompt + " Output valid JSON only. No leading or trailing text."
                raw_text = _call_gemini(config.gemini_api_key, candidate, strict_prompt, temperature=0.8)
                brief = _parse_brief_json(raw_text, category_id)
                if brief is not None:
                    return brief

                time.sleep(2)

            except Exception as exc:  # try the next model
                logger.warning("Gemini model %s failed: %s", candidate, exc)
                last_exc = exc

        # If we reach here, all model attempts failed or returned invalid JSON.
        if last_exc:
            err_path = config.workspace / "gemini_error.log"
            err_text = f"Last Gemini exception for models {candidate_models}: {last_exc!r}\n"
            try:
                err_path.write_text(err_text, encoding="utf-8")
                logger.info("Wrote Gemini diagnostic to %s", err_path)
            except Exception:
                logger.exception("Failed to write Gemini diagnostic file.")

        # Try a low-cost external LLM (OpenAI gpt-4o-mini) if available.
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                logger.info("Attempting OpenAI gpt-4o-mini as fallback")
                openai_text = _try_openai_prompt(openai_key, prompt)
                brief = _parse_brief_json(openai_text or "", category_id)
                if brief is not None:
                    return brief
                logger.warning("OpenAI response was not valid JSON, falling through to local pool.")
            except Exception as exc:
                logger.warning("OpenAI fallback failed: %s", exc)
    except Exception as exc:
        logger.warning("Gemini generation failed (%s). Falling back to local briefs.", exc)

    # ── Local high-quality fallback pool (15 scripts across 5 categories) ──────
    # These are pre-written using the 4-beat viral formula so even fallback runs
    # produce content that follows the algorithm-optimised structure.
    pool = [
        # FINANCE
        ContentBrief(
            title="Your Savings Account Is Losing You Money #shorts",
            description="The bank thanks you for your generous donation.\n#shorts #finance #money #wealth #investing #facts #mindblowing #moneytips",
            script=(
                "Your savings account is not saving you. "
                "Inflation runs at around three percent. The average savings rate is zero point four. "
                "Every year you leave money in that account you lose two and a half percent of its value. "
                "The bank lends your money out at twenty percent and gives you back zero point four. "
                "Your savings account is not saving you."
            ),
            search_query="money",
            category="finance",
        ),
        ContentBrief(
            title="The Rich Get Richer Because of This One Law #shorts",
            description="Compound interest: the eighth wonder of the world, unless you are on the wrong side of it.\n#shorts #finance #money #wealth #investing #facts #mindblowing #moneytips",
            script=(
                "The tax system is designed to reward people who already have money. "
                "Wages are taxed at up to thirty seven percent. "
                "Capital gains from investments are taxed at twenty percent. "
                "The more wealth you hold in assets instead of wages, the less tax you pay. "
                "The system is not broken. It is working exactly as designed."
            ),
            search_query="stock market",
            category="finance",
        ),
        ContentBrief(
            title="Banks Create Money From Nothing. Legally. #shorts",
            description="Fractional reserve banking is the polite name for it.\n#shorts #finance #money #wealth #banking #facts #mindblown #moneytips",
            script=(
                "Banks do not lend out money they have. "
                "When you take out a loan the bank types a number into a computer and that number becomes real money. "
                "They charge you interest on money that did not exist before you asked for it. "
                "This is legal. It is called fractional reserve banking. "
                "Banks do not lend out money they have."
            ),
            search_query="bank",
            category="finance",
        ),
        # AI / TECH
        ContentBrief(
            title="Your Phone Knows You Are Depressed Before You Do #shorts",
            description="The algorithm knows your emotional state better than your therapist.\n#shorts #ai #tech #technology #artificialintelligence #facts #scary #future",
            script=(
                "AI can detect depression from your phone usage before you notice symptoms. "
                "It tracks typing speed, scroll patterns, time of messages, and app usage duration. "
                "A study found it predicted depressive episodes two weeks in advance with eighty percent accuracy. "
                "Your phone has been watching your mental health. "
                "Whether anyone is using that data responsibly is a different question entirely."
            ),
            search_query="artificial intelligence",
            category="ai_tech",
        ),
        ContentBrief(
            title="The Algorithm Chose Your Political Views #shorts",
            description="You thought you formed your own opinions. Adorable.\n#shorts #ai #tech #socialmedia #algorithm #facts #mindblown #future",
            script=(
                "Your political views were partially shaped by an algorithm. "
                "Social media platforms optimise for engagement. "
                "Outrage generates more engagement than nuance. "
                "So the algorithm promotes outrage. Consistently. At scale. To billions of people. "
                "The most radicalising media system ever built runs on the same servers as your cat photos. "
                "Your political views were partially shaped by an algorithm."
            ),
            search_query="server room",
            category="ai_tech",
        ),
        ContentBrief(
            title="Your Phone Is Listening. Here Is the Proof. #shorts",
            description="Coincidence is just surveillance you cannot prove yet.\n#shorts #ai #tech #privacy #surveillance #facts #scary #technology",
            script=(
                "Apps do not need to listen to your microphone to know what you talked about. "
                "They track your location, your friends locations, your browsing, and your purchase history. "
                "Cross-referencing those data points is more accurate than listening. "
                "The ad you saw after that conversation was not a coincidence. "
                "It was a prediction. A correct one."
            ),
            search_query="technology",
            category="ai_tech",
        ),
        # DARK PSYCHOLOGY
        ContentBrief(
            title="Casinos Use This Trick to Make You Lose More #shorts",
            description="There are no windows or clocks in a casino. That is not an accident.\n#shorts #psychology #darkpsychology #mindcontrol #facts #manipulation #mindblown",
            script=(
                "You have already fallen for this today. "
                "Variable reward schedules are the most addictive behavioral pattern known to science. "
                "Slot machines use them. So do social media feeds. So do dating apps. "
                "Every notification might be a reward. Your brain cannot stop checking. "
                "B.F. Skinner discovered this with pigeons in nineteen thirty eight. "
                "You are still falling for it today."
            ),
            search_query="crowd",
            category="dark_psychology",
        ),
        ContentBrief(
            title="This Psychological Trick Runs Your Entire Life #shorts",
            description="Loss aversion: the reason you make worse decisions when it matters most.\n#shorts #psychology #darkpsychology #mindcontrol #facts #manipulation #cognitive",
            script=(
                "Losing twenty dollars feels twice as bad as finding twenty dollars feels good. "
                "This is called loss aversion and every major company exploits it. "
                "Free trials end with cancellation friction. Sales create fake scarcity. "
                "Insurance sells fear of loss not value of protection. "
                "Your brain evolved this bias to survive predators. "
                "Now corporations use it to sell you streaming services."
            ),
            search_query="silhouette",
            category="dark_psychology",
        ),
        ContentBrief(
            title="Supermarkets Are Engineered to Manipulate You #shorts",
            description="You never just needed milk. You were steered.\n#shorts #psychology #darkpsychology #manipulation #facts #mindblown #shoppinghacks",
            script=(
                "The grocery store layout is a psychological trap. "
                "Milk is always at the back so you walk past everything else first. "
                "Eye-level shelves cost brands sixty percent more to occupy. "
                "The bakery is near the entrance because fresh bread smell triggers hunger and impulsive buying. "
                "You thought you made those choices. "
                "The store made them weeks before you arrived."
            ),
            search_query="shopping",
            category="dark_psychology",
        ),
        # DARK HISTORY
        ContentBrief(
            title="The US Government Did This to Its Own Citizens #shorts",
            description="MKUltra was declassified. Most of it is still redacted.\n#shorts #history #darkhistory #facts #scary #conspiracy #mindblown #educational",
            script=(
                "The United States government ran a mind control program on its own citizens without consent. "
                "Operation MKUltra dosed prisoners, mental patients, and soldiers with LSD to test behavioral control. "
                "It ran for twenty years. It was only exposed because of a filing error. "
                "Most of the files were ordered destroyed in nineteen seventy three. "
                "The program was real. The full extent remains classified."
            ),
            search_query="abandoned",
            category="dark_history",
        ),
        ContentBrief(
            title="The Great Molasses Flood Was Real and Deadly #shorts",
            description="Boston 1919: when capitalism moved too fast and the molasses moved faster.\n#shorts #history #darkhistory #facts #weird #mindblown #educational",
            script=(
                "In nineteen nineteen a tank of two million gallons of molasses exploded in Boston. "
                "The wave was fifteen feet high and moved at thirty five miles per hour. "
                "It killed twenty one people and injured one hundred and fifty. "
                "The company had ignored structural warnings for years because repairs cost money. "
                "Buildings smelled of molasses in summer for decades. "
                "This actually happened."
            ),
            search_query="vintage",
            category="dark_history",
        ),
        # NATURE HORROR
        ContentBrief(
            title="This Parasite Controls Your Brain Right Now #shorts",
            description="Toxoplasma gondii: the mind control parasite in a third of all humans.\n#shorts #nature #science #biology #scary #wildlife #facts #mindblowing",
            script=(
                "A parasite called Toxoplasma gondii has infected roughly a third of all humans alive. "
                "In rats it removes the fear of cats so the rat gets eaten and the parasite reaches its target host. "
                "Studies suggest infected humans show measurably different risk tolerance and reaction times. "
                "You likely got it from undercooked meat or a cat litter box. "
                "There is no cure."
            ),
            search_query="wildlife",
            category="nature_horror",
        ),
        ContentBrief(
            title="The Deep Sea Is Actively Trying to Kill You #shorts",
            description="Sixty percent of Earth's surface is deeper than two hundred meters and we have barely looked.\n#shorts #nature #deepocean #science #scary #biology #facts #mindblowing",
            script=(
                "Below two hundred meters sunlight stops. Pressure crushes. Predators hunt without eyes. "
                "The anglerfish lures prey with its own bioluminescent light then swallows creatures larger than itself. "
                "The Greenland shark lives for five hundred years, moves slowly, and eats everything. "
                "We have explored less than twenty percent of the ocean floor. "
                "We have no idea what is in the rest."
            ),
            search_query="ocean",
            category="nature_horror",
        ),
        ContentBrief(
            title="Cordyceps Fungi Are Real and They Control Ants #shorts",
            description="The Last of Us was a documentary. Scientifically speaking.\n#shorts #nature #science #biology #fungi #scary #wildlife #facts",
            script=(
                "Cordyceps fungi infect ants, hijack their nervous systems, and force them to climb to the ideal height. "
                "Then the fungus kills the ant and bursts spores from its head onto the colony below. "
                "The ant walks to its own death believing it is making a choice. "
                "There are over six hundred species of cordyceps. "
                "None of them target humans. Yet."
            ),
            search_query="jungle",
            category="nature_horror",
        ),
    ]
    return random.choice(pool)


def generate_voice_and_captions(config: PipelineConfig, script: str) -> tuple[Path, Path] | tuple[None, None]:
    # Target 30 seconds — the research-backed sweet spot for Shorts completion rate.
    # 150 WPM × 0.5 min = 75 words maximum before padding is needed.
    script = _stretch_script_to_target(script, target_seconds=30, wpm=150)
    try:
        asyncio.run(_generate_audio_and_srt(config.voice, script, config.output_audio, config.output_srt))
        return config.output_audio, config.output_srt
    except Exception as exc:  # noqa: BLE001
        logger.error("Voice generation failed: %s", exc)
        return None, None


def _stretch_script_to_target(script: str, *, target_seconds: float = 30.0, wpm: int = 150) -> str:
    if not script or not isinstance(script, str):
        return script
    words = script.split()
    target_words = int(wpm * (target_seconds / 60.0))
    if len(words) >= target_words:
        return script

    fillers = [
        "Also, here's a tiny twist you didn't expect.",
        "Strangely, that actually makes it worse — and funnier.",
        "Which is to say: history has a dark sense of humor.",
        "And yes, that detail makes the whole story deliciously awkward.",
        "Quick aside: don't try this at home unless you enjoy surprises.",
    ]
    i = 0
    while len(words) < target_words:
        words.extend(fillers[i % len(fillers)].split())
        i += 1
    return " ".join(words[: max(len(words), target_words)])


async def _generate_audio_and_srt(voice: str, script: str, audio_path: Path, srt_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    communicate = edge_tts.Communicate(script, voice, boundary="WordBoundary")
    submaker = edge_tts.SubMaker()

    with audio_path.open("wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                submaker.feed(chunk)

    srt_path.write_text(submaker.get_srt(), encoding="utf-8")


def _download_nasa_video(search_query: str, output_path: Path) -> bool:
    """Search the NASA Image and Video Library and download the best matching video.

    This API is completely free, requires no API key, and returns actual NASA
    footage of space phenomena (nebulae, black holes, galaxies, etc.).
    Returns True if a video was successfully downloaded, False otherwise.
    """
    try:
        resp = requests.get(
            NASA_VIDEO_SEARCH_URL,
            params={"q": search_query, "media_type": "video"},
            timeout=20,
        )
        resp.raise_for_status()
        items = resp.json().get("collection", {}).get("items", [])
        if not items:
            logger.info("NASA search returned 0 results for '%s'.", search_query)
            return False

        # Try up to 5 items before giving up
        for item in items[:5]:
            asset_manifest_url = item.get("href", "")
            if not asset_manifest_url:
                continue
            try:
                manifest_resp = requests.get(asset_manifest_url, timeout=15)
                manifest_resp.raise_for_status()
                asset_urls: list[str] = manifest_resp.json()

                # Prefer mobile/medium MP4 — large NASA originals can be multi-GB
                mp4_urls = [u for u in asset_urls if u.lower().endswith(".mp4")]
                if not mp4_urls:
                    continue

                # Tier preference: mobile < small < medium < large < orig
                # We want medium/small to balance quality vs download time on CI
                def _tier_score(url: str) -> int:
                    u = url.lower()
                    if "~mobile" in u:  return 5
                    if "~small"  in u:  return 4
                    if "~medium" in u:  return 3
                    if "~large"  in u:  return 2
                    if "~orig"   in u:  return 1
                    return 0

                mp4_urls.sort(key=_tier_score, reverse=True)
                download_url = mp4_urls[0]

                logger.info("Downloading NASA video for query '%s': %s", search_query, download_url)
                with requests.get(download_url, stream=True, timeout=180) as vr:
                    vr.raise_for_status()
                    with output_path.open("wb") as f:
                        for chunk in vr.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                logger.info("NASA video downloaded: %s", output_path)
                return True

            except Exception as exc:
                logger.debug("NASA asset '%s' failed: %s", asset_manifest_url, exc)
                continue

        logger.info("No downloadable NASA video found for '%s'.", search_query)
        return False

    except Exception as exc:  # noqa: BLE001
        logger.warning("NASA video search failed for '%s': %s", search_query, exc)
        return False


def _pexels_query_chain(search_query: str) -> list[str]:
    """Build an ordered list of queries to try on Pexels for a given search_query.

    The first entry is the exact query from Gemini.  Subsequent entries are
    progressively broader fallbacks so that even niche terms like 'magnetar'
    or 'vacuum decay' eventually resolve to footage that exists on Pexels.
    """
    q_lower = search_query.lower()
    # Find matching fallback chain from the keyword table
    for keyword, fallbacks in PEXELS_QUERY_FALLBACKS.items():
        if keyword in q_lower:
            # Start with the original query, then the specific fallbacks
            chain = [search_query] + fallbacks
            # Always end with the generic fallbacks
            for f in PEXELS_DEFAULT_FALLBACKS:
                if f not in chain:
                    chain.append(f)
            return chain
    # No specific match — just use original + generic chain
    chain = [search_query] + [
        f for f in PEXELS_DEFAULT_FALLBACKS if f != search_query
    ]
    return chain


def _pexels_download_for_query(
    query: str,
    api_key: str,
    output_path: Path,
) -> bool:
    """Try downloading a portrait video from Pexels for *query*.
    Returns True on success.
    """
    try:
        headers = {"Authorization": api_key}
        resp = requests.get(
            PEXELS_VIDEO_SEARCH_URL,
            headers=headers,
            params={"query": query, "orientation": "portrait", "per_page": 15},
            timeout=30,
        )
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
        if not videos:
            logger.info("Pexels returned 0 results for '%s'.", query)
            return False

        # Try candidates in order; the best portrait clip scores highest
        for video in videos[:5]:
            video_url = _best_mp4_link(video.get("video_files", []))
            if not video_url:
                continue
            logger.info("Downloading Pexels video for query '%s'.", query)
            try:
                with requests.get(video_url, stream=True, timeout=180) as vr:
                    vr.raise_for_status()
                    with output_path.open("wb") as f:
                        for chunk in vr.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                return True
            except Exception as exc:
                logger.debug("Pexels download for '%s' failed: %s", video_url, exc)
                continue

        return False
    except Exception as exc:  # noqa: BLE001
        logger.debug("Pexels search for '%s' failed: %s", query, exc)
        return False


def download_background_video(config: PipelineConfig, search_query: str) -> Path | None:
    """Download the best available background video for *search_query*.

    Strategy (in order):
    1. NASA Image and Video Library — free, no key, actual space/science footage
       that directly matches the script topic (black holes, nebulae, etc.).
    2. Pexels — with a query-broadening fallback chain so niche terms like
       'magnetar' eventually resolve to footage that exists in the library.

    Returns the path to the downloaded file, or None on total failure.
    """
    # ── Step 1: NASA (primary, no key required) ───────────────────────────────
    if _download_nasa_video(search_query, config.output_background):
        return config.output_background

    logger.info("NASA video unavailable; falling back to Pexels.")

    # ── Step 2: Pexels with fallback query chain ──────────────────────────────
    if not config.pexels_api_key:
        logger.error("PEXELS_API_KEY is missing and NASA also failed. Cannot get background video.")
        return None

    for query in _pexels_query_chain(search_query):
        if _pexels_download_for_query(query, config.pexels_api_key, config.output_background):
            return config.output_background
        logger.info("Pexels query '%s' produced no usable video; trying next fallback.", query)

    logger.error("All video sources exhausted for query '%s'.", search_query)
    return None


def assemble_video(config: PipelineConfig, background_path: Path, audio_path: Path, srt_path: Path) -> Path | None:
    try:
        audio_clip = AudioFileClip(str(audio_path))
        background_clip = VideoFileClip(str(background_path))
        subtitle_cues = _parse_srt_file(srt_path)
        subtitle_clips = [_subtitle_clip_for_cue(cue) for cue in subtitle_cues]

        # Ensure a minimum runtime for the short. If TTS output is very short,
        # pad with silence so the final video is at least 30 seconds long
        # (research-backed sweet spot: 15-35s for maximum Shorts completion rate).
        min_duration = 30.0
        if audio_clip.duration < min_duration:
            silence_duration = min_duration - audio_clip.duration
            fps = getattr(audio_clip, "fps", 44100)
            nchannels = getattr(audio_clip, "nchannels", 1)
            n_samples = int(silence_duration * fps)
            if nchannels == 1:
                arr = np.zeros((n_samples, 1), dtype=float)
            else:
                arr = np.zeros((n_samples, nchannels), dtype=float)
            silence_clip = AudioArrayClip(arr, fps)
            composite_audio = CompositeAudioClip([
                audio_clip.set_start(0),
                silence_clip.set_start(audio_clip.duration),
            ])
            audio_clip = composite_audio

        try:
            target_duration = max(min_duration, float(audio_clip.duration))
            scale = max(TARGET_WIDTH / background_clip.w, TARGET_HEIGHT / background_clip.h)
            background = background_clip.resize(scale)
            background = background.fx(
                vfx.crop,
                width=TARGET_WIDTH,
                height=TARGET_HEIGHT,
                x_center=background.w / 2,
                y_center=background.h / 2,
            )
            background = background.fx(vfx.loop, duration=target_duration).subclip(0, target_duration)
            background = background.set_audio(audio_clip)
            composite = CompositeVideoClip([background, *subtitle_clips], size=(TARGET_WIDTH, TARGET_HEIGHT))
            composite = composite.set_duration(target_duration).set_audio(audio_clip)
            config.output_final.parent.mkdir(parents=True, exist_ok=True)
            composite.write_videofile(
                str(config.output_final),
                fps=30,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="medium",
                verbose=False,
                logger=None,
            )
        finally:
            for clip in subtitle_clips:
                clip.close()
            background_clip.close()
            audio_clip.close()
        logger.info("Rendered final video: %s", config.output_final)
        return config.output_final
    except Exception as exc:  # noqa: BLE001
        logger.error("Video assembly failed: %s", exc)
        return None


def upload_to_youtube(config: PipelineConfig, video_path: Path, title: str, description: str) -> bool:
    """Upload *video_path* to YouTube. Returns True on success, False on any failure."""
    if not config.youtube_credentials.exists():
        logger.error("YouTube credentials file is missing at %s. Skipping upload.", config.youtube_credentials)
        return False

    try:
        credentials = Credentials.from_authorized_user_file(str(config.youtube_credentials), scopes=[YOUTUBE_UPLOAD_SCOPE])

        if credentials.expired and credentials.refresh_token:
            logger.info("YouTube token expired. Refreshing token...")
            try:
                credentials.refresh(Request())
            except Exception as refresh_exc:  # noqa: BLE001
                err_str = str(refresh_exc)
                # `invalid_grant` means the refresh token has been permanently revoked by Google.
                # This is NOT a transient error — it cannot be fixed by retrying.
                # Common causes:
                #   1. OAuth consent screen is in "Testing" status → tokens expire after 7 days.
                #      Fix: publish the app to "Production" in Google Cloud Console, then
                #           re-run get_credentials.py and update the YOUTUBE_CREDENTIALS_JSON secret.
                #   2. Token has not been used for > 6 months (Google sliding-window policy).
                #   3. More than 50 tokens issued for the same OAuth client (oldest are auto-revoked).
                if "invalid_grant" in err_str:
                    logger.error(
                        "YouTube OAuth refresh_token has been permanently revoked (invalid_grant). "
                        "This is a fatal, non-retryable error. "
                        "ACTION REQUIRED: Re-run get_credentials.py locally to obtain a fresh refresh_token, "
                        "then update the YOUTUBE_CREDENTIALS_JSON GitHub secret. "
                        "If your OAuth app is in 'Testing' status, publish it to 'Production' first "
                        "(Google Cloud Console > APIs & Services > OAuth consent screen). "
                        "Raw error: %s",
                        refresh_exc,
                    )
                else:
                    logger.error("YouTube token refresh failed with unexpected error: %s", refresh_exc)
                return False

            new_creds_json = credentials.to_json()
            config.youtube_credentials.write_text(new_creds_json, encoding="utf-8")

            # Write the new token to GITHUB_OUTPUT so the workflow file can capture and save it permanently.
            # NOTE: This block must remain AFTER a successful refresh (not inside the try above)
            # so that invalid_grant failures never reach it.
            github_output = os.environ.get("GITHUB_OUTPUT")
            if github_output:
                with open(github_output, "a", encoding="utf-8") as f:
                    delimiter = "EOF_CREDS"
                    f.write(f"new_youtube_creds<<{delimiter}\n{new_creds_json}\n{delimiter}\n")
                logger.info("Exported updated credentials to GITHUB_OUTPUT for GitHub Actions to capture.")

        youtube = build("youtube", "v3", credentials=credentials)
        body = {
            "snippet": {
                "title": _ensure_shorts_tag(title),
                "description": _ensure_shorts_hashtag(description),
                "categoryId": "22",
                "tags": ["shorts", "comedy", "darkhumor", "history", "facts"],
            },
            "status": {"privacyStatus": config.privacy_status},
        }

        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=MediaFileUpload(str(video_path), chunksize=-1, resumable=True),
        )
        response = None
        retries = 0
        while response is None:
            try:
                _, response = request.next_chunk()
                if response and "id" in response:
                    logger.info("Uploaded YouTube Short with video id: %s", response["id"])
                    return True
            except HttpError as exc:
                if exc.resp.status in {500, 502, 503, 504} and retries < 5:
                    retries += 1
                    sleep_seconds = random.uniform(1.0, 2.0 ** retries)
                    logger.warning("Transient YouTube error %s. Retrying in %.1f seconds.", exc.resp.status, sleep_seconds)
                    time.sleep(sleep_seconds)
                    continue
                raise
        logger.error("YouTube upload completed without a usable response payload.")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("YouTube upload failed: %s", exc)
        return False


def _response_text(response: Any) -> str:
    if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip():
        return response.text
    if hasattr(response, "parts"):
        parts: list[str] = []
        for candidate in getattr(response, "candidates", []):
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []):
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
        if parts:
            return "".join(parts)
    return str(response)


def _try_openai_prompt(api_key: str, prompt: str) -> str | None:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "response_format": { "type": "json_object" },
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 2048,  # Increased to match Gemini limit
        "top_p": 0.95,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Defensive parsing
    if not isinstance(data, dict):
        return None
    choices = data.get("choices") or []
    if not choices or not isinstance(choices, list):
        return None
    message = choices[0].get("message") or {}
    content = message.get("content") or choices[0].get("text")
    return content


def _parse_brief_json(raw_text: str, category_id: str = "space") -> ContentBrief | None:
    """Parse a ContentBrief from Gemini output.

    Gemini (especially the deprecated SDK) sometimes wraps JSON in markdown
    code fences like ```json\n{...}\n```.  It may also emit a flat object
    without outer braces when the response_mime_type hint is ignored.
    This function handles all three cases robustly.
    """
    if not raw_text:
        return None

    try:
        # Step 1: strip markdown code fences (``` or ```json)
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", raw_text.strip(), flags=re.MULTILINE)
        stripped = re.sub(r"```\s*$", "", stripped.strip(), flags=re.MULTILINE).strip()

        # Step 2: find JSON object boundaries
        start = stripped.find('{')
        end = stripped.rfind('}')

        if start != -1 and end != -1 and end >= start:
            # Normal case: JSON object found
            cleaned = stripped[start:end + 1]
        elif start == -1 and end == -1:
            # Degenerate case: Gemini emitted flat key-value pairs without {}
            cleaned = '{' + stripped + '}'
            logger.warning("Gemini returned flat JSON (no braces); wrapped automatically. Preview: %s", repr(raw_text[:120]))
        else:
            logger.warning("No JSON object found in raw text. Raw output was: %s", repr(raw_text[:200]))
            return None

        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("JSON parsing failed: %s. Raw text: %s", exc, raw_text[:300])
        return None

    if not isinstance(payload, dict):
        return None

    title = _normalize_text(payload.get("title", ""), max_length=120)
    description = _normalize_text(payload.get("description", ""), max_length=500)
    script = _normalize_text(payload.get("script", ""), max_length=1500)
    search_query = _normalize_search_query(payload.get("search_query", ""))

    if not title or not description or not script or not search_query:
        logger.warning("Parsed JSON is missing required fields. Payload received: %s", payload)
        return None

    script = _limit_words(script, 90)
    description = _ensure_shorts_hashtag(description)
    return ContentBrief(title=title, description=description, script=script, search_query=search_query, category=category_id)


def _normalize_text(value: Any, *, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = " ".join(value.strip().split())
    return cleaned[:max_length].strip()


def _normalize_search_query(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    words = re.findall(r"[A-Za-z0-9]+", value.lower())
    if not words:
        return ""
    return " ".join(words[:2])


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _ensure_shorts_hashtag(description: str) -> str:
    desc = description.strip()
    required_tags = ["#shorts", "#comedy", "#darkhumor"]
    lower = desc.lower()
    missing = [tag for tag in required_tags if tag not in lower]
    if missing:
        if desc:
            desc = f"{desc}\n\n{' '.join(missing)}"
        else:
            desc = " ".join(missing)
    return desc


def _ensure_shorts_tag(title: str) -> str:
    cleaned = title.strip()
    if "#shorts" not in cleaned.lower():
        cleaned = f"{cleaned} #shorts".strip()
    return cleaned


def _best_mp4_link(video_files: Iterable[dict[str, Any]]) -> str | None:
    """Select the best MP4 link, strongly preferring portrait (tall) orientation
    since we are producing a 9:16 YouTube Short.

    Scoring:
      - Portrait clips (height > width) get a large bonus so they are always
        preferred over landscape clips of the same resolution.
      - Within each orientation tier we pick the highest resolution * bitrate.
    """
    candidates = []
    for item in video_files:
        if not isinstance(item, dict):
            continue
        link = item.get("link")
        if not isinstance(link, str) or not link:
            continue
        file_type = str(item.get("file_type", "")).lower()
        if file_type not in {"video/mp4", "application/mp4", ""} and not link.lower().endswith(".mp4"):
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        bitrate = int(item.get("bitrate") or 0)
        # Heavy bonus for portrait clips — a 1080x1920 clip is ideal; landscape
        # clips at any resolution score lower than any portrait clip.
        portrait_bonus = 10_000_000_000 if height > width else 0
        score = portrait_bonus + (width * height) + bitrate
        candidates.append((score, link))
    if not candidates:
        return None
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    return candidates[0][1]


def _parse_srt_file(path: Path) -> list[SrtCue]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    cues: list[SrtCue] = []
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        timing_line_index = 0 if "-->" in lines[0] else 1 if len(lines) > 1 and "-->" in lines[1] else -1
        if timing_line_index == -1:
            continue
        timing_line = lines[timing_line_index]
        content_lines = lines[timing_line_index + 1 :]
        start_text, end_text = [part.strip() for part in timing_line.split("-->", 1)]
        start_seconds = _srt_timestamp_to_seconds(start_text)
        end_seconds = _srt_timestamp_to_seconds(end_text)
        content = " ".join(content_lines).strip()
        if content:
            cues.append(SrtCue(start=start_seconds, end=end_seconds, text=content))
    return cues


def _srt_timestamp_to_seconds(timestamp: str) -> float:
    hours, minutes, rest = timestamp.split(":")
    seconds, milliseconds = rest.split(",")
    total = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return float(total) + (int(milliseconds) / 1000.0)


# Caption panel dimensions (bottom-third of a 1080x1920 frame)
_CAPTION_PANEL_W = TARGET_WIDTH          # 1080
_CAPTION_PANEL_H = 480                   # tall enough for 3 wrapped lines
_CAPTION_FONT_SIZE = 72
_CAPTION_PADDING_X = 48
_CAPTION_PADDING_Y = 30
_CAPTION_BG_COLOR = (0, 0, 0, 170)       # semi-transparent black pill
_CAPTION_BG_RADIUS = 28                  # pill corner radius
_CAPTION_TEXT_COLOR = (255, 255, 255, 255)
_CAPTION_STROKE_COLOR = (0, 0, 0, 255)
_CAPTION_STROKE_WIDTH = 3

# How far up from the bottom of the full frame the caption panel sits (px)
_CAPTION_BOTTOM_OFFSET = 220


def _subtitle_clip_for_cue(cue: SrtCue) -> ImageClip:
    image_path = _render_caption_image(cue.text)
    clip = ImageClip(str(image_path)).set_start(cue.start).set_duration(max(cue.end - cue.start, 0.2))
    # Pin to bottom-third: y is measured from top of the full 1920px frame
    y_pos = TARGET_HEIGHT - _CAPTION_PANEL_H - _CAPTION_BOTTOM_OFFSET
    return clip.set_position(("center", y_pos))


def _draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int, int],
) -> None:
    """Draw a rounded rectangle (pill) on an RGBA canvas."""
    x0, y0, x1, y1 = xy
    # Four corner circles
    draw.ellipse((x0, y0, x0 + 2 * radius, y0 + 2 * radius), fill=fill)
    draw.ellipse((x1 - 2 * radius, y0, x1, y0 + 2 * radius), fill=fill)
    draw.ellipse((x0, y1 - 2 * radius, x0 + 2 * radius, y1), fill=fill)
    draw.ellipse((x1 - 2 * radius, y1 - 2 * radius, x1, y1), fill=fill)
    # Fill body
    draw.rectangle((x0 + radius, y0, x1 - radius, y1), fill=fill)
    draw.rectangle((x0, y0 + radius, x1, y1 - radius), fill=fill)


def _render_caption_image(text: str) -> Path:
    """Render a single caption cue as a PNG with a TikTok-style pill background.

    Layout (all measurements in px, canvas = 1080 x _CAPTION_PANEL_H):
      - Semi-transparent rounded-rect pill behind the text
      - White text with black stroke for readability on any background
      - Text is horizontally centered, vertically centered within the pill
    """
    temp_dir = Path(__file__).resolve().parent / "_caption_cache"
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"caption_{abs(hash(text))}.png"
    output_path = temp_dir / file_name

    font = _load_caption_font(_CAPTION_FONT_SIZE)

    # -- Measure wrapped text on a temporary canvas ---------------------------
    probe = Image.new("RGBA", (1, 1))
    probe_draw = ImageDraw.Draw(probe)
    wrapped_text = _wrap_text(probe_draw, text, font, max_width=_CAPTION_PANEL_W - 2 * _CAPTION_PADDING_X)
    bbox = probe_draw.multiline_textbbox(
        (0, 0), wrapped_text, font=font, spacing=10, stroke_width=_CAPTION_STROKE_WIDTH
    )
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # -- Pill dimensions ------------------------------------------------------
    pill_w = min(text_w + 2 * _CAPTION_PADDING_X, _CAPTION_PANEL_W - 40)
    pill_h = text_h + 2 * _CAPTION_PADDING_Y
    pill_x0 = (_CAPTION_PANEL_W - pill_w) // 2
    pill_y0 = (_CAPTION_PANEL_H - pill_h) // 2
    pill_x1 = pill_x0 + pill_w
    pill_y1 = pill_y0 + pill_h

    # -- Render ---------------------------------------------------------------
    image = Image.new("RGBA", (_CAPTION_PANEL_W, _CAPTION_PANEL_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Pill background
    _draw_rounded_rect(draw, (pill_x0, pill_y0, pill_x1, pill_y1), _CAPTION_BG_RADIUS, _CAPTION_BG_COLOR)

    # Text position: centered inside pill
    text_x = (_CAPTION_PANEL_W - text_w) / 2
    text_y = pill_y0 + _CAPTION_PADDING_Y

    # Drop shadow (offset +3, +3, slightly transparent)
    draw.multiline_text(
        (text_x + 3, text_y + 3),
        wrapped_text,
        font=font,
        fill=(0, 0, 0, 140),
        spacing=10,
        align="center",
    )
    # Main text with stroke
    draw.multiline_text(
        (text_x, text_y),
        wrapped_text,
        font=font,
        fill=_CAPTION_TEXT_COLOR,
        spacing=10,
        align="center",
        stroke_width=_CAPTION_STROKE_WIDTH,
        stroke_fill=_CAPTION_STROKE_COLOR,
    )

    image.save(output_path)
    return output_path


def _load_caption_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    words = text.split()
    if not words:
        return text

    lines: list[str] = []
    current_line = words[0]
    for word in words[1:]:
        candidate = f"{current_line} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
