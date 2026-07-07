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


def generate_content_brief(config: PipelineConfig) -> ContentBrief | None:
    if not config.gemini_api_key:
        logger.error("GEMINI_API_KEY is missing. Skipping content generation.")
        return None
        
    try:
        # CRITICAL: The prompt is written as a raw JSON instruction so that even the
        # deprecated google-generativeai SDK (which ignores response_mime_type) is
        # forced to output a bare JSON object — no markdown fences, no prose.
        prompt = (
            "OUTPUT ONLY A SINGLE RAW JSON OBJECT. "
            "DO NOT wrap in markdown code fences. DO NOT add any text before or after the JSON. "
            "The very first character of your response MUST be '{' and the very last MUST be '}'. "
            "Use exactly these four string keys: title, description, script, search_query.\n\n"
            "RULES:\n"
            "- title: a punchy, ominous clickbait question under 80 characters. Include '#shorts' at the end.\n"
            "- description: 1 sarcastic sentence + 6-8 relevant hashtags including #shorts #space #facts.\n"
            "- script: EXACTLY 65-75 words of deadpan spoken narration. "
            "Structure: (1) a jaw-dropping space fact stated as fact, "
            "(2) an escalation that makes it worse, "
            "(3) a cynical punchline twist at the very end. "
            "No filler words. Pure deadpan. Short punchy sentences. "
            "Sound like a documentary narrator who has given up on humanity.\n"
            "- search_query: 1-2 English words for stock video search (e.g. 'black hole', 'galaxy', 'nebula', 'cosmos').\n\n"
            "SUBJECT: Pick one genuinely terrifying, mind-bending space fact from this list or invent a new one: "
            "vacuum decay, the Bootes Void, rogue planets, neutron star density, "
            "magnetar magnetic fields, Sagittarius A* tidal forces, heat death of the universe, "
            "the Great Attractor, galactic cannibalism, or cosmic strings. "
            "Keep tone existentially dreadful. No gore, no profanity.\n\n"
            "EXAMPLE OUTPUT (follow this exact structure):\n"
            '{"title": "What If Space Just... Deleted You? #shorts", '
            '"description": "Turns out the universe has no refund policy. #shorts #space #facts #scaryspace #cosmichorror #mindblown #science #universe", '
            '"script": "Somewhere in the universe a magnetar spins sixty times per second. Its magnetic field is so strong it can rearrange the atoms in your body from halfway across the solar system. Not destroy them. Rearrange them. Like a cosmic blender that forgot to ask. The good news is you would not feel it. The bad news is everything else would.", '
            '"search_query": "magnetar"}'
        )

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
                brief = _parse_brief_json(raw_text)
                if brief is not None:
                    return brief
                    
                logger.warning("Gemini response from model %s was not valid JSON.", candidate)
                
                # Small sleep to prevent instantly blowing the 5/min rate limit
                time.sleep(2)
                
                # Retry once with a stricter instruction
                strict_prompt = prompt + " Output valid JSON only. No leading or trailing text."
                raw_text = _call_gemini(config.gemini_api_key, candidate, strict_prompt, temperature=0.8)
                brief = _parse_brief_json(raw_text)
                if brief is not None:
                    return brief
                
                # Sleep again before moving to the next model fallback to save quota
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
                brief = _parse_brief_json(openai_text or "")
                if brief is not None:
                    return brief
                logger.warning("OpenAI response was not valid JSON, falling through to local pool.")
            except Exception as exc:
                logger.warning("OpenAI fallback failed: %s", exc)
    except Exception as exc:
        logger.warning("Gemini generation failed (%s). Falling back to local briefs.", exc)

    # local pool of short, safe briefs
    pool = [
            ContentBrief(
                title="The Moon's Missing Water",
                description="#space #fact #shorts The Moon hides water in cold, shadowed craters — and yes, it's inconveniently chill.",
                script=(
                    "Scientists discovered pockets of ice trapped in permanently shadowed lunar craters. "
                    "It's like the Moon opened a tiny freezer and forgot the key. Imagine astronauts coming back with frozen coffee and a note: 'Do not thaw.' "
                    "The surprising part: the Moon's backyard is more hydrated than some urban houseplants. Weird, but useful."
                ),
                search_query="moon",
            ),
            ContentBrief(
                title="The Roman Concrete Secret",
                description="#history #fact #shorts Romans mixed seawater into concrete and somehow beat time at its own game.",
                script=(
                    "Roman builders used seawater chemistry to make concrete that actually got stronger over centuries. "
                    "So while our modern buildings try their best, ancient Romans basically invented forever-pavement. "
                    "Imagine a contractor pitching: 'Give me three centuries and I’ll make this sidewalk legendary.' It's a neat reminder: old tricks sometimes outlast new trends."
                ),
                search_query="ancient ruins",
            ),
            ContentBrief(
                title="A Simple Mind Trick",
                description="#psychology #hack #shorts A tiny posture trick that fools people into thinking you own the room.",
                script=(
                    "Want to seem more confident instantly? Stand tall, breathe deep, and pretend you have absolutely no idea you're being watched. "
                    "People assume confident people always mean business — and the trick is they rarely check receipts. "
                    "Use it before presentations or awkward elevator conversations. It's not magic, just convincing acting."
                ),
                search_query="person",
            ),
        ]
    return random.choice(pool)


def generate_voice_and_captions(config: PipelineConfig, script: str) -> tuple[Path, Path] | tuple[None, None]:
    # Ensure script is long enough for target duration by appending short
    # humorous filler lines when necessary.
    script = _stretch_script_to_target(script, target_seconds=40, wpm=150)
    try:
        asyncio.run(_generate_audio_and_srt(config.voice, script, config.output_audio, config.output_srt))
        return config.output_audio, config.output_srt
    except Exception as exc:  # noqa: BLE001
        logger.error("Voice generation failed: %s", exc)
        return None, None


def _stretch_script_to_target(script: str, *, target_seconds: float = 40.0, wpm: int = 150) -> str:
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


def download_background_video(config: PipelineConfig, search_query: str) -> Path | None:
    if not config.pexels_api_key:
        logger.error("PEXELS_API_KEY is missing. Skipping background video download.")
        return None

    try:
        headers = {"Authorization": config.pexels_api_key}
        response = requests.get(
            PEXELS_VIDEO_SEARCH_URL,
            headers=headers,
            params={"query": search_query, "orientation": "portrait", "per_page": 10},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        videos = payload.get("videos", [])
        if not videos:
            raise PipelineError(f"No Pexels video results for query: {search_query}")

        selected_video = videos[0]
        video_url = _best_mp4_link(selected_video.get("video_files", []))
        if not video_url:
            raise PipelineError("No MP4 video file found in the first Pexels result.")

        logger.info("Downloading background video for query '%s'.", search_query)
        with requests.get(video_url, stream=True, timeout=120) as video_response:
            video_response.raise_for_status()
            with config.output_background.open("wb") as output_file:
                for chunk in video_response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output_file.write(chunk)
        return config.output_background
    except Exception as exc:  # noqa: BLE001
        logger.error("Background video download failed: %s", exc)
        return None


def assemble_video(config: PipelineConfig, background_path: Path, audio_path: Path, srt_path: Path) -> Path | None:
    try:
        audio_clip = AudioFileClip(str(audio_path))
        background_clip = VideoFileClip(str(background_path))
        subtitle_cues = _parse_srt_file(srt_path)
        subtitle_clips = [_subtitle_clip_for_cue(cue) for cue in subtitle_cues]

        # Ensure a minimum runtime for the short. If TTS output is very short,
        # pad with silence so the final video is at least 40 seconds long.
        min_duration = 40.0
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


def _parse_brief_json(raw_text: str) -> ContentBrief | None:
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
            # e.g. just:  "title": "...",\n  "description": "..."\n  ...
            # Wrap them and try to parse
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
    return ContentBrief(title=title, description=description, script=script, search_query=search_query)


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
