import asyncio
import re
from pathlib import Path
import edge_tts

from config import PipelineConfig, SrtCue, logger

def generate_voice_and_captions(config: PipelineConfig, script: str) -> tuple[Path, Path] | tuple[None, None]:
    # Target 30 seconds — the sweet spot for Shorts completion rate.
    # ~150 WPM at +18% rate lands the storytelling pace of the reference video.
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

    # Conversational storytelling fillers — extend the anecdote naturally
    # without breaking the "talking to a friend" voice.
    fillers = [
        "And the worst part? He was completely serious.",
        "I couldn't even look him in the eye.",
        "No remorse. Zero. None whatsoever.",
        "Which is honestly the most on-brand thing ever.",
        "And I say this with love.",
        "This is the same person, by the way.",
        "I don't know what I expected.",
    ]
    i = 0
    while len(words) < target_words:
        words.extend(fillers[i % len(fillers)].split())
        i += 1
    return " ".join(words[: max(len(words), target_words)])


async def _generate_audio_and_srt(voice: str, script: str, audio_path: Path, srt_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract [PAUSE] BEFORE bracket stripping — otherwise it gets destroyed.
    # We split on the raw marker first, clean each part, then reassemble.
    has_pause = "[PAUSE]" in script
    if has_pause:
        raw_parts = script.split("[PAUSE]", 1)
    else:
        raw_parts = [script]

    def _clean_part(text: str) -> str:
        t = re.sub(r'\*.*?\*', '', text)
        t = re.sub(r'[\.\\-—_…]{2,}', ',', t)
        t = re.sub(r'[\"\'()[\]]', '', t)
        return re.sub(r'\s+', ' ', t).strip()

    cleaned_parts = [_clean_part(p) for p in raw_parts]

    # Handle [PAUSE] tag — the LLM inserts this before the punchline.
    # Wrap in SSML so edge_tts renders it as a genuine 900ms silence break.
    # NOTE: clean_script is set for the non-pause path; SSML path uses cleaned_parts.
    clean_script = cleaned_parts[0]  # used if no pause

    if has_pause and len(cleaned_parts) == 2:
        ssml_script = (
            f"<speak>"
            f"<s>{cleaned_parts[0].strip()}</s>"
            f"<break time=\"900ms\"/>"
            f"<s>{cleaned_parts[1].strip()}</s>"
            f"</speak>"
        )
        communicate = edge_tts.Communicate(ssml_script, voice, rate="+18%", boundary="WordBoundary")
    else:
        # +18% rate: fast enough to feel punchy, slow enough to stay conversational.
        communicate = edge_tts.Communicate(clean_script, voice, rate="+18%", boundary="WordBoundary")
    submaker = edge_tts.SubMaker()

    with audio_path.open("wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                submaker.feed(chunk)

    srt_path.write_text(submaker.get_srt(), encoding="utf-8")


def _parse_srt_file(path: Path) -> list[SrtCue]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    cues: list[SrtCue] = []
    blocks = text.split("\n\n")
    for block in blocks:
        lines = block.split("\n")
        if len(lines) >= 3:
            times = lines[1].split(" --> ")
            if len(times) == 2:
                start = _srt_timestamp_to_seconds(times[0])
                end = _srt_timestamp_to_seconds(times[1])
                content = " ".join(lines[2:])
                cues.append(SrtCue(start=start, end=end, text=content))
    return cues

def _srt_timestamp_to_seconds(timestamp: str) -> float:
    # 00:00:01,234 -> 1.234
    timestamp = timestamp.strip()
    if "," in timestamp:
        time_str, ms_str = timestamp.split(",")
    else:
        time_str, ms_str = timestamp, "0"
    
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms_str) / 1000.0
