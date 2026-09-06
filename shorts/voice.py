"""Stage 8: speech with comedic timing.

The old pipeline sent one long string to edge-tts at a flat "+18%" rate and replaced its own
[PAUSE] marker with a full stop. That is why the delivery was flat.

Here each line is synthesised separately, which Orpheus's 200-character cap forces anyway, and
the silence between lines is *designed* (see prompts/05_delivery.md) rather than left to the
engine. Comic timing lives in those silences more than in any tag.
"""
from __future__ import annotations

import asyncio
import re
import base64
import io
import math
import struct
import subprocess
import wave
from pathlib import Path

import requests

from .config import (
    EDGE_FALLBACK_VOICE, ORPHEUS_CHAR_LIMIT, ORPHEUS_MODEL, Config, logger,
)
from .write import Line

SAMPLE_RATE = 24000


class VoiceError(RuntimeError):
    pass


def ffmpeg_bin() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# ── Synthesis providers ─────────────────────────────────────────────────────

def _orpheus(cfg: Config, text: str, voice: str, direction: str | None) -> bytes:
    """Groq-hosted Orpheus. Free tier: 10 RPM / 100 RPD. 200 chars per request, hard."""
    payload = f"[{direction}] {text}" if direction else text
    if len(payload) > ORPHEUS_CHAR_LIMIT:
        raise VoiceError(f"line too long for orpheus ({len(payload)} chars)")
    r = requests.post(
        "https://api.groq.com/openai/v1/audio/speech",
        headers={"Authorization": f"Bearer {cfg.groq_key}", "Content-Type": "application/json"},
        json={"model": ORPHEUS_MODEL, "voice": voice, "input": payload, "response_format": "wav"},
        timeout=120,
    )
    if r.status_code != 200:
        raise VoiceError(f"orpheus {r.status_code}: {r.text[:200]}")
    if len(r.content) < 1000:
        raise VoiceError("orpheus returned an empty clip")
    return r.content


_NONVERBAL_TAG = re.compile(r"<(?:laugh|sigh|giggle|groan|chuckle|gasp|cough|sniff|yawn)>", re.I)


def _edge(text: str, rate: str = "+8%") -> bytes:
    """Keyless floor. Microsoft removed custom SSML, so only rate/volume/pitch are available.

    Orpheus non-verbals must be stripped first: edge-tts has no concept of them and reads the
    literal characters aloud.
    """
    import edge_tts

    text = re.sub(r"\s+", " ", _NONVERBAL_TAG.sub("", text)).strip()
    if not text:
        raise VoiceError("nothing left to speak after stripping tags")

    async def go() -> bytes:
        buf = io.BytesIO()
        comm = edge_tts.Communicate(text, EDGE_FALLBACK_VOICE, rate=rate)
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    data = asyncio.run(go())
    if len(data) < 800:
        raise VoiceError("edge-tts returned an empty clip")
    return data


# ── Audio helpers (ffmpeg, no moviepy) ──────────────────────────────────────

def _to_wav(raw: bytes, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [ffmpeg_bin(), "-y", "-loglevel", "error", "-i", "pipe:0",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out)],
        input=raw, capture_output=True,
    )
    if proc.returncode != 0 or not out.exists():
        raise VoiceError(f"ffmpeg decode failed: {proc.stderr.decode()[:200]}")
    return out


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _silence(seconds: float, out: Path) -> Path:
    n = max(1, int(seconds * SAMPLE_RATE))
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * n)
    return out


def _trim_edges(path: Path, out: Path) -> Path:
    """Strip leading/trailing near-silence.

    Voice models pad clips with room tone. Left in, that padding stacks on top of the designed
    pauses and every beat drifts late, which is exactly what kills timing.
    """
    proc = subprocess.run(
        [ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(path),
         "-af", "silenceremove=start_periods=1:start_silence=0.02:start_threshold=-45dB:"
                "detection=peak,areverse,"
                "silenceremove=start_periods=1:start_silence=0.02:start_threshold=-45dB:"
                "detection=peak,areverse",
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(out)],
        capture_output=True,
    )
    if proc.returncode != 0 or not out.exists() or out.stat().st_size < 200:
        return path
    return out


def _split_for_limit(text: str, limit: int) -> list[str]:
    """Split an over-long line on sentence boundaries, then on words."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for chunk in text.replace("? ", "?|").replace("! ", "!|").replace(". ", ".|").split("|"):
        if len(cur) + len(chunk) + 1 <= limit:
            cur = f"{cur} {chunk}".strip()
        else:
            if cur:
                parts.append(cur)
            cur = chunk
    if cur:
        parts.append(cur)

    out: list[str] = []
    for p in parts:
        while len(p) > limit:
            cut = p.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


# ── Main entry ──────────────────────────────────────────────────────────────

def synthesize(cfg: Config, lines: list[Line], voice: str, work: Path) -> tuple[Path, float, str]:
    """Render every line, insert designed silences, concatenate.

    Mutates each Line with its measured `duration` and absolute `start`, so captions and shot
    cuts downstream lock to the real performance rather than an estimate.
    """
    audio_dir = work / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    use_orpheus = bool(cfg.groq_key)
    engine = "orpheus" if use_orpheus else "edge-tts"
    segments: list[Path] = []
    cursor = 0.0
    degraded = False

    for line in lines:
        if line.pause_before_ms > 0:
            sp = _silence(line.pause_before_ms / 1000.0, audio_dir / f"pause_{line.index:02d}.wav")
            segments.append(sp)
            cursor += line.pause_before_ms / 1000.0

        chunks = _split_for_limit(line.text, ORPHEUS_CHAR_LIMIT - 16)
        line_start = cursor
        for ci, chunk in enumerate(chunks):
            raw: bytes | None = None
            if use_orpheus:
                try:
                    raw = _orpheus(cfg, chunk, voice, line.direction if ci == 0 else None)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("orpheus failed on line %d chunk %d (%s); using edge-tts",
                                   line.index, ci, str(exc)[:120])
                    degraded = True
            if raw is None:
                raw = _edge(chunk)

            wav = _to_wav(raw, audio_dir / f"line_{line.index:02d}_{ci}.wav")
            wav = _trim_edges(wav, audio_dir / f"line_{line.index:02d}_{ci}_t.wav")
            dur = wav_duration(wav)
            segments.append(wav)
            cursor += dur

            if ci < len(chunks) - 1:      # small breath between forced splits
                sp = _silence(0.12, audio_dir / f"gap_{line.index:02d}_{ci}.wav")
                segments.append(sp)
                cursor += 0.12

        line.start = line_start
        line.duration = cursor - line_start
        line.audio_path = str(wav)

    out = work / "voice.wav"
    _concat(segments, out)
    total = wav_duration(out)
    if degraded and engine == "orpheus":
        engine = "orpheus (partial, edge-tts fallback used)"
    logger.info("stage 8: %.1fs of speech via %s across %d lines", total, engine, len(lines))
    return out, total, engine


def _concat(segments: list[Path], out: Path) -> Path:
    """Sample-accurate concatenation by writing raw PCM frames directly.

    ffmpeg's concat demuxer re-muxes and can drift by a frame or two per join; with a dozen
    joins that is enough to desynchronise captions from speech.
    """
    frames = bytearray()
    for seg in segments:
        with wave.open(str(seg), "rb") as w:
            if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
                raise VoiceError(f"segment {seg.name} has unexpected format")
            frames += w.readframes(w.getnframes())
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(frames))
    return out


def estimate_word_times(lines: list[Line]) -> list[tuple[float, float, str, bool]]:
    """Per-word timings for captions: (start, end, word, is_emphasis).

    Distributed within each measured line by character weight, with a small extra weight on
    long words. The line boundaries themselves are exact because each line was synthesised
    separately, so drift cannot accumulate across the video.
    """
    out: list[tuple[float, float, str, bool]] = []
    for line in lines:
        words = line.text.split()
        if not words or line.duration <= 0:
            continue
        emph = {e.strip(".,!?").upper() for e in line.emphasis}
        weights = [len(w) + 2.0 for w in words]
        total_w = sum(weights) or 1.0
        t = line.start
        for w, weight in zip(words, weights):
            span = line.duration * (weight / total_w)
            clean = w.strip(".,!?;:\"'").upper()
            out.append((t, t + span, w, clean in emph))
            t += span
    return out
