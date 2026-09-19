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
import threading
import time
import wave
from pathlib import Path

import numpy as np
import requests

from .config import (
    EDGE_FALLBACK_VOICE, GROQ_TTS_MIN_INTERVAL, MIN_VOICE_PEAK_DBFS,
    ORPHEUS_CHAR_LIMIT, ORPHEUS_MODEL, Config, logger,
)
from .write import Line

SAMPLE_RATE = 24000


class VoiceError(RuntimeError):
    pass


def ffmpeg_bin() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# ── Synthesis providers ─────────────────────────────────────────────────────

_groq_tts_lock = threading.Lock()
_groq_tts_last = 0.0


def _groq_tts_wait() -> None:
    """Process-wide pacing for Orpheus, mirroring images._pollinations_wait.

    Orpheus is 10 RPM. A 7-beat script fires 7+ calls, so without this every run with a key
    trips the limit partway through and silently falls back to edge-tts mid-script.
    """
    global _groq_tts_last
    with _groq_tts_lock:
        gap = time.monotonic() - _groq_tts_last
        if gap < GROQ_TTS_MIN_INTERVAL:
            time.sleep(GROQ_TTS_MIN_INTERVAL - gap)
        _groq_tts_last = time.monotonic()


def _orpheus(cfg: Config, text: str, voice: str, direction: str | None) -> bytes:
    """Groq-hosted Orpheus. Free tier: 10 RPM / 100 RPD. 200 chars per request, hard."""
    payload = f"[{direction}] {text}" if direction else text
    if len(payload) > ORPHEUS_CHAR_LIMIT:
        raise VoiceError(f"line too long for orpheus ({len(payload)} chars)")
    _groq_tts_wait()
    r = requests.post(
        "https://api.groq.com/openai/v1/audio/speech",
        headers={"Authorization": f"Bearer {cfg.groq_key}", "Content-Type": "application/json"},
        json={"model": ORPHEUS_MODEL, "voice": voice, "input": payload, "response_format": "wav"},
        timeout=120,
    )
    # Groq reports the remaining budget on every response. Log it: this is the only way to
    # settle whether the documented 3.6K TPD counts characters (~5 videos/day) or tokens
    # (~23/day), which is the binding constraint on how many videos a day are possible.
    remaining = {k: v for k, v in r.headers.items()
                 if k.lower().startswith("x-ratelimit-remaining")}
    if remaining:
        logger.info("orpheus budget: %s", ", ".join(f"{k.split('-')[-1]}={v}"
                                                    for k, v in sorted(remaining.items())))
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


def peak_dbfs(path: Path) -> float:
    """Peak amplitude of a WAV in dBFS. -inf silence is reported as -120.0.

    Cheap enough to run on every chunk: numpy over the raw frames, no subprocess.
    """
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    if not frames:
        return -120.0
    samples = np.frombuffer(frames, dtype=np.int16)
    if samples.size == 0:
        return -120.0
    peak = int(np.abs(samples).max())
    if peak == 0:
        return -120.0
    return 20.0 * math.log10(peak / 32768.0)


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

class _OrpheusUnavailable(RuntimeError):
    """Raised to abandon Orpheus for the whole script, not just one chunk."""


def _speak(cfg: Config, text: str, voice: str, direction: str | None, *,
           engine: str, out_stem: Path) -> tuple[Path, float]:
    """Synthesise one chunk and prove it contains sound. Returns (trimmed wav, duration).

    The liveness check is the point. edge-tts answers a rejected request with a well-formed,
    correctly-sized, entirely silent clip - so byte-length tells you nothing. Without this a
    silent video renders cleanly, passes QC and publishes, which is exactly how the previous
    pipeline uploaded a video built from a placeholder.
    """
    if engine == "orpheus":
        try:
            raw = _orpheus(cfg, text, voice, direction)
        except Exception as exc:  # noqa: BLE001
            raise _OrpheusUnavailable(str(exc)[:160]) from exc
    else:
        raw = _edge(text)

    wav = _to_wav(raw, out_stem.with_name(out_stem.name + ".wav"))
    wav = _trim_edges(wav, out_stem.with_name(out_stem.name + "_t.wav"))

    peak = peak_dbfs(wav)
    if peak <= MIN_VOICE_PEAK_DBFS:
        msg = f"{engine} returned silence ({peak:.1f} dBFS) for {text[:40]!r}"
        if engine == "orpheus":
            raise _OrpheusUnavailable(msg)
        raise VoiceError(
            f"{msg}. edge-tts serves silent audio when Microsoft's anti-abuse check rejects "
            f"the caller, which is normal from datacenter IPs such as CI runners."
        )
    return wav, wav_duration(wav)


def _perform(cfg: Config, lines: list[Line], voice: str, audio_dir: Path,
             engine: str) -> tuple[list[Path], float]:
    """Synthesise every line with one engine. Mutates Line.start/.duration/.audio_path."""
    segments: list[Path] = []
    cursor = 0.0

    for line in lines:
        if line.pause_before_ms > 0:
            sp = _silence(line.pause_before_ms / 1000.0, audio_dir / f"pause_{line.index:02d}.wav")
            segments.append(sp)
            cursor += line.pause_before_ms / 1000.0

        chunks = _split_for_limit(line.text, ORPHEUS_CHAR_LIMIT - 16)
        line_start = cursor
        wav = None
        for ci, chunk in enumerate(chunks):
            wav, dur = _speak(
                cfg, chunk, voice, line.direction if ci == 0 else None,
                engine=engine, out_stem=audio_dir / f"line_{line.index:02d}_{ci}",
            )
            segments.append(wav)
            cursor += dur

            if ci < len(chunks) - 1:      # small breath between forced splits
                sp = _silence(0.12, audio_dir / f"gap_{line.index:02d}_{ci}.wav")
                segments.append(sp)
                cursor += 0.12

        line.start = line_start
        line.duration = cursor - line_start
        line.audio_path = str(wav)

    return segments, cursor


# ── Main entry ──────────────────────────────────────────────────────────────

def synthesize(cfg: Config, lines: list[Line], voice: str,
               work: Path) -> tuple[Path, float, str, float]:
    """Render every line, insert designed silences, concatenate.

    Mutates each Line with its measured `duration` and absolute `start`, so captions and shot
    cuts downstream lock to the real performance rather than an estimate.

    Returns (path, seconds, engine, peak_dbfs). The peak is returned rather than merely checked
    so the QC gate can fail on it independently, downstream of the render.

    The engine is chosen once for the whole script. The previous per-chunk fallback meant an
    Orpheus rate-limit at line 4 produced a video that changed narrator halfway through the
    joke - audible, and only recorded as "partial" in a log line. Re-synthesising from line 0
    is nearly free: this stage runs before image generation in both entry paths, so a restart
    costs seconds rather than any expensive artefact.
    """
    audio_dir = work / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    engine = "orpheus" if cfg.groq_key else "edge-tts"
    try:
        segments, _ = _perform(cfg, lines, voice, audio_dir, engine)
    except _OrpheusUnavailable as exc:
        logger.warning("orpheus unavailable (%s); re-synthesising the whole script with edge-tts",
                       exc)
        engine = "edge-tts"
        segments, _ = _perform(cfg, lines, voice, audio_dir, engine)

    out = work / "voice.wav"
    _concat(segments, out)
    total = wav_duration(out)

    peak = peak_dbfs(out)
    if peak <= MIN_VOICE_PEAK_DBFS:
        raise VoiceError(f"assembled narration is silent ({peak:.1f} dBFS); refusing to render")

    logger.info("stage 8: %.1fs of speech via %s across %d lines (peak %.1f dBFS)",
                total, engine, len(lines), peak)
    return out, total, engine, peak


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
