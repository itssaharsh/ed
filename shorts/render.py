"""Stage 10: assemble the video with a single ffmpeg pass per shot, then one concat.

Two measured decisions drive this module:

`zoompan` is unusable. Rendering 3 seconds of 1080x1920 from a 2160x3840 still took **109
seconds** - it rescales the entire input on every output frame. The replacement below does zoom
with `scale:eval=frame` and pan/handheld with time-varying `crop` offsets, and renders a 4-second
shot in **4.3 seconds**, roughly 30x faster.

Captions go through libass rather than PIL-rendered PNGs: this ffmpeg build has no `drawtext`,
and libass gives real typography and per-word karaoke in one burn-in pass.
"""
from __future__ import annotations

import math
import shlex
import subprocess
from pathlib import Path

from .config import FPS, HEIGHT, MASTER_H, MASTER_W, WIDTH, logger


class RenderError(RuntimeError):
    pass


def ffmpeg_bin() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: list[str], what: str, timeout: int = 900) -> None:
    proc = subprocess.run(args, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-1200:]
        raise RenderError(f"{what} failed:\n{tail}")


# ── Camera moves ────────────────────────────────────────────────────────────
# Each returns a filter chain taking the 1296x2304 master to a 1080x1920 clip of `duration`s.
# The master is 1.2x the output, which is the headroom the move travels through.

def _move_chain(motion: str, duration: float, seed: int) -> str:
    d = max(0.35, duration)
    # Deterministic per-shot handheld phase so two shots never float in sync.
    phase = (seed % 100) / 100.0 * math.pi * 2
    fx, fy = 0.55 + (seed % 7) * 0.03, 0.41 + (seed % 5) * 0.04
    ax, ay = 9, 7          # handheld amplitude in px - subtle; more reads as a shaky-cam gag

    cx = f"(in_w-{WIDTH})/2 + {ax}*sin(t*{fx:.2f}+{phase:.2f})"
    cy = f"(in_h-{HEIGHT})/2 + {ay}*sin(t*{fy:.2f}+{phase:.2f})"

    if motion == "push-in":
        # Grow the master 1.00 -> 1.12 while holding a fixed crop window: a slow dolly in.
        scale = f"scale=w='2*floor({MASTER_W}*(1+0.12*t/{d:.3f})/2)':h=-2:eval=frame"
        return f"{scale},crop={WIDTH}:{HEIGHT}:x='{cx}':y='{cy}'"
    if motion == "pull-out":
        scale = f"scale=w='2*floor({MASTER_W}*(1.12-0.12*t/{d:.3f})/2)':h=-2:eval=frame"
        return f"{scale},crop={WIDTH}:{HEIGHT}:x='{cx}':y='{cy}'"
    if motion in ("drift-left", "drift-right"):
        # No rescale at all - the cheapest move, pure pointer arithmetic in crop.
        span = MASTER_W - WIDTH
        travel = f"{span}*(t/{d:.3f})" if motion == "drift-right" else f"{span}*(1-t/{d:.3f})"
        return (f"crop={WIDTH}:{HEIGHT}:"
                f"x='{travel}':y='(in_h-{HEIGHT})/2 + {ay}*sin(t*{fy:.2f}+{phase:.2f})'")
    # static-float: hold, with only the handheld drift. For punch shots where the image is the gag.
    return f"crop={WIDTH}:{HEIGHT}:x='{cx}':y='{cy}'"


def _grade(style: str) -> str:
    """A light finishing pass. Grain in particular hides banding from flat AI gradients."""
    if style == "grain_docu":
        return "eq=contrast=1.04:saturation=0.88:gamma=1.02,noise=alls=9:allf=t+u,vignette=PI/4.5"
    if style == "neon_late":
        return "eq=contrast=1.12:saturation=1.18,noise=alls=4:allf=t,vignette=PI/5"
    return "eq=contrast=1.05:saturation=1.06,noise=alls=3:allf=t"


def render_shot(image: Path, out: Path, *, motion: str, duration: float,
                style: str, seed: int) -> Path:
    chain = _move_chain(motion, duration, seed)
    vf = f"{chain},{_grade(style)},setsar=1,fps={FPS},format=yuv420p"
    _run([
        ffmpeg_bin(), "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(FPS), "-t", f"{duration:.3f}", "-i", str(image),
        "-vf", vf, "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        str(out),
    ], f"shot render ({image.name})", timeout=300)
    return out


def concat_shots(clips: list[Path], out: Path, work: Path) -> Path:
    """Hard cuts between shots.

    Deliberately not cross-fades: a cut lands on a comic beat, a dissolve smears it. The one place
    softness helps is the loop point, handled in `finalize`.
    """
    listing = work / "concat.txt"
    listing.write_text("".join(f"file '{c.resolve()}'\n" for c in clips), encoding="utf-8")
    _run([
        ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(listing), "-c", "copy", str(out),
    ], "shot concat")
    return out


def finalize(video: Path, audio: Path, ass_file: Path, out: Path, *,
             duration: float, fonts_dir: Path | None, music: Path | None = None) -> Path:
    """Burn captions, mix audio, normalise loudness, encode for YouTube."""
    ass_arg = f"ass={shlex.quote(str(ass_file))}"
    if fonts_dir:
        ass_arg += f":fontsdir={shlex.quote(str(fonts_dir))}"

    args = [ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio)]
    if music:
        args += ["-i", str(music)]

    # Hold the last frame past the end before the -t truncation. Per-shot durations are rounded
    # to whole frames, so the concatenated video can land a few milliseconds short of the audio;
    # without this the difference shows up as a black tail.
    vfilter = f"[0:v]tpad=stop_mode=clone:stop_duration=2,{ass_arg},format=yuv420p[v]"
    if music:
        # Music sits far under the voice; it is texture, not a bed. sidechaincompress would be
        # better but costs a second pass for a marginal gain at this volume.
        afilter = (
            "[1:a]aformat=channel_layouts=stereo,volume=1.0[voice];"
            "[2:a]aformat=channel_layouts=stereo,volume=0.10,afade=t=out:st="
            f"{max(0.0, duration - 1.2):.2f}:d=1.2[bed];"
            "[voice][bed]amix=inputs=2:duration=first:dropout_transition=0[mix];"
            "[mix]loudnorm=I=-14:TP=-1.5:LRA=11[a]"
        )
    else:
        afilter = "[1:a]aformat=channel_layouts=stereo,loudnorm=I=-14:TP=-1.5:LRA=11[a]"

    args += [
        "-filter_complex", f"{vfilter};{afilter}",
        "-map", "[v]", "-map", "[a]",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-profile:v", "high", "-level", "4.0", "-pix_fmt", "yuv420p",
        "-g", str(FPS * 2), "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        str(out),
    ]
    _run(args, "finalize", timeout=1800)
    return out


def probe(path: Path) -> dict:
    """Duration / stream facts, read back from the rendered file for the QC gate."""
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run([exe, "-hide_banner", "-i", str(path)],
                          capture_output=True, text=True)
    text = proc.stderr
    info: dict = {"has_video": "Video:" in text, "has_audio": "Audio:" in text}
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", text)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", text)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    return info


def measure_loudness(path: Path) -> float | None:
    """Integrated LUFS, for the QC gate."""
    proc = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-i", str(path), "-af",
         "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=summary", "-f", "null", "-"],
        capture_output=True, text=True, timeout=300,
    )
    import re
    m = re.search(r"Input Integrated:\s*(-?[\d.]+)\s*LUFS", proc.stderr)
    return float(m.group(1)) if m else None
