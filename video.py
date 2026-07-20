import os
from pathlib import Path
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    CompositeAudioClip,
    ImageClip,
    VideoFileClip,
    vfx,
    AudioClip,
    AudioArrayClip,
    afx,
)
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import PipelineConfig, SrtCue, TARGET_WIDTH, TARGET_HEIGHT, logger
from audio import _parse_srt_file

# ── Caption style: center-screen word-burst (matches reference video) ────────
# Captions appear dead center, 2 words at a time, uppercase bold white text
# with a thick black stroke — exactly like the reference YouTube Short.
_CAPTION_PANEL_W = TARGET_WIDTH   # 1080
_CAPTION_PANEL_H = 340            # just enough for 1-2 lines at large font
_CAPTION_FONT_SIZE = 96           # big and impactful
_CAPTION_PADDING_X = 40
_CAPTION_PADDING_Y = 20
_CAPTION_TEXT_COLOR = (255, 255, 255, 255)  # crisp white
_CAPTION_STROKE_COLOR = (0, 0, 0, 255)
_CAPTION_STROKE_WIDTH = 6         # thick stroke so it pops on any background
_CAPTION_WORDS_PER_BURST = 2      # words shown simultaneously

def _ken_burns_zoom(
    clip: "VideoFileClip",
    target_w: int,
    target_h: int,
    zoom_start: float = 1.0,
    zoom_end: float = 1.08,
) -> "VideoFileClip":
    """Apply a slow Ken Burns zoom-in/out effect to *clip*.

    The clip is first scaled up slightly to give room to zoom without
    revealing the edges, then a per-frame crop walks from zoom_start to
    zoom_end (or vice-versa for zoom-out).  This turns even a 5-second
    static clip into an animated, cinematic-feeling segment.

    Args:
        clip:        The source VideoFileClip (already loaded).
        target_w:    Final crop width in pixels.
        target_h:    Final crop height in pixels.
        zoom_start:  Scale factor at the start of the clip (1.0 = exact fit).
        zoom_end:    Scale factor at the end   of the clip.
    """
    # Over-scale so we have pixels to zoom into
    max_zoom = max(zoom_start, zoom_end)
    base_scale = max(target_w / clip.w, target_h / clip.h) * max_zoom
    scaled = clip.resized(base_scale)

    duration = clip.duration

    def _make_frame(t: float):
        # Linear interpolation between zoom_start and zoom_end
        progress = t / duration if duration > 0 else 0.0
        scale = zoom_start + (zoom_end - zoom_start) * progress

        # Current view size (in the over-scaled frame)
        view_w = target_w / scale * max_zoom
        view_h = target_h / scale * max_zoom

        # Clamp to frame bounds
        view_w = min(view_w, scaled.w)
        view_h = min(view_h, scaled.h)

        x1 = (scaled.w - view_w) / 2
        y1 = (scaled.h - view_h) / 2

        frame = scaled.get_frame(t)
        # Crop manually
        import numpy as np
        x1i, y1i = int(x1), int(y1)
        cropped = frame[y1i: y1i + int(view_h), x1i: x1i + int(view_w)]
        # Resize back to target
        from PIL import Image as PILImage
        img = PILImage.fromarray(cropped).resize((target_w, target_h), PILImage.LANCZOS)
        return np.array(img)

    from moviepy import VideoClip
    result = VideoClip(_make_frame, duration=duration)
    result = result.with_fps(clip.fps or 30)
    if clip.audio:
        result = result.with_audio(clip.audio)
    return result

def _process_clip_segment(
    clip_path: Path,
    target_w: int,
    target_h: int,
    segment_duration: float,
    zoom_direction: str = "in",
) -> "VideoFileClip | None":
    """Load, trim, crop, and Ken-Burns-zoom a single background clip.

    Returns a clip of exactly *segment_duration* seconds at *target_w x target_h*,
    or None if the file cannot be loaded.
    """
    try:
        raw = VideoFileClip(str(clip_path))

        # Trim to segment_duration (loop only if the source is shorter than 3s;
        # otherwise just take the first segment_duration seconds of unique footage).
        if raw.duration < 3.0:
            # Very short clip — loop it just once to reach minimum
            from moviepy import vfx
            raw = raw.with_effects([vfx.Loop(duration=max(segment_duration, raw.duration * 2))])

        segment = raw.subclipped(0, min(segment_duration, raw.duration))

        # If the clip is still shorter than needed, pad with freeze-frame
        if segment.duration < segment_duration:
            freeze = segment.to_ImageClip(t=segment.duration - 0.05)
            freeze = freeze.with_duration(segment_duration - segment.duration)
            from moviepy import concatenate_videoclips
            segment = concatenate_videoclips([segment, freeze])

        # Scale + center-crop to portrait
        scale = max(target_w / segment.w, target_h / segment.h) * 1.12  # 12% headroom for zoom
        segment = segment.resized(scale)
        from moviepy import vfx
        segment = segment.with_effects([vfx.Crop(width=target_w, height=target_h,
                                                 x_center=segment.w / 2, y_center=segment.h / 2)])

        # Ken Burns direction alternates so adjacent clips feel different
        zoom_s, zoom_e = (1.0, 1.06) if zoom_direction == "in" else (1.06, 1.0)
        segment = _ken_burns_zoom(segment, target_w, target_h, zoom_s, zoom_e)

        return segment
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not process clip segment '%s': %s", clip_path.name, exc)
        return None

def assemble_video(
    config: "PipelineConfig",
    background_paths: "list[Path] | Path",
    audio_path: Path,
    srt_path: Path,
) -> Path | None:
    """Assemble the final Short from multiple background clips + audio + captions.

    Each background clip gets an equal share of the total duration and has its
    own Ken Burns zoom direction (alternating in/out) so the video feels
    dynamic without any looping or frame repetition.
    """
    # Normalise to list
    if isinstance(background_paths, Path):
        background_paths = [background_paths]

    if not background_paths:
        logger.error("No background clips provided to assemble_video.")
        return None

    try:
        audio_clip = AudioFileClip(str(audio_path))

        # ── Word-burst captions — group individual words into 2-word pops ────
        word_cues = _parse_srt_file(srt_path)
        subtitle_cues = _group_word_cues(word_cues, words_per_group=_CAPTION_WORDS_PER_BURST)
        subtitle_clips = [_subtitle_clip_for_cue(cue) for cue in subtitle_cues]

        # ── Determine total video duration ───────────────────────────────────
        min_duration = 30.0
        target_duration = max(min_duration, float(audio_clip.duration))

        # Pad audio with silence if needed
        if audio_clip.duration < target_duration:
            silence_duration = target_duration - audio_clip.duration
            fps_a = getattr(audio_clip, "fps", 44100)
            nch = getattr(audio_clip, "nchannels", 1)
            n_samples = int(silence_duration * fps_a)
            arr = np.zeros((n_samples, nch if nch > 1 else 1), dtype=float)
            silence_clip = AudioArrayClip(arr, fps_a)
            audio_clip = CompositeAudioClip([
                audio_clip.with_start(0),
                silence_clip.with_start(audio_clip.duration),
            ])

        # ── Split duration equally across all clips ───────────────────────────
        n_clips = len(background_paths)
        seg_duration = target_duration / n_clips
        zoom_dirs = ["in", "out", "in", "out"]  # alternating

        segments: list = []
        raw_clips: list = []
        try:
            for i, clip_path in enumerate(background_paths):
                seg = _process_clip_segment(
                    clip_path,
                    TARGET_WIDTH,
                    TARGET_HEIGHT,
                    seg_duration,
                    zoom_direction=zoom_dirs[i % len(zoom_dirs)],
                )
                if seg is not None:
                    segments.append(seg)
                    raw_clips.append(seg)  # track for cleanup

            if not segments:
                logger.error("All clip segments failed to process.")
                return None

            # If some clips failed, stretch the good ones to fill total duration
            if len(segments) < n_clips:
                each = target_duration / len(segments)
                new_segs = []
                for i, seg in enumerate(segments):
                    # Reprocess with corrected duration
                    corrected = _process_clip_segment(
                        background_paths[i], TARGET_WIDTH, TARGET_HEIGHT,
                        each, zoom_dirs[i % len(zoom_dirs)]
                    )
                    new_segs.append(corrected if corrected is not None else seg)
                segments = new_segs

            # Concatenate all segments
            from moviepy import concatenate_videoclips
            background = concatenate_videoclips(segments, method="compose")
            # Ensure exact duration (floating-point drift)
            background = background.subclipped(0, target_duration)
            background = background.with_audio(audio_clip)

            composite = CompositeVideoClip([background, *subtitle_clips], size=(TARGET_WIDTH, TARGET_HEIGHT))
            composite = composite.with_duration(target_duration).with_audio(audio_clip)

            config.output_final.parent.mkdir(parents=True, exist_ok=True)
            composite.write_videofile(
                str(config.output_final),
                fps=30,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="medium",
                logger=None,
            )
        finally:
            for clip in subtitle_clips:
                try: clip.close()
                except Exception: pass
            for clip in raw_clips:
                try: clip.close()
                except Exception: pass
            try: audio_clip.close()
            except Exception: pass

        logger.info("Rendered final video: %s", config.output_final)
        return config.output_final

    except Exception as exc:  # noqa: BLE001
        logger.error("Video assembly failed: %s", exc)
        return None

def _group_word_cues(cues: list[SrtCue], words_per_group: int = 2) -> list[SrtCue]:
    """Group per-word SRT cues into N-word burst captions.

    edge_tts with WordBoundary emits one SRT entry per word.  Grouping
    them into pairs of 2 creates the snappy pop-up caption style seen in
    the reference video and virtually every viral Short/Reel.
    Text is UPPERCASED for maximum visual impact.
    """
    if not cues:
        return []
    grouped: list[SrtCue] = []
    for i in range(0, len(cues), words_per_group):
        chunk = cues[i : i + words_per_group]
        text = " ".join(c.text for c in chunk).upper()
        start = chunk[0].start
        end = chunk[-1].end
        # Give each burst at least 0.25s on screen so fast words are readable
        end = max(end, start + 0.25)
        grouped.append(SrtCue(start=start, end=end, text=text))
    return grouped


def _subtitle_clip_for_cue(cue: SrtCue) -> ImageClip:
    image_path = _render_caption_image(cue.text)
    clip = (
        ImageClip(str(image_path))
        .with_start(cue.start)
        .with_duration(max(cue.end - cue.start, 0.25))
    )
    # Dead center on screen — matches the reference video's caption placement
    return clip.with_position(("center", "center"))

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
    """Render a word-burst caption as a transparent PNG.

    Style: bold uppercase white text with thick black stroke on a fully
    transparent background.  No pill — the text floats cleanly over the
    video footage, exactly like the reference Short.
    """
    temp_dir = Path(__file__).resolve().parent / "_caption_cache"
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"caption_{abs(hash(text))}.png"
    output_path = temp_dir / file_name
    if output_path.exists():
        return output_path  # use cache to avoid re-rendering duplicates

    font = _load_caption_font(_CAPTION_FONT_SIZE)

    # -- Measure text on a probe canvas ---------------------------------------
    probe = Image.new("RGBA", (1, 1))
    probe_draw = ImageDraw.Draw(probe)
    max_text_w = _CAPTION_PANEL_W - 2 * _CAPTION_PADDING_X
    wrapped_text = _wrap_text(probe_draw, text, font, max_width=max_text_w)
    bbox = probe_draw.multiline_textbbox(
        (0, 0), wrapped_text, font=font, spacing=8, stroke_width=_CAPTION_STROKE_WIDTH
    )
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # -- Canvas: just big enough to hold the text with padding ----------------
    canvas_w = _CAPTION_PANEL_W
    canvas_h = text_h + 2 * _CAPTION_PADDING_Y
    image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Centered text position
    text_x = (canvas_w - text_w) / 2
    text_y = _CAPTION_PADDING_Y

    # Main text: white fill + thick black stroke (no separate shadow needed)
    draw.multiline_text(
        (text_x, text_y),
        wrapped_text,
        font=font,
        fill=_CAPTION_TEXT_COLOR,
        spacing=8,
        align="center",
        stroke_width=_CAPTION_STROKE_WIDTH,
        stroke_fill=_CAPTION_STROKE_COLOR,
    )

    image.save(output_path)
    return output_path

def _load_caption_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the boldest available font. Checks Windows paths first for local dev,
    then Linux paths for GitHub Actions (Ubuntu runner)."""
    font_candidates = [
        # Windows — Arial Bold is ideal for the reference video's look
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arial Bold.ttf",
        # Linux / GitHub Actions (Ubuntu)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    logger.warning("No TrueType bold font found; falling back to PIL default (captions will look basic).")
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

