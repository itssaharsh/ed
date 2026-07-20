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

# Caption panel dimensions
_CAPTION_PANEL_W = TARGET_WIDTH          # 1080
_CAPTION_PANEL_H = 500                   # tall enough for 3 wrapped lines
_CAPTION_FONT_SIZE = 80                  # bigger = more comedic impact
_CAPTION_PADDING_X = 52
_CAPTION_PADDING_Y = 32
_CAPTION_BG_COLOR = (10, 10, 10, 185)    # near-black pill, high contrast
_CAPTION_BG_RADIUS = 32                  # pill corner radius
_CAPTION_TEXT_COLOR = (255, 230, 0, 255) # BRIGHT YELLOW — comedy gold
_CAPTION_STROKE_COLOR = (0, 0, 0, 255)
_CAPTION_STROKE_WIDTH = 5               # chunky stroke for maximum pop

# How far up from the bottom of the full frame the caption panel sits (px)
_CAPTION_BOTTOM_OFFSET = 160

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
        subtitle_cues = _parse_srt_file(srt_path)
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

def _subtitle_clip_for_cue(cue: SrtCue) -> ImageClip:
    image_path = _render_caption_image(cue.text)
    clip = ImageClip(str(image_path)).with_start(cue.start).with_duration(max(cue.end - cue.start, 0.2))
    # Pin to bottom-third: y is measured from top of the full 1920px frame
    y_pos = TARGET_HEIGHT - _CAPTION_PANEL_H - _CAPTION_BOTTOM_OFFSET
    return clip.with_position(("center", y_pos))

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

