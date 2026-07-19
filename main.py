import sys
import logging
from config import build_config, logger
from gemini import generate_content_brief
from assets import download_background_clips
from youtube import upload_to_youtube
from audio import generate_voice_and_captions
from video import assemble_video

def main() -> int:
    config = build_config()
    try:
        brief = generate_content_brief(config)
        if brief is None:
            return 0

        audio_path, srt_path = generate_voice_and_captions(config, brief.script)
        if audio_path is None or srt_path is None:
            return 0

        # Note: assemble_video expects a single video path or a list of paths.
        # Since we use multiple clips, we pass the list.
        background_paths = download_background_clips(config, brief)
        if not background_paths:
            return 0

        final_video_path = assemble_video(config, background_paths, audio_path, srt_path)
        if final_video_path is None:
            return 0

        upload_ok = upload_to_youtube(config, final_video_path, brief.title, brief.description)
        return 0 if upload_ok else 1
    except Exception as exc:
        logger.exception("Unhandled pipeline error: %s", exc)
        return 0

if __name__ == "__main__":
    sys.exit(main())
