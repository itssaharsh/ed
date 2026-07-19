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
            logger.error("Failed to generate content brief.")
            sys.exit(1)

        audio_path, srt_path = generate_voice_and_captions(config, brief.script)
        if audio_path is None or srt_path is None:
            logger.error("Failed to generate audio and captions.")
            sys.exit(1)

        # Note: assemble_video expects a single video path or a list of paths.
        # Since we use multiple clips, we pass the list.
        background_paths = download_background_clips(config, brief)
        if not background_paths:
            logger.error("Failed to download background clips.")
            sys.exit(1)

        final_video_path = assemble_video(config, background_paths, audio_path, srt_path)
        if final_video_path is None:
            logger.error("Failed to assemble video.")
            sys.exit(1)

        upload_ok = upload_to_youtube(config, final_video_path, brief.title, brief.description)
        if not upload_ok:
            logger.error("Failed to upload to YouTube.")
            sys.exit(1)
        return 0
    except Exception as exc:
        logger.exception("Unhandled pipeline error: %s", exc)
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
