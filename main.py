import os
import sys
from pathlib import Path
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

        final_video_path = assemble_video(
            config,
            background_paths,
            audio_path,
            srt_path,
            punchline_words=set(brief.punchline_words),
        )
        if final_video_path is None:
            logger.error("Failed to assemble video.")
            sys.exit(1)

        creds_1 = Path(os.getenv("YOUTUBE_CREDENTIALS_PATH_1", "credentials_1.json"))
        creds_2 = Path(os.getenv("YOUTUBE_CREDENTIALS_PATH_2", "credentials_2.json"))

        logger.info("Uploading to Channel 1...")
        upload_ok_1 = upload_to_youtube(config, final_video_path, brief.title, brief.description, creds_1, "1")
        if not upload_ok_1:
            logger.error("Failed to upload to YouTube Channel 1.")

        logger.info("Uploading to Channel 2...")
        upload_ok_2 = upload_to_youtube(config, final_video_path, brief.title, brief.description, creds_2, "2")
        if not upload_ok_2:
            logger.error("Failed to upload to YouTube Channel 2.")

        if not upload_ok_1 and not upload_ok_2:
            logger.error("Both YouTube uploads failed.")
            sys.exit(1)
            
        return 0
    except Exception as exc:
        logger.exception("Unhandled pipeline error: %s", exc)
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
