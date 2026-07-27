import os
import time
from pathlib import Path
from typing import Any

import random

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from config import PipelineConfig, YOUTUBE_UPLOAD_SCOPE, logger
from utils import _ensure_shorts_tag, _ensure_shorts_hashtag

def upload_to_youtube(config: PipelineConfig, video_path: Path, title: str, description: str, creds_path: Path, creds_id: str) -> bool:
    """Upload *video_path* to YouTube. Returns True on success, False on any failure."""
    if not creds_path.exists():
        logger.error("YouTube credentials file is missing at %s. Skipping upload.", creds_path)
        return False

    try:
        credentials = Credentials.from_authorized_user_file(str(creds_path), scopes=[YOUTUBE_UPLOAD_SCOPE])

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
            creds_path.write_text(new_creds_json, encoding="utf-8")

            # Write the new token to GITHUB_OUTPUT so the workflow file can capture and save it permanently.
            # NOTE: This block must remain AFTER a successful refresh (not inside the try above)
            # so that invalid_grant failures never reach it.
            github_output = os.environ.get("GITHUB_OUTPUT")
            if github_output:
                with open(github_output, "a", encoding="utf-8") as f:
                    delimiter = f"EOF_CREDS_{creds_id}"
                    f.write(f"new_youtube_creds_{creds_id}<<{delimiter}\n{new_creds_json}\n{delimiter}\n")
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

