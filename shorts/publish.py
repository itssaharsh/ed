"""Stage 12: upload to YouTube.

Carried over from the old pipeline with three changes:

  * It no longer uploads the *same file* to two channels. That is a real duplicate-content risk
    under the mass-produced/repetitive content policy. One render, one channel.
  * `invalid_grant` is reported as the terminal, human-fixable error it is rather than being
    retried.
  * Uploads default to `private`. Publishing is opted into explicitly, so a bad run costs nothing.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

from .config import YOUTUBE_UPLOAD_SCOPE, Config, logger


class PublishError(RuntimeError):
    pass


def _ensure_shorts(text: str, *, limit: int) -> str:
    t = text.strip()
    if "#shorts" not in t.lower():
        candidate = f"{t} #shorts"
        t = candidate if len(candidate) <= limit else t
    return t[:limit]


def upload(cfg: Config, video: Path, *, title: str, description: str, tags: list[str],
           creds_path: Path, creds_id: str = "1") -> str | None:
    """Upload and return the video id, or None on failure."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    if not creds_path.exists():
        raise PublishError(f"no YouTube credentials at {creds_path}")
    if not video.exists():
        raise PublishError(f"no video at {video}")

    creds = Credentials.from_authorized_user_file(str(creds_path), scopes=[YOUTUBE_UPLOAD_SCOPE])

    if creds.expired and creds.refresh_token:
        logger.info("refreshing YouTube token")
        try:
            creds.refresh(Request())
        except Exception as exc:  # noqa: BLE001
            if "invalid_grant" in str(exc):
                raise PublishError(
                    "the YouTube refresh token has been revoked (invalid_grant). This is terminal "
                    "and retrying will not help. Re-run the OAuth flow locally and update the "
                    "YOUTUBE_CREDENTIALS_JSON secret. If the OAuth consent screen is still in "
                    "'Testing', publish it to 'Production' first, or tokens will expire every "
                    "7 days."
                ) from exc
            raise PublishError(f"token refresh failed: {exc}") from exc

        new_json = creds.to_json()
        creds_path.write_text(new_json, encoding="utf-8")
        gh_out = os.environ.get("GITHUB_OUTPUT")
        if gh_out:
            delim = f"EOF_CREDS_{creds_id}"
            with open(gh_out, "a", encoding="utf-8") as fh:
                fh.write(f"new_youtube_creds_{creds_id}<<{delim}\n{new_json}\n{delim}\n")
            logger.info("exported refreshed credentials for the workflow to persist")

    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
    body = {
        "snippet": {
            "title": _ensure_shorts(title, limit=100),
            "description": description[:4900],
            "categoryId": "23",                     # Comedy
            "tags": [t[:30] for t in tags][:15],
        },
        "status": {
            "privacyStatus": cfg.privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    request = youtube.videos().insert(
        part="snippet,status", body=body,
        media_body=MediaFileUpload(str(video), chunksize=-1, resumable=True),
    )

    retries = 0
    while True:
        try:
            _, response = request.next_chunk()
            if response and "id" in response:
                vid = response["id"]
                logger.info("uploaded as %s (privacy=%s) https://youtube.com/shorts/%s",
                            vid, cfg.privacy, vid)
                return vid
        except HttpError as exc:
            if exc.resp.status in {500, 502, 503, 504} and retries < 5:
                retries += 1
                wait = random.uniform(1.0, 2.0 ** retries)
                logger.warning("transient YouTube error %s; retrying in %.1fs",
                               exc.resp.status, wait)
                time.sleep(wait)
                continue
            if exc.resp.status == 403 and "quota" in str(exc).lower():
                raise PublishError(
                    "YouTube API quota exhausted. videos.insert costs 1600 units of the "
                    "10,000/day allowance, so 6 uploads/day is the hard ceiling."
                ) from exc
            raise PublishError(f"upload failed: {exc}") from exc
