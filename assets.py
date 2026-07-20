import re
import requests
from pathlib import Path
from typing import Iterable, Any

from config import PipelineConfig, ContentBrief, PEXELS_VIDEO_SEARCH_URL, _CONTENT_CATEGORIES, logger

def _pexels_fetch_url(
    query: str,
    api_key: str,
    exclude_urls: set[str] | None = None,
) -> str | None:
    """Search Pexels and return the best MP4 download URL for *query*."""
    exclude_urls = exclude_urls or set()
    try:
        headers = {"Authorization": api_key}
        resp = requests.get(
            PEXELS_VIDEO_SEARCH_URL,
            headers=headers,
            params={"query": query, "orientation": "portrait", "per_page": 15},
            timeout=30,
        )
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
        if not videos:
            logger.info("Pexels returned 0 results for '%s'.", query)
            return None

        for video in videos[:10]:
            url = _best_mp4_link(video.get("video_files", []))
            if url and url not in exclude_urls:
                return url

        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Pexels search for '%s' failed: %s", query, exc)
        return None

def _pexels_download_for_query(
    query: str,
    api_key: str,
    output_path: Path,
) -> bool:
    """Try downloading a portrait video from Pexels for *query*.
    Returns True on success.
    """
    try:
        headers = {"Authorization": api_key}
        resp = requests.get(
            PEXELS_VIDEO_SEARCH_URL,
            headers=headers,
            params={"query": query, "orientation": "portrait", "per_page": 15},
            timeout=30,
        )
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
        if not videos:
            logger.info("Pexels returned 0 results for '%s'.", query)
            return False

        # Try candidates in order; the best portrait clip scores highest
        for video in videos[:5]:
            video_url = _best_mp4_link(video.get("video_files", []))
            if not video_url:
                continue
            logger.info("Downloading Pexels video for query '%s'.", query)
            try:
                vr = requests.get(video_url, timeout=180)
                vr.raise_for_status()
                output_path.write_bytes(vr.content)
                return True
            except Exception as exc:
                logger.debug("Pexels download for '%s' failed: %s", video_url, exc)
                continue

        return False
    except Exception as exc:  # noqa: BLE001
        logger.debug("Pexels search for '%s' failed: %s", query, exc)
        return False

def _extract_visual_keywords(brief: "ContentBrief") -> list[str]:
    """Build the ordered list of stock-footage search queries.

    Primary: use the AI-generated ``video_keywords`` tuple (up to 8 terms).
    These are scene-by-scene terms that map to 3-4 second cuts in the video,
    giving the fast-cut stock montage look of the reference Short.

    Fallback: original behaviour using search_query + category terms.
    """
    if brief.video_keywords:
        logger.info(
            "Using %d AI-generated video_keywords for clip search.",
            len(brief.video_keywords),
        )
        return list(brief.video_keywords)  # up to 8 terms

    # Fallback — used when video_keywords is missing (old brief format)
    logger.warning("video_keywords empty; falling back to search_query + category terms.")
    primary = brief.search_query
    category_terms: list[str] = []
    for cat_id, _, _, terms, _ in _CONTENT_CATEGORIES:
        if cat_id == brief.category:
            category_terms = terms[:]
            break
    queries: list[str] = [primary]
    for term in category_terms:
        if term.lower() != primary.lower() and len(queries) < 8:
            queries.append(term)
    return queries

def _download_clip(url: str, output_path: Path) -> bool:
    """Download a single video URL to *output_path*. Returns True on success."""
    try:
        vr = requests.get(url, timeout=180)
        vr.raise_for_status()
        output_path.write_bytes(vr.content)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Clip download failed (%s): %s", url, exc)
        return False

def download_background_clips(config: "PipelineConfig", brief: "ContentBrief") -> list[Path]:
    queries = _extract_visual_keywords(brief)
    workspace = config.workspace
    clips: list[Path] = []
    used_urls: set[str] = set()

    for idx, query in enumerate(queries):
        clip_path = workspace / f"bg_clip_{idx}.mp4"

        if not config.pexels_api_key:
            logger.warning("PEXELS_API_KEY missing; cannot fetch clip %d.", idx + 1)
            continue

        url = _pexels_fetch_url(query, config.pexels_api_key, exclude_urls=used_urls)
        if url and _download_clip(url, clip_path):
            used_urls.add(url)
            clips.append(clip_path)
            logger.info("Clip %d/%d: Pexels '%s' -> %s", idx + 1, len(queries), query, clip_path.name)
        else:
            logger.warning("Could not download clip %d for query '%s'; skipping.", idx + 1, query)

    if not clips:
        logger.error("All clip downloads failed for brief '%s'.", brief.title)

    return clips

def _best_mp4_link(video_files: Iterable[dict[str, Any]]) -> str | None:
    """Select the best MP4 link, strongly preferring portrait (tall) orientation
    since we are producing a 9:16 YouTube Short.

    Scoring:
      - Portrait clips (height > width) get a large bonus so they are always
        preferred over landscape clips of the same resolution.
      - Within each orientation tier we pick the highest resolution * bitrate.
    """
    candidates = []
    for item in video_files:
        if not isinstance(item, dict):
            continue
        link = item.get("link")
        if not isinstance(link, str) or not link:
            continue
        file_type = str(item.get("file_type", "")).lower()
        if file_type not in {"video/mp4", "application/mp4", ""} and not link.lower().endswith(".mp4"):
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        bitrate = int(item.get("bitrate") or 0)
        # Heavy bonus for portrait clips — a 1080x1920 clip is ideal; landscape
        # clips at any resolution score lower than any portrait clip.
        portrait_bonus = 10_000_000_000 if height > width else 0
        score = portrait_bonus + (width * height) + bitrate
        candidates.append((score, link))
    if not candidates:
        return None
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    return candidates[0][1]


