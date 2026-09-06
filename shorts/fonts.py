"""Caption font. Ships nothing binary in git; fetched once and cached.

Anton (SIL Open Font License 1.1) is the heavy condensed sans that reads as the Shorts caption
look. If the download fails we fall back to whatever bold face the system has, which is why
`resolve` returns a family name rather than a path.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import requests

from .config import ROOT, logger

FONT_DIR = ROOT / "assets" / "fonts"
ANTON_URL = "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"


def ensure_font() -> tuple[str, Path | None]:
    """Returns (family_name, fonts_dir_for_ffmpeg)."""
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    target = FONT_DIR / "Anton-Regular.ttf"
    if not target.exists():
        try:
            r = requests.get(ANTON_URL, timeout=60)
            r.raise_for_status()
            if len(r.content) < 20_000:
                raise ValueError(f"font download too small: {len(r.content)} bytes")
            target.write_bytes(r.content)
            logger.info("downloaded Anton (%d KB)", len(r.content) // 1024)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not fetch Anton (%s); falling back to a system bold face", exc)
            return _system_bold(), None
    return "Anton", FONT_DIR


def _system_bold() -> str:
    for family in ("DejaVu Sans", "Liberation Sans", "FreeSans"):
        try:
            out = subprocess.run(["fc-list", family], capture_output=True, text=True, timeout=10)
            if out.stdout.strip():
                return family
        except Exception:  # noqa: BLE001
            continue
    return "DejaVu Sans"
