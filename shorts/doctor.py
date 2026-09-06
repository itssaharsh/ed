"""Preflight: check every provider before a run, so failures surface in seconds not minutes.

`python run.py --doctor` makes one cheap live call per configured service and reports exactly
what this machine can do right now, what it will fall back to, and what is missing.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from .config import CF_IMAGE_MODEL, Config, ORPHEUS_MODEL, logger

OK, WARN, BAD = "ok", "degraded", "missing"
_MARK = {OK: "  ok    ", WARN: "  warn  ", BAD: "  FAIL  "}


def _row(status: str, label: str, detail: str) -> tuple[str, str, str]:
    print(f"{_MARK[status]}{label:22s} {detail}")
    return (status, label, detail)


def check_llm(cfg: Config) -> tuple[str, str, str]:
    if cfg.gemini_key:
        try:
            from google import genai
            from google.genai import types
            from .config import GEMINI_MODELS
            c = genai.Client(api_key=cfg.gemini_key)
            r = c.models.generate_content(
                model=GEMINI_MODELS[0], contents="Reply with the single word OK.",
                config=types.GenerateContentConfig(max_output_tokens=2000, temperature=0),
            )
            if r.text:
                return _row(OK, "LLM", f"gemini {GEMINI_MODELS[0]} responding (500 req/day free)")
            return _row(BAD, "LLM", "gemini returned empty text")
        except Exception as exc:  # noqa: BLE001
            return _row(BAD, "LLM", f"gemini key set but failing: {str(exc)[:90]}")

    if cfg.openrouter_key or cfg.groq_key:
        which = "openrouter" if cfg.openrouter_key else "groq"
        return _row(WARN, "LLM", f"{which} configured (not probed); gemini recommended")

    return _row(BAD, "LLM", "no key - the pipeline cannot run. Set GEMINI_API_KEY.")


def check_images(cfg: Config) -> tuple[str, str, str]:
    if cfg.cf_account and cfg.cf_token:
        try:
            r = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{cfg.cf_account}/ai/run/{CF_IMAGE_MODEL}",
                headers={"Authorization": f"Bearer {cfg.cf_token}"},
                json={"prompt": "a plain grey circle on white", "steps": 4},
                timeout=90,
            )
            if r.status_code == 200:
                return _row(OK, "images", "cloudflare flux-1-schnell (~145 portrait images/day free)")
            return _row(BAD, "images", f"cloudflare returned {r.status_code}: {r.text[:80]}")
        except Exception as exc:  # noqa: BLE001
            return _row(BAD, "images", f"cloudflare unreachable: {str(exc)[:80]}")

    tier = "token" if cfg.pollinations_token else "anonymous"
    try:
        t0 = time.time()
        r = requests.get("https://image.pollinations.ai/prompt/a%20plain%20grey%20circle",
                         params={"width": 256, "height": 256, "nologo": "true"}, timeout=90)
        elapsed = time.time() - t0
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
            model = r.headers.get("x-model-used", "?")
            return _row(WARN, "images",
                        f"pollinations {tier} ({model}, {elapsed:.0f}s/image) - "
                        f"low quality. Add CLOUDFLARE_* for FLUX.")
        return _row(BAD, "images", f"pollinations returned {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        return _row(BAD, "images", f"no working image provider: {str(exc)[:80]}")


def check_voice(cfg: Config) -> tuple[str, str, str]:
    if cfg.groq_key:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/speech",
                headers={"Authorization": f"Bearer {cfg.groq_key}", "Content-Type": "application/json"},
                json={"model": ORPHEUS_MODEL, "voice": "austin",
                      "input": "[deadpan] Testing.", "response_format": "wav"},
                timeout=90,
            )
            if r.status_code == 200 and len(r.content) > 1000:
                return _row(OK, "voice", f"groq orpheus ({len(r.content)//1024} KB clip, 100 req/day free)")
            return _row(BAD, "voice", f"orpheus returned {r.status_code}: {r.text[:80]}")
        except Exception as exc:  # noqa: BLE001
            return _row(BAD, "voice", f"groq unreachable: {str(exc)[:80]}")
    return _row(WARN, "voice", "edge-tts fallback - flat delivery, no pauses or emphasis. "
                               "Add GROQ_API_KEY for directed performance.")


def check_render() -> tuple[str, str, str]:
    try:
        import subprocess
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        filters = subprocess.run([exe, "-hide_banner", "-filters"],
                                 capture_output=True, text=True, timeout=30).stdout
        needed = ["ass", "scale", "crop", "loudnorm", "tpad"]
        missing = [f for f in needed if f" {f} " not in filters]
        if missing:
            return _row(BAD, "render", f"ffmpeg is missing filters: {missing}")
        ver = subprocess.run([exe, "-version"], capture_output=True, text=True,
                             timeout=30).stdout.splitlines()[0].split()[2]
        return _row(OK, "render", f"static ffmpeg {ver} with libass")
    except Exception as exc:  # noqa: BLE001
        return _row(BAD, "render", f"ffmpeg unavailable: {str(exc)[:80]}")


def check_font() -> tuple[str, str, str]:
    from .fonts import ensure_font
    family, d = ensure_font()
    if family == "Anton":
        return _row(OK, "captions", "Anton available")
    return _row(WARN, "captions", f"falling back to {family} - captions will look less punchy")


def check_upload(cfg: Config) -> tuple[str, str, str]:
    import os
    creds = Path(os.environ.get("YOUTUBE_CREDENTIALS_PATH", "credentials.json"))
    if not creds.exists():
        return _row(WARN, "upload", f"no {creds} - runs will build but not publish (--dry-run)")
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from .config import YOUTUBE_UPLOAD_SCOPE
        c = Credentials.from_authorized_user_file(str(creds), scopes=[YOUTUBE_UPLOAD_SCOPE])
        if c.expired and c.refresh_token:
            c.refresh(Request())
        if c.valid:
            return _row(OK, "upload", f"youtube token valid (privacy={cfg.privacy}, 6 uploads/day cap)")
        return _row(BAD, "upload", "youtube token is not valid")
    except Exception as exc:  # noqa: BLE001
        hint = (" - refresh token revoked; re-run the OAuth flow and publish your consent "
                "screen to Production") if "invalid_grant" in str(exc) else ""
        return _row(BAD, "upload", f"{str(exc)[:80]}{hint}")


def check_memory(cfg: Config) -> tuple[str, str, str]:
    from .store import Store
    s = Store(cfg.store_path)
    n = len(s.entries)
    if n == 0:
        return _row(WARN, "memory", "premise store empty - first run, nothing to dedupe against")
    return _row(OK, "memory", f"{n} past premises recorded, dedup active")


def run(cfg: Config) -> int:
    print("\nchecking providers\n")
    rows = [
        check_llm(cfg),
        check_images(cfg),
        check_voice(cfg),
        check_render(),
        check_font(),
        check_memory(cfg),
        check_upload(cfg),
    ]
    bad = [r for r in rows if r[0] == BAD]
    warn = [r for r in rows if r[0] == WARN]

    print()
    blocking = [r for r in bad if r[1] in ("LLM", "images", "render")]
    if blocking:
        print(f"cannot run: {', '.join(r[1] for r in blocking)}")
        print("see docs/SETUP.md")
        return 1
    if warn:
        print(f"will run, degraded: {', '.join(r[1] for r in warn)}")
        print("see docs/SETUP.md to unlock the good paths")
    else:
        print("all systems go")
    return 0
