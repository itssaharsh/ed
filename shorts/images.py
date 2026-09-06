"""Image generation: Cloudflare FLUX-schnell -> Pollinations (token) -> Pollinations (keyless).

Every ladder rung ends somewhere that needs no API key, so the pipeline can always produce
frames. Quality degrades; availability does not.

Licence note: FLUX.1-*schnell* is Apache-2.0 and fine for monetised video. FLUX.1-*dev* is
non-commercial. Do not swap the model id without checking that.
"""
from __future__ import annotations

import base64
import io
import random
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .config import (
    CF_IMAGE_MODEL, IMAGE_H, IMAGE_STEPS, IMAGE_W, MASTER_H, MASTER_W, Config, logger,
)

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/"


class ImageError(RuntimeError):
    pass


def _cloudflare(cfg: Config, prompt: str, negative: str, seed: int) -> bytes:
    """Cloudflare Workers AI. Free tier is 10,000 neurons/day, about 145 portrait images.

    Two shapes are tolerated deliberately:

      * The response may be raw image bytes OR a JSON envelope with a base64 `image` field,
        depending on the model and the gateway.
      * Cloudflare's own docs disagree about whether `flux-1-schnell` accepts `width`/`height`
        (one page documents 256-1920, the model schema page omits them). Rather than guess, we
        ask for 9:16 and, if the request is rejected for the parameters, retry without them and
        let `_to_master` cover-crop the square. That way either behaviour works.
    """
    url = (f"https://api.cloudflare.com/client/v4/accounts/{cfg.cf_account}"
           f"/ai/run/{CF_IMAGE_MODEL}")
    headers = {"Authorization": f"Bearer {cfg.cf_token}"}

    def call(with_size: bool) -> requests.Response:
        body: dict[str, object] = {"prompt": prompt, "steps": IMAGE_STEPS, "seed": seed}
        if with_size:
            body["width"], body["height"] = IMAGE_W, IMAGE_H
        if negative:
            body["negative_prompt"] = negative
        return requests.post(url, headers=headers, json=body, timeout=120)

    r = call(with_size=True)
    if r.status_code == 400:
        logger.info("cloudflare rejected explicit width/height; retrying at the model default")
        r = call(with_size=False)
    if r.status_code != 200:
        raise ImageError(f"cloudflare {r.status_code}: {r.text[:200]}")

    if r.headers.get("content-type", "").startswith("image/"):
        return r.content

    payload = r.json()
    if not payload.get("success", True):
        raise ImageError(f"cloudflare error: {str(payload.get('errors'))[:200]}")
    result = payload.get("result") or {}
    b64 = result.get("image") if isinstance(result, dict) else None
    if not b64:
        raise ImageError(f"cloudflare returned no image: {str(payload)[:200]}")
    return base64.b64decode(b64)


# Pollinations' anonymous tier limits by IP, roughly one request every 15 seconds. Exceeding it
# returns 429, and because every retry also counts, hammering it makes throughput *worse* — three
# parallel workers produced one image in seven minutes, versus a steady one per ~20s when paced.
# This gate serialises and spaces every anonymous request across the whole process.
_POLLINATIONS_MIN_INTERVAL = 16.0
_pollinations_lock = threading.Lock()
_pollinations_last = 0.0


def _pollinations_wait() -> None:
    global _pollinations_last
    with _pollinations_lock:
        gap = time.monotonic() - _pollinations_last
        if gap < _POLLINATIONS_MIN_INTERVAL:
            time.sleep(_POLLINATIONS_MIN_INTERVAL - gap)
        _pollinations_last = time.monotonic()


def _pollinations(cfg: Config, prompt: str, negative: str, seed: int) -> bytes:
    """Keyless floor. Anonymous tier serves only `sana` at ~580x1015 and ignores seed."""
    params = {"width": IMAGE_W, "height": IMAGE_H, "seed": seed,
              "nologo": "true", "private": "true", "safe": "false"}
    if negative:
        params["negative_prompt"] = negative
    if cfg.pollinations_token:
        params["model"] = "flux"
    url = POLLINATIONS_URL + urllib.parse.quote(prompt[:1800], safe="")
    headers = {"Authorization": f"Bearer {cfg.pollinations_token}"} if cfg.pollinations_token else {}
    if not cfg.pollinations_token:
        _pollinations_wait()
    r = requests.get(url, params=params, headers=headers, timeout=180)
    if r.status_code == 429:
        # Back off past the whole window; a fast retry just spends another slot on a refusal.
        raise ImageError("pollinations 429 (rate limited) - pacing gate will widen")
    if r.status_code != 200:
        raise ImageError(f"pollinations {r.status_code}: {r.text[:160]}")
    if not r.headers.get("content-type", "").startswith("image/"):
        raise ImageError("pollinations returned non-image")
    return r.content


def inspect(data: bytes) -> dict:
    """Cheap structural checks on a generated image.

    Free image models fail in two visible ways: a near-flat colour field, or a smooth abstract
    banding pattern with no subject in it. Both are obvious to a person and invisible to a
    pipeline that only checks the HTTP status.

    Two statistics catch both:

      * `stdev` - overall contrast. A near-uniform frame has almost none.
      * `directionality` - min(rowdiff/coldiff, coldiff/rowdiff). A real photograph or
        illustration varies in both axes, so the ratio sits near 1. A 1-D gradient or stripe
        field varies in only one axis and the ratio collapses toward 0. Edge density does NOT
        work here: hard stripe boundaries score *higher* than a real subject.
    """
    from PIL import Image, ImageStat
    import numpy as np

    img = Image.open(io.BytesIO(data)).convert("RGB")
    small = img.resize((128, 224), Image.BILINEAR)
    grey = np.asarray(small.convert("L"), dtype=np.float32)

    rowdiff = float(np.abs(np.diff(grey, axis=0)).mean())
    coldiff = float(np.abs(np.diff(grey, axis=1)).mean())
    if rowdiff <= 1e-6 or coldiff <= 1e-6:
        directionality = 0.0
    else:
        directionality = min(rowdiff / coldiff, coldiff / rowdiff)

    return {
        "stdev": round(ImageStat.Stat(small.convert("L")).stddev[0], 2),
        "directionality": round(directionality, 4),
        "rowdiff": round(rowdiff, 3),
        "coldiff": round(coldiff, 3),
        "width": img.width,
        "height": img.height,
    }


# Calibrated on real output from the keyless provider: an abstract vertical-band frame scored
# directionality 0.062, while four usable shots scored 0.535-0.809.
MIN_STDEV = 10.0
MIN_DIRECTIONALITY = 0.18


def is_usable(stats: dict) -> tuple[bool, str]:
    if stats["stdev"] < MIN_STDEV:
        return False, f"near-flat image (stdev {stats['stdev']} < {MIN_STDEV})"
    if stats["directionality"] < MIN_DIRECTIONALITY:
        return False, (f"abstract gradient, no subject "
                       f"(directionality {stats['directionality']} < {MIN_DIRECTIONALITY})")
    return True, "ok"


def _to_master(data: bytes, out_path: Path) -> Path:
    """Normalise to the 1296x2304 master used by the renderer.

    That is 1.2x the 1080x1920 output, which is the headroom the camera move zooms and pans into.
    Cover-crop, never letterbox — a black bar in a Short is fatal.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    scale = max(MASTER_W / img.width, MASTER_H / img.height)
    new = (max(MASTER_W, int(img.width * scale + 0.5)), max(MASTER_H, int(img.height * scale + 0.5)))
    img = img.resize(new, Image.LANCZOS)
    left, top = (img.width - MASTER_W) // 2, (img.height - MASTER_H) // 2
    img = img.crop((left, top, left + MASTER_W, top + MASTER_H))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def generate_one(cfg: Config, prompt: str, negative: str, seed: int, out_path: Path) -> tuple[Path, str]:
    """Try each provider in turn. Returns (path, provider_used)."""
    attempts: list[tuple[str, object]] = []
    if cfg.cf_account and cfg.cf_token:
        attempts.append(("cloudflare", _cloudflare))
    attempts.append(("pollinations", _pollinations))

    errors: list[str] = []
    for name, fn in attempts:
        for attempt in (1, 2):
            try:
                # Vary the seed per attempt: re-requesting a failed generation with the same
                # seed usually reproduces the same failed image.
                data = fn(cfg, prompt, negative, seed + (attempt - 1) * 10_007)
                if len(data) < 2000:
                    raise ImageError(f"suspiciously small payload ({len(data)} bytes)")
                stats = inspect(data)
                ok, reason = is_usable(stats)
                if not ok:
                    raise ImageError(reason)
                _to_master(data, out_path)
                return out_path, name
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}#{attempt}: {str(exc)[:120]}")
                # A rate-limit refusal needs to clear the provider's whole window; anything
                # shorter spends the next slot on another refusal.
                rate_limited = "429" in str(exc) or "rate limit" in str(exc).lower()
                time.sleep((_POLLINATIONS_MIN_INTERVAL if rate_limited else 2.0 * attempt)
                           + random.uniform(0, 1.5))
    raise ImageError("all image providers failed:\n  " + "\n  ".join(errors))


def generate_all(cfg: Config, shots: list[dict], style_contract: str, negative: str,
                 out_dir: Path, seed_base: int) -> list[dict]:
    """Generate every shot's image.

    Concurrency is deliberately low: the free tiers rate-limit hard (Pollinations anonymous is one
    request per 15s), and hammering them turns a slow run into a failed one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = 3 if (cfg.cf_account and cfg.cf_token) else 1

    # Circuit breaker. When a free tier goes down or hits its ceiling, every shot fails the same
    # way, and each one burns two attempts with timeouts and backoff. On a rate-limited provider
    # that is minutes per shot - enough to eat a whole CI job before the run gives up. After this
    # many consecutive total failures, stop trying and let the caller abort with a clear reason.
    breaker_limit = max(3, len(shots) // 2)
    state = {"consecutive_failures": 0, "tripped": False}
    lock = threading.Lock()

    def build(i_shot: tuple[int, dict]) -> dict:
        i, shot = i_shot
        with lock:
            if state["tripped"]:
                return {**shot, "image": None, "provider": None, "seed": 0, "ok": False,
                        "skipped": True}
        # Subject first, then a SHORT style tag. Leading with a long style paragraph makes weak
        # models drop the subject entirely (a request for "a man at an office fridge" came back
        # as a portrait of a stranger); leading with the subject keeps it in the frame.
        full = f"{shot['prompt'].strip().rstrip('.')}. {style_contract.strip()}"
        path = out_dir / f"shot_{i:02d}.png"
        # Seed derived from run seed + index: reproducible, and distinct per shot so the model
        # does not return near-identical frames.
        seed = (seed_base * 7919 + i * 104729) % 2_000_000_000
        try:
            p, provider = generate_one(cfg, full, negative, seed, path)
            with lock:
                state["consecutive_failures"] = 0
            return {**shot, "image": str(p), "provider": provider, "seed": seed, "ok": True}
        except ImageError as exc:
            logger.error("shot %d image failed: %s", i, exc)
            with lock:
                state["consecutive_failures"] += 1
                if state["consecutive_failures"] >= breaker_limit and not state["tripped"]:
                    state["tripped"] = True
                    logger.error(
                        "image generation circuit breaker tripped after %d consecutive failures "
                        "- the provider is down or out of quota. Abandoning the remaining shots.",
                        breaker_limit,
                    )
            return {**shot, "image": None, "provider": None, "seed": seed, "ok": False}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(build, enumerate(shots)))

    # Trust the filesystem, not the return value. The renderer depends on these files existing,
    # and a truncated or vanished write here surfaces as a confusing ffmpeg error three stages
    # later. Verify now, while we still know which shot it was.
    for r in results:
        if r["ok"]:
            path = Path(r["image"]) if r["image"] else None
            if path is None or not path.exists() or path.stat().st_size < 10_000:
                logger.error("shot image missing or truncated after generation: %s", path)
                r["ok"], r["image"] = False, None

    ok = sum(1 for r in results if r["ok"])
    skipped = sum(1 for r in results if r.get("skipped"))
    logger.info("stage 7: %d/%d images generated%s (%s)", ok, len(results),
                f", {skipped} skipped after the breaker tripped" if skipped else "",
                ", ".join(sorted({r["provider"] for r in results if r["provider"]}) or ["none"]))
    return results
