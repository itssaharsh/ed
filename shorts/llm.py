"""LLM access with a provider ladder that ends in a keyless floor.

The old pipeline hardcoded `gemini-1.5-flash` / `gemini-1.5-pro`, which now 404. Every model
failed, the pipeline swallowed it, and uploaded a video built from a placeholder. Two rules
follow from that:

  1. Model ids live in config.py, never inline, and a 404 demotes a provider immediately
     instead of being retried.
  2. `complete_json` raises when it cannot produce valid JSON. Callers must not paper over it.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import (
    JUDGE_MAX_TOKENS, ROLE_ORDER,
    GEMINI_MAX_OUTPUT_TOKENS, GEMINI_THINKING_BUDGET,
    LLM_MAX_OUTPUT_TOKENS,
    GEMINI_MODELS, GROQ_MODELS, OPENROUTER_MODEL, POLLINATIONS_TEXT_MODEL, Config, logger,
)


class LLMError(RuntimeError):
    pass


@dataclass
class Provider:
    name: str
    call: Any
    healthy: bool = True


# ── JSON extraction ─────────────────────────────────────────────────────────

def extract_json(raw: str) -> Any:
    """Pull a JSON value out of a model response.

    Handles: clean JSON, ```json fences, and prose wrapped around an object. Deliberately does
    NOT attempt to repair malformed JSON — a silent bad parse is worse than a retry.
    """
    if not raw or not raw.strip():
        raise LLMError("empty response")
    text = raw.strip()

    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"no parseable JSON in response: {text[:300]!r}")


# ── Providers ───────────────────────────────────────────────────────────────

class TruncatedError(LLMError):
    """The provider stopped at its token ceiling. Not a formatting problem - retrying the same
    provider with a "please output valid JSON" hint cannot fix it."""


def _raise_if_truncated(model: str, resp: Any) -> None:
    """Name a max-tokens stop as what it is.

    Without this, a truncated JSON response surfaced as "no parseable JSON in response" with a
    300-character preview that cannot show where the text ended - which is exactly what the first
    live CI run reported, three times, before dying.
    """
    try:
        cand = resp.candidates[0]
        reason = str(getattr(cand, "finish_reason", "") or "")
    except (AttributeError, IndexError, TypeError):
        return
    if "MAX_TOKENS" in reason:
        um = getattr(resp, "usage_metadata", None)
        thoughts = getattr(um, "thoughts_token_count", None)
        out = getattr(um, "candidates_token_count", None)
        raise TruncatedError(
            f"{model} hit max_output_tokens (thinking={thoughts}, answer={out}); "
            f"raise GEMINI_MAX_OUTPUT_TOKENS or lower GEMINI_THINKING_BUDGET"
        )


class RateLimitedError(LLMError):
    """A provider refused on rate or quota grounds. Carries how long to wait, if it said."""

    def __init__(self, msg: str, *, wait: float | None = None, daily: bool = False):
        super().__init__(msg)
        self.wait = wait
        self.daily = daily


# Per-model cool-downs (model -> monotonic time it may be tried again). Not a run-long ban: in CI
# on 2026-09-19 Gemini refused with a "per day" quota at 14:00:07 and answered normally at 14:05,
# so a model marked spent for the whole run throws away capacity that comes back mid-run.
_COOLDOWN: dict[str, float] = {}
DAILY_COOLDOWN = 600.0          # seconds to skip a model after a per-day refusal
MAX_WAIT = 70.0                 # longest single wait for a per-minute window to clear


def _cooling(model: str) -> bool:
    return _COOLDOWN.get(model, 0.0) > time.monotonic()


def _cool(model: str, seconds: float) -> None:
    _COOLDOWN[model] = time.monotonic() + seconds


def _is_rate_limit(msg: str) -> bool:
    m = msg.lower()
    return "429" in m or "resource_exhausted" in m or "rate limit" in m or "quota" in m


def _is_daily(msg: str) -> bool:
    m = msg.lower().replace("_", "").replace("-", "").replace(" ", "")
    return "perday" in m or "(rpd)" in m or "(tpd)" in m


def _parse_wait(text: str) -> float | None:
    """Seconds to wait, from "try again in 1m2.5s" / "8.3s" / "450ms" / "retryDelay: '23s'"."""
    # (?!s) so the "m" of "ms" is never read as minutes - "450ms" once parsed as 7.5 hours.
    m = re.search(r"try again in\s+(?:(\d+)m(?!s))?\s*([\d.]+)?\s*(ms|s)?", text, re.I)
    if m and (m.group(1) or m.group(2)):
        mins = float(m.group(1) or 0)
        val = float(m.group(2) or 0)
        return mins * 60 + (val / 1000 if (m.group(3) or "").lower() == "ms" else val)
    m = re.search(r"retry(?:Delay|\s+in)['\"]?\s*[:=]?\s*['\"]?([\d.]+)\s*s", text, re.I)
    return float(m.group(1)) if m else None


def _gemini(cfg: Config, prompt: str, *, temperature: float, want_json: bool,
            max_tokens: int | None = None) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cfg.gemini_key)
    safety = [
        types.SafetySetting(category=c, threshold="BLOCK_NONE")
        for c in (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        )
    ]
    last: Exception | None = None
    limited: list[RateLimitedError] = []
    other: Exception | None = None      # any failure that is not a rate limit
    for model in GEMINI_MODELS:
        if _cooling(model):
            continue
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=0.95,
                    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET),
                    response_mime_type="application/json" if want_json else "text/plain",
                    safety_settings=safety,
                ),
            )
            _raise_if_truncated(model, resp)
            if resp.text:
                return resp.text
            last = other = LLMError(f"{model} returned no text")
        except TruncatedError:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc)
            if "404" in msg or "NOT_FOUND" in msg:
                logger.warning("gemini model %s is gone (404) — check config.GEMINI_MODELS", model)
                other = exc
                continue
            if _is_rate_limit(msg):
                # Quotas are per model, so the next model has its own allowance.
                daily, wait = _is_daily(msg), _parse_wait(msg)
                _cool(model, DAILY_COOLDOWN if daily else (wait or 20.0))
                logger.warning("gemini %s %s; cooling %.0fs", model,
                               "daily quota refused" if daily else "rate limited",
                               DAILY_COOLDOWN if daily else (wait or 20.0))
                limited.append(RateLimitedError(msg[:160], wait=wait, daily=daily))
                continue
            raise
    if other is None:
        # Every model was either rate limited just now or still cooling down from earlier.
        # Report it as a rate limit so the ladder waits or moves on, instead of treating the
        # provider as broken.
        waits = [e.wait for e in limited if e.wait]
        raise RateLimitedError(
            f"all gemini models rate limited or cooling: {limited[-1] if limited else 'cooling'}",
            wait=min(waits) if waits else None,
            daily=(not limited) or all(e.daily for e in limited),
        )
    raise LLMError(f"all gemini models failed: {other}")


def _openai_compatible(url: str, key: str, model: str, prompt: str, *,
                       temperature: float, want_json: bool, extra_headers: dict | None = None,
                       max_tokens: int | None = None) -> str:
    cap = max_tokens or LLM_MAX_OUTPUT_TOKENS
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": cap,
    }
    if want_json:
        body["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model:
        # Reasoning model, same trap as Gemini thinking: reasoning tokens come out of max_tokens.
        body["reasoning_effort"] = "low"
    elif "qwen" in model:
        # Qwen3 on Groq otherwise emits its reasoning inside the content as <think>...</think>,
        # ahead of the JSON - which extract_json would then have to dig through.
        body["reasoning_format"] = "hidden"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    headers.update(extra_headers or {})
    r = requests.post(url, headers=headers, json=body, timeout=120)
    if r.status_code == 429:
        # Read the wait from the provider, not a guess. Groq's message runs past 200 characters,
        # so the old truncated error string cut off the "try again in Ns" it carried.
        text = r.text or ""
        wait = None
        try:
            wait = float(r.headers.get("retry-after", ""))
        except (TypeError, ValueError):
            wait = _parse_wait(text)
        raise RateLimitedError(f"{model} 429: {text[:160]}", wait=wait, daily=_is_daily(text))
    if r.status_code != 200:
        raise LLMError(f"{url} -> {r.status_code}: {r.text[:200]}")
    choice = r.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        raise TruncatedError(f"{model} hit max_tokens ({cap}) before finishing")
    return choice["message"]["content"] or ""


def _openrouter(cfg: Config, prompt: str, *, temperature: float, want_json: bool,
                max_tokens: int | None = None) -> str:
    return _openai_compatible(
        "https://openrouter.ai/api/v1/chat/completions", cfg.openrouter_key, OPENROUTER_MODEL,
        prompt, temperature=temperature, want_json=want_json, max_tokens=max_tokens,
        extra_headers={"HTTP-Referer": "https://github.com/", "X-Title": "shorts-pipeline"},
    )


def _groq(cfg: Config, prompt: str, *, temperature: float, want_json: bool,
          max_tokens: int | None = None) -> str:
    """Try each Groq model in turn. Limits are per model, so one model's 429 is not Groq's."""
    last: Exception | None = None
    limited: list[RateLimitedError] = []
    for model in GROQ_MODELS:
        if _cooling(model):
            continue
        try:
            return _openai_compatible(
                "https://api.groq.com/openai/v1/chat/completions", cfg.groq_key, model,
                prompt, temperature=temperature, want_json=want_json, max_tokens=max_tokens,
            )
        except RateLimitedError as exc:
            # Checked first, and by type. A string test for "400" also matches Groq's own 429
            # text ("Limit 8000, Used 7400"), which would misfile a one-minute wait as a
            # permanently broken model.
            _cool(model, DAILY_COOLDOWN if exc.daily else (exc.wait or 20.0))
            logger.warning("groq %s rate limited (%s); cooling %.0fs", model,
                           "daily" if exc.daily else "per-minute",
                           DAILY_COOLDOWN if exc.daily else (exc.wait or 20.0))
            limited.append(exc)
            last = exc
        except TruncatedError as exc:
            logger.warning("groq %s truncated; trying the next model", model)
            last = exc
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc).lower()
            # Match the status code as _openai_compatible formats it ("-> 404: ..."), not as a
            # bare number that can appear anywhere in a response body.
            if any(f"-> {code}:" in msg for code in ("404", "400", "413")) or \
                    "decommissioned" in msg or "not supported" in msg:
                logger.warning("groq model %s unavailable (%s)", model, str(exc)[:120])
                if "-> 404:" in msg:
                    _cool(model, 3600.0)        # it does not exist; stop asking this run
                continue
            raise
    if limited:
        # At least one model is only rate limited: that is a wait, not a failure. This is the
        # case the CI run of 2026-09-19 got wrong - two 429s and a 404, where the 404 came last
        # and hid the rate limit, so nothing waited and the stage gave up in four seconds.
        waits = [e.wait for e in limited if e.wait]
        raise RateLimitedError(f"all groq models rate limited or unavailable: {last}",
                               wait=min(waits) if waits else None,
                               daily=all(e.daily for e in limited))
    if last is None:
        raise RateLimitedError("all groq models cooling down", wait=None, daily=False)
    raise LLMError(f"all groq models failed: {last}")


def _pollinations(cfg: Config, prompt: str, *, temperature: float, want_json: bool,
                  max_tokens: int | None = None) -> str:
    """Keyless floor - the rung that lets the pipeline run with no API keys at all.

    Two live quirks, both confirmed by probing:

      * The old `text.pollinations.ai/openai` endpoint now answers 402 with a deprecation
        notice. The working OpenAI-compatible endpoint is on gen.pollinations.ai.
      * The anonymous tier accepts ONLY `model` plus a single user message. Adding
        `temperature`, `max_tokens`, a system message, or `response_format` returns 401
        "A valid API key is required". So with no token we send the bare minimum and fold the
        JSON instruction into the user content instead.
    """
    headers = {"Content-Type": "application/json"}
    body: dict[str, Any]

    if cfg.pollinations_token:
        headers["Authorization"] = f"Bearer {cfg.pollinations_token}"
        body = {
            "model": POLLINATIONS_TEXT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if want_json:
            body["response_format"] = {"type": "json_object"}
    else:
        content = prompt
        if want_json:
            content += ("\n\nRespond with raw JSON only. No prose, no markdown fences. "
                        "The first character must be { or [.")
        body = {
            "model": POLLINATIONS_TEXT_MODEL,
            "messages": [{"role": "user", "content": content}],
        }

    r = requests.post("https://gen.pollinations.ai/v1/chat/completions",
                      headers=headers, json=body, timeout=180)
    if r.status_code != 200:
        raise LLMError(f"pollinations -> {r.status_code}: {r.text[:200]}")
    try:
        return r.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, ValueError, IndexError):
        return r.text


class LLM:
    """Provider ladder. Demotes a provider on failure, retries the next one."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.providers: list[Provider] = []
        if cfg.gemini_key:
            self.providers.append(Provider("gemini", _gemini))
        if cfg.openrouter_key:
            self.providers.append(Provider("openrouter", _openrouter))
        if cfg.groq_key:
            self.providers.append(Provider("groq", _groq))
        if cfg.pollinations_token:
            # Keyless Pollinations text is gone: probed 2026-09-12, gen.pollinations.ai returns
            # 401 "A valid API key is required" on the *first* anonymous request, not after a
            # handful. Keeping it in the ladder bought nothing and cost two 120s timeouts on
            # every total-failure path. CLAUDE.md already says there is no working keyless text
            # tier; this makes the ladder agree with it.
            self.providers.append(Provider("pollinations", _pollinations))
        self.calls = 0
        self.last_provider: str | None = None

    def _ordered(self, role: str) -> list[Provider]:
        """Providers in this role's preference order (config.ROLE_ORDER), unknown names last."""
        order = ROLE_ORDER.get(role, ROLE_ORDER["generate"])
        rank = {name: i for i, name in enumerate(order)}
        return sorted(self.providers, key=lambda p: rank.get(p.name, len(order)))

    def complete(self, prompt: str, *, temperature: float = 0.9, want_json: bool = True,
                 exclude: frozenset[str] = frozenset(), role: str = "generate") -> str:
        """First healthy provider for `role` not in `exclude`. Records it in last_provider.

        role="generate" prefers the best writer; role="judge" prefers the provider with the
        most requests to spare and caps output at JUDGE_MAX_TOKENS. See config.ROLE_ORDER.
        """
        self.last_provider = None
        max_tokens = JUDGE_MAX_TOKENS if role == "judge" else None
        if not any(p.healthy for p in self.providers):
            raise LLMError(
                "every LLM provider is exhausted or unreachable: "
                + ", ".join(p.name for p in self.providers)
                + ". Add a key (see docs/SETUP.md) or wait for the rate limit to reset."
            )
        errors: list[str] = []
        # Passes over the whole ladder. A rate limit on one provider moves straight on to the
        # next - waiting on a limited provider while another is free was the old behaviour. Only
        # when *every* provider is rate limited does it sleep, and then for the shortest wait any
        # of them reported, up to MAX_WAIT. Per-minute limits never demote a provider: the old
        # code demoted after two, which on 2026-09-19 took the best writer out of the run for
        # good one minute in, over a window that clears in sixty seconds.
        for rnd in range(1, 4):
            waits: list[float] = []
            for p in self._ordered(role):
                if not p.healthy or p.name in exclude:
                    continue
                try:
                    self.calls += 1
                    out = p.call(self.cfg, prompt, temperature=temperature, want_json=want_json,
                                 max_tokens=max_tokens)
                    if out and out.strip():
                        self.last_provider = p.name
                        return out
                    errors.append(f"{p.name}: empty")
                except RateLimitedError as exc:
                    errors.append(f"{p.name}: {str(exc)[:140]}")
                    if not exc.daily:
                        waits.append(exc.wait or 20.0)
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    errors.append(f"{p.name}: {msg[:140]}")
                    if _is_rate_limit(msg):      # providers without a structured 429
                        waits.append(20.0)
            if not waits or rnd == 3:
                break
            wait = min(max(min(waits), 1.0), MAX_WAIT) + random.uniform(0, 2)
            logger.warning("every %s provider is rate limited; waiting %.0fs (pass %d/3)",
                           role, wait, rnd)
            time.sleep(wait)
        raise LLMError("all providers failed:\n  " + "\n  ".join(errors[-6:]))

    def complete_json(self, prompt: str, *, temperature: float = 0.9, retries: int = 2,
                      role: str = "generate") -> Any:
        """Complete and parse JSON. Raises LLMError rather than returning a placeholder."""
        last: Exception | None = None
        # Providers that already returned something unusable for *this* prompt. The retry goes to
        # a different provider rather than straight back to the same one: the first live CI run
        # sent all three attempts to Gemini, which truncated identically each time, while Groq
        # sat configured and idle. A parse failure never demotes a provider globally - the same
        # model may be fine on the next, shorter prompt - it only steers this prompt's retries.
        tried_bad: set[str] = set()
        for attempt in range(retries + 1):
            p = prompt if attempt == 0 else (
                prompt + "\n\nOutput valid JSON only. No prose, no markdown fence. "
                "The first character must be { or [."
            )
            healthy = {pr.name for pr in self.providers if pr.healthy}
            exclude = frozenset(tried_bad) if healthy - tried_bad else frozenset()
            try:
                return extract_json(self.complete(p, temperature=temperature, want_json=True,
                                                  exclude=exclude, role=role))
            except LLMError as exc:
                last = exc
                if self.last_provider:
                    tried_bad.add(self.last_provider)
                logger.warning("json parse failed (attempt %d/%d, %s): %s", attempt + 1,
                               retries + 1, self.last_provider or "?", exc)
                time.sleep(1.5)
        raise LLMError(f"could not obtain valid JSON after {retries + 1} attempts: {last}")
