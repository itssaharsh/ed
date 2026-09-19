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


def _gemini(cfg: Config, prompt: str, *, temperature: float, want_json: bool) -> str:
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
    for model in GEMINI_MODELS:
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
            last = LLMError(f"{model} returned no text")
        except Exception as exc:  # noqa: BLE001
            last = exc
            if "404" in str(exc) or "NOT_FOUND" in str(exc):
                logger.warning("gemini model %s is gone (404) — check config.GEMINI_MODELS", model)
                continue
            raise
    raise LLMError(f"all gemini models failed: {last}")


def _openai_compatible(url: str, key: str, model: str, prompt: str, *,
                       temperature: float, want_json: bool, extra_headers: dict | None = None) -> str:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": LLM_MAX_OUTPUT_TOKENS,
    }
    if want_json:
        body["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model:
        # Reasoning model, same trap as Gemini thinking: reasoning tokens come out of max_tokens.
        body["reasoning_effort"] = "low"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    headers.update(extra_headers or {})
    r = requests.post(url, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        raise LLMError(f"{url} -> {r.status_code}: {r.text[:200]}")
    choice = r.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        raise TruncatedError(f"{model} hit max_tokens ({LLM_MAX_OUTPUT_TOKENS}) before finishing")
    return choice["message"]["content"] or ""


def _openrouter(cfg: Config, prompt: str, *, temperature: float, want_json: bool) -> str:
    return _openai_compatible(
        "https://openrouter.ai/api/v1/chat/completions", cfg.openrouter_key, OPENROUTER_MODEL,
        prompt, temperature=temperature, want_json=want_json,
        extra_headers={"HTTP-Referer": "https://github.com/", "X-Title": "shorts-pipeline"},
    )


def _groq(cfg: Config, prompt: str, *, temperature: float, want_json: bool) -> str:
    last: Exception | None = None
    for model in GROQ_MODELS:
        try:
            return _openai_compatible(
                "https://api.groq.com/openai/v1/chat/completions", cfg.groq_key, model,
                prompt, temperature=temperature, want_json=want_json,
            )
        except Exception as exc:  # noqa: BLE001
            last = exc
            # Advance to the next model rather than killing the whole rung. A 400 (unsupported
            # response_format) or 413 (prompt + max_tokens over the TPM ceiling) is a property
            # of *this model*, not of Groq - raising here took the provider down entirely and
            # dropped the run onto a tier that cannot serve it.
            msg = str(exc).lower()
            if any(t in msg for t in ("404", "400", "413", "decommissioned",
                                      "not supported", "request too large")):
                logger.warning("groq model %s unavailable (%s)", model, str(exc)[:120])
                continue
            raise
    raise LLMError(f"all groq models failed: {last}")


def _pollinations(cfg: Config, prompt: str, *, temperature: float, want_json: bool) -> str:
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

    def complete(self, prompt: str, *, temperature: float = 0.9, want_json: bool = True,
                 exclude: frozenset[str] = frozenset()) -> str:
        """First healthy provider not in `exclude`. Records which one answered in last_provider."""
        self.last_provider = None
        if not any(p.healthy for p in self.providers):
            raise LLMError(
                "every LLM provider is exhausted or unreachable: "
                + ", ".join(p.name for p in self.providers)
                + ". Add a key (see docs/SETUP.md) or wait for the rate limit to reset."
            )
        errors: list[str] = []
        for p in self.providers:
            if not p.healthy or p.name in exclude:
                continue
            for attempt in (1, 2):
                try:
                    self.calls += 1
                    out = p.call(self.cfg, prompt, temperature=temperature, want_json=want_json)
                    if out and out.strip():
                        self.last_provider = p.name
                        return out
                    errors.append(f"{p.name}: empty")
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    errors.append(f"{p.name}: {msg[:140]}")
                    if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
                        if attempt == 1:
                            wait = 20 * attempt + random.uniform(0, 5)
                            logger.warning("%s rate limited; waiting %.0fs", p.name, wait)
                            time.sleep(wait)
                            continue
                        logger.warning("%s exhausted — demoting", p.name)
                        p.healthy = False
                    break
        raise LLMError("all providers failed:\n  " + "\n  ".join(errors))

    def complete_json(self, prompt: str, *, temperature: float = 0.9, retries: int = 2) -> Any:
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
                                                  exclude=exclude))
            except LLMError as exc:
                last = exc
                if self.last_provider:
                    tried_bad.add(self.last_provider)
                logger.warning("json parse failed (attempt %d/%d, %s): %s", attempt + 1,
                               retries + 1, self.last_provider or "?", exc)
                time.sleep(1.5)
        raise LLMError(f"could not obtain valid JSON after {retries + 1} attempts: {last}")
