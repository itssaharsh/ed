"""Load prompt files and substitute {{VARIABLES}}.

Prompts live in prompts/*.md as prose. Each file has a human-facing header, then a `---`
separator, then the body that is actually sent to the model. A missing variable raises rather
than silently rendering empty — a half-substituted prompt produces plausible garbage, which is
the worst failure mode in a generative pipeline.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .config import ROOT, logger

PROMPTS_DIR = ROOT / "prompts"
_VAR = re.compile(r"\{\{([A-Z_]+)\}\}")


class PromptError(RuntimeError):
    pass


@lru_cache(maxsize=32)
def _raw(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise PromptError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _body(name: str) -> str:
    """The part of the file after the first '---' line — the header is documentation for humans."""
    text = _raw(name)
    parts = re.split(r"^---\s*$", text, maxsplit=1, flags=re.MULTILINE)
    return (parts[1] if len(parts) > 1 else text).strip()


@lru_cache(maxsize=1)
def voice_block() -> str:
    """The shared comedian persona, injected into writing prompts as {{VOICE}}.

    Deliberately NOT injected into judging prompts: a judge carrying the persona rates its own
    style highly, which collapses the tournament.
    """
    return _body("00_voice")


def render(name: str, **vars: object) -> str:
    """Render a prompt, substituting {{VARS}}. Raises on any unfilled variable."""
    body = _body(name)
    supplied = {k: ("" if v is None else str(v)) for k, v in vars.items()}
    supplied.setdefault("VOICE", voice_block())

    missing: list[str] = []

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in supplied:
            missing.append(key)
            return m.group(0)
        return supplied[key]

    out = _VAR.sub(sub, body)
    if missing:
        raise PromptError(f"{name}.md: unfilled variables {sorted(set(missing))}")
    return out


def style_contract(style: str) -> tuple[str, str]:
    """Extract (contract, negative_prompt) for a style from 07_styles.md.

    Parsed out of the markdown so the prompt file stays the single source of truth — editing the
    style there changes generation without touching Python.
    """
    text = _raw("07_styles")
    block = re.search(
        rf"^## `{re.escape(style)}`.*?\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL
    )
    if not block:
        raise PromptError(f"style '{style}' not found in 07_styles.md")
    section = block.group(1)

    quoted = re.findall(r"^> (.*)$", section, re.MULTILINE)
    if not quoted:
        raise PromptError(f"style '{style}' has no '>' contract paragraph")
    contract = " ".join(q.strip() for q in quoted)

    neg_m = re.search(r"\*\*Negative:\*\*\s*(.+?)(?=\n- \*\*|\n\n|\Z)", section, re.DOTALL)
    negative = " ".join(neg_m.group(1).split()) if neg_m else ""

    uni = re.search(r"## Universal negative prompt\n+(?:.*?\n)*?> (.+?)(?=\n\n)", text, re.DOTALL)
    if uni:
        negative = f"{negative} {' '.join(uni.group(1).split())}".strip()
    return contract, negative


def style_suffix(style: str) -> str:
    """The SHORT style tag that is actually appended to an image prompt.

    Distinct from `style_contract`, which is the long descriptive paragraph shown to the
    shot-list LLM as art direction. Weak image models weight a long style preamble so heavily
    that the subject gets ignored entirely - a request for "a man at an office fridge" came back
    as a portrait of a stranger. So generation uses subject-first ordering with this compact tag
    appended, while the long contract stays as context for the writing stage.
    """
    text = _raw("07_styles")
    block = re.search(
        rf"^## `{re.escape(style)}`.*?\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL
    )
    if not block:
        raise PromptError(f"style '{style}' not found in 07_styles.md")
    m = re.search(r"\*\*Prompt suffix:\*\*\s*`([^`]+)`", block.group(1))
    if not m:
        raise PromptError(f"style '{style}' has no **Prompt suffix:** line")
    return m.group(1).strip()


def style_palette(style: str) -> str:
    text = _raw("07_styles")
    block = re.search(
        rf"^## `{re.escape(style)}`.*?\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL
    )
    if not block:
        return ""
    m = re.search(r"\*\*Palette:\*\*\s*(.+)", block.group(1))
    return m.group(1).strip() if m else ""


def available_styles() -> list[str]:
    return re.findall(r"^## `([a-z_]+)`", _raw("07_styles"), re.MULTILINE)
