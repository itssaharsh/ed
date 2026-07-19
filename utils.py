import re
from typing import Any

def _normalize_text(value: Any, *, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    val = value.strip()
    return val[:max_length] if len(val) > max_length else val

def _normalize_search_query(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    val = value.lower().strip()
    val = re.sub(r"[^a-z0-9\s]", "", val)
    val = re.sub(r"\s+", " ", val).strip()
    return val

def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

def _ensure_shorts_hashtag(description: str) -> str:
    desc = description.strip()
    if "#shorts" not in desc.lower():
        if desc and not desc.endswith((".", "!", "?")):
            desc += "."
        desc += "\n#shorts"
    return desc

def _ensure_shorts_tag(title: str) -> str:
    t = title.strip()
    if "#shorts" not in t.lower():
        t += " #shorts"
    return t
