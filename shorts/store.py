"""Premise memory - the fix for "it keeps making the same joke".

The old pipeline had no state at all, so nothing stopped it re-deriving the same premise every
run. This is a JSONL file committed back to the repo: no database, no embedding service, no
network call. Similarity uses character 4-grams (Jaccard) plus content-word overlap, which is
enough to catch "the same joke wearing a hat" without needing a model.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import logger

SIMILARITY_THRESHOLD = 0.32
RECENT_FOR_PROMPT = 40

_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "on", "at", "for", "with", "his", "her", "their", "its", "he", "she", "they", "it",
    "you", "your", "who", "that", "this", "then", "than", "so", "as", "by", "from", "has", "have",
    "had", "not", "no", "one", "up", "out", "about", "into", "over", "after", "just", "like",
}


def _stem(w: str) -> str:
    """Crude suffix stripping. Enough to make microwave/microwaved/microwaving collide."""
    for suf in ("ingly", "edly", "ing", "ies", "ied", "ed", "es", "ly", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _words(text: str) -> set[str]:
    return {
        _stem(w) for w in re.findall(r"[a-z']+", text.lower())
        if len(w) > 2 and w not in _STOP
    }


def _grams(text: str, n: int = 4) -> set[str]:
    s = re.sub(r"[^a-z ]", "", text.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))} or {s}


def similarity(a: str, b: str) -> float:
    """How much two premises are "the same joke".

    Weighted toward the **overlap coefficient** of stemmed content words rather than Jaccard.
    Jaccard punishes a rewording for the words it does *not* share, which is exactly wrong here:
    two tellings of the same premise differ in filler and agree on the nouns that matter. The
    character-ngram term is kept as a smaller signal for near-verbatim repeats.
    """
    ga, gb = _grams(a), _grams(b)
    wa, wb = _words(a), _words(b)
    gram_j = len(ga & gb) / len(ga | gb) if (ga | gb) else 0.0
    if wa and wb:
        inter = len(wa & wb)
        word_overlap = inter / min(len(wa), len(wb))     # containment
        word_j = inter / len(wa | wb)
    else:
        word_overlap = word_j = 0.0
    return 0.20 * gram_j + 0.55 * word_overlap + 0.25 * word_j


@dataclass
class Entry:
    run_id: str
    ts: float
    category: str
    premise: str
    script: str
    title: str
    video_id: str | None = None
    published: bool = False

    @classmethod
    def from_json(cls, d: dict) -> "Entry":
        return cls(
            run_id=str(d.get("run_id", "")), ts=float(d.get("ts", 0)),
            category=str(d.get("category", "")), premise=str(d.get("premise", "")),
            script=str(d.get("script", "")), title=str(d.get("title", "")),
            video_id=d.get("video_id"), published=bool(d.get("published", False)),
        )


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.entries: list[Entry] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self.entries.append(Entry.from_json(json.loads(line)))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        logger.info("store: %d past premises loaded from %s", len(self.entries), path.name)

    def recent_premises(self, n: int = RECENT_FOR_PROMPT) -> list[str]:
        return [e.premise for e in sorted(self.entries, key=lambda e: -e.ts)[:n] if e.premise]

    def most_similar(self, premise: str) -> tuple[float, Entry | None]:
        best, who = 0.0, None
        for e in self.entries:
            if not e.premise:
                continue
            s = similarity(premise, e.premise)
            if s > best:
                best, who = s, e
        return best, who

    def is_duplicate(self, premise: str, threshold: float = SIMILARITY_THRESHOLD) -> tuple[bool, float, Entry | None]:
        score, entry = self.most_similar(premise)
        return score >= threshold, score, entry

    def seen_script(self, script: str) -> bool:
        norm = re.sub(r"\s+", " ", script.lower()).strip()
        return any(re.sub(r"\s+", " ", e.script.lower()).strip() == norm for e in self.entries)

    def filter_new(self, premises: list, key, threshold: float = SIMILARITY_THRESHOLD) -> list:
        """Drop candidates too close to history, and to each other."""
        kept: list = []
        for p in premises:
            text = key(p)
            dup, score, _ = self.is_duplicate(text, threshold)
            if dup:
                continue
            if any(similarity(text, key(k)) >= threshold for k in kept):
                continue
            kept.append(p)
        dropped = len(premises) - len(kept)
        if dropped:
            logger.info("store: dropped %d/%d premises as too similar", dropped, len(premises))
        return kept

    def append(self, entry: Entry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")
        self.entries.append(entry)
        logger.info("store: recorded run %s (%d total)", entry.run_id, len(self.entries))
