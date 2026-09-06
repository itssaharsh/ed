"""Pairwise comedy selection: Swiss pairing + Bradley-Terry.

Why not just ask the model to score each candidate 0-100 and take the max? Because that does not
work. In HumorRank's evaluation, absolute rubric scoring collapsed — 88.5% of structured scores
came out identical, with only a ~20-point spread across genuinely diverse candidates. Pairwise
judging on the same material reached cross-judge agreement of tau=0.889 and matched human-human
agreement on hard calls.

So the judge is only ever asked "which of these two, and why". Bradley-Terry then converts the
pairwise outcomes into a global ranking.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .config import logger


@dataclass
class Bout:
    a: int
    b: int
    winner: int
    criterion: str = ""
    why: str = ""
    confidence: str = "medium"


@dataclass
class Standing:
    index: int
    wins: int = 0
    losses: int = 0
    rating: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.wins - self.losses


def bradley_terry(n: int, bouts: Sequence[Bout], iters: int = 200) -> list[float]:
    """MLE strengths from pairwise outcomes, via minorisation-maximisation.

    Regularised with a virtual half-win to each side of every observed pair. Without it, a
    candidate that won all its bouts has an unbounded MLE (perfect separation) and the ratings
    saturate into meaningless ties — which is exactly what a small Swiss tournament produces.

    Returns log-strengths, mean-centred. Candidates with no bouts stay at 0.
    """
    if not bouts:
        return [0.0] * n

    wins = [0.5] * n                      # virtual half-win prior
    pairs: dict[tuple[int, int], int] = {}
    for bt in bouts:
        wins[bt.winner] += 1.0
        key = (min(bt.a, bt.b), max(bt.a, bt.b))
        pairs[key] = pairs.get(key, 0) + 1

    # One virtual tie per observed pair regularises both sides symmetrically.
    for (x, y), _ in pairs.items():
        wins[x] += 0.5
        wins[y] += 0.5
    counts = {k: v + 1 for k, v in pairs.items()}

    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for (x, y), cnt in counts.items():
        adj[x].append((y, cnt))
        adj[y].append((x, cnt))

    p = [1.0] * n
    for _ in range(iters):
        new = p[:]
        for i in range(n):
            if not adj[i]:
                continue
            denom = sum(cnt / (p[i] + p[j]) for j, cnt in adj[i])
            new[i] = wins[i] / denom if denom > 0 else p[i]
        total = sum(new) or 1.0
        p = [max(v / total * n, 1e-9) for v in new]

    logs = [math.log(v) for v in p]
    mean = sum(logs) / len(logs)
    return [v - mean for v in logs]


def run_tournament(
    candidates: Sequence[str],
    judge: Callable[[str, str], dict],
    *,
    rounds: int = 3,
    rng: random.Random | None = None,
) -> tuple[int, list[Standing], list[Bout]]:
    """Swiss-paired pairwise tournament.

    `judge(a_text, b_text)` returns {"winner": "A"|"B", "deciding_criterion", "why", "confidence"}.

    Each round pairs candidates with similar records (Swiss), which concentrates comparisons on
    close matchups instead of the O(n^2) full round-robin. A/B order is randomised per bout so
    position bias cannot accumulate.
    """
    n = len(candidates)
    if n == 0:
        raise ValueError("no candidates")
    if n == 1:
        return 0, [Standing(0)], []

    rng = rng or random.Random()
    standings = [Standing(i) for i in range(n)]
    bouts: list[Bout] = []
    seen: set[tuple[int, int]] = set()

    for rnd in range(rounds):
        order = sorted(standings, key=lambda s: (-s.score, rng.random()))
        used: set[int] = set()
        for si, s in enumerate(order):
            if s.index in used:
                continue
            opponent = None
            for t in order[si + 1 :]:
                if t.index in used:
                    continue
                key = (min(s.index, t.index), max(s.index, t.index))
                if key in seen:
                    continue
                opponent = t
                break
            if opponent is None:  # everyone left has already been played
                continue

            used.add(s.index)
            used.add(opponent.index)
            seen.add((min(s.index, opponent.index), max(s.index, opponent.index)))

            # Randomise presentation order so "A" position carries no systematic advantage.
            flip = rng.random() < 0.5
            left, right = (opponent.index, s.index) if flip else (s.index, opponent.index)
            try:
                verdict = judge(candidates[left], candidates[right])
            except Exception as exc:  # noqa: BLE001
                logger.warning("judge failed on bout %s vs %s: %s", left, right, exc)
                continue

            pick = str(verdict.get("winner", "A")).strip().upper()
            win_idx = left if pick == "A" else right
            lose_idx = right if pick == "A" else left

            bouts.append(Bout(
                a=left, b=right, winner=win_idx,
                criterion=str(verdict.get("deciding_criterion", "")),
                why=str(verdict.get("why", "")),
                confidence=str(verdict.get("confidence", "medium")),
            ))
            standings[win_idx].wins += 1
            standings[lose_idx].losses += 1
            if verdict.get("loser_flaw"):
                standings[lose_idx].notes.append(str(verdict["loser_flaw"]))

        logger.info("tournament round %d/%d: %d bouts total", rnd + 1, rounds, len(bouts))

    ratings = bradley_terry(n, bouts)
    for s in standings:
        s.rating = ratings[s.index]

    # Playoff: Swiss over few rounds leaves the top candidates poorly separated, so settle the
    # final call with a direct head-to-head between the two leaders. One extra judge call buys
    # a decisive result instead of an arbitrary float tie-break.
    ranked = sorted(standings, key=lambda s: (-s.rating, -s.score, s.index))
    top, runner_up = ranked[0].index, ranked[1].index
    if (min(top, runner_up), max(top, runner_up)) not in seen:
        flip = rng.random() < 0.5
        left, right = (runner_up, top) if flip else (top, runner_up)
        try:
            verdict = judge(candidates[left], candidates[right])
            pick = str(verdict.get("winner", "A")).strip().upper()
            final = left if pick == "A" else right
            bouts.append(Bout(
                a=left, b=right, winner=final,
                criterion=str(verdict.get("deciding_criterion", "")),
                why=str(verdict.get("why", "")),
                confidence=str(verdict.get("confidence", "medium")),
            ))
            logger.info("playoff: candidate %d beats %d", final, right if final == left else left)
            return final, standings, bouts
        except Exception as exc:  # noqa: BLE001
            logger.warning("playoff judge failed, falling back to ratings: %s", exc)

    return top, standings, bouts
