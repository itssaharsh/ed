#!/usr/bin/env python3
"""Render every brief in a directory, one at a time.

    python batch.py                      # all briefs in briefs/, dry run
    python batch.py --publish            # upload each one that passes the gate
    python batch.py --only fridge,lift   # just these
    python batch.py --limit 3

**Serial on purpose.** Free image tiers rate-limit by IP. Running briefs in parallel makes
throughput *worse*, not better — measured here, three concurrent renders produced one image in
seven minutes and spent the rest of the time in 429 backoff, while a single paced render produced
one image every 60-90 seconds. One at a time is faster.

Each brief gets its own run directory and its own exit status; one failure does not stop the
batch. A summary prints at the end.
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from shorts.brief import BriefError, load  # noqa: E402
from shorts.config import Config, logger  # noqa: E402

def _meaning(code: int, published: bool) -> str:
    """Exit code 0 means the run completed cleanly — which is 'uploaded' only if we asked it to."""
    return {
        0: "uploaded" if published else "rendered (dry run)",
        1: "build error",
        2: "rejected by the quality gate",
        3: "upload failed",
        4: "no LLM key",
    }.get(code, f"exit {code}")


@dataclass
class Outcome:
    name: str
    code: int
    seconds: float
    video: Path | None

    @property
    def ok(self) -> bool:
        return self.code == 0


def run_one(brief: Path, *, publish: bool, seed: int, privacy: str) -> Outcome:
    args = [sys.executable, "-u", str(ROOT / "run.py"), "--brief", str(brief), "--seed", str(seed)]
    if not publish:
        args.append("--dry-run")
    else:
        args += ["--privacy", privacy]

    t0 = time.time()
    logger.info("=" * 72)
    logger.info("brief %s (seed %d)%s", brief.name, seed, "" if publish else "  [dry run]")
    logger.info("=" * 72)
    proc = subprocess.run(args, cwd=ROOT)
    elapsed = time.time() - t0

    latest = None
    work = Config().work
    if work.exists():
        candidates = sorted(work.glob(f"*-{seed % 10000:04d}/final.mp4"))
        latest = candidates[-1] if candidates else None
    return Outcome(brief.stem, proc.returncode, elapsed, latest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "briefs")
    ap.add_argument("--publish", action="store_true",
                    help="actually upload (default is a dry run)")
    ap.add_argument("--privacy", choices=["private", "unlisted", "public"], default="private")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated substrings; only briefs matching one are rendered")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    briefs = sorted(args.dir.glob("*.json"))
    if args.only:
        wanted = [w.strip().lower() for w in args.only.split(",") if w.strip()]
        briefs = [b for b in briefs if any(w in b.stem.lower() for w in wanted)]
    if not briefs:
        print(f"no briefs found in {args.dir}")
        return 1

    # Validate everything up front. A typo in brief #6 should not surface an hour into the batch.
    valid: list[Path] = []
    for b in briefs:
        try:
            load(b)
            valid.append(b)
        except BriefError as exc:
            logger.error("invalid brief %s: %s", b.name, exc)
    if not valid:
        print("no valid briefs")
        return 1
    if args.limit:
        valid = valid[: args.limit]

    base_seed = args.seed if args.seed is not None else random.randrange(10**6)
    logger.info("rendering %d brief(s) serially%s",
                len(valid), "" if args.publish else " (dry run)")

    results: list[Outcome] = []
    t0 = time.time()
    for i, brief in enumerate(valid):
        results.append(run_one(brief, publish=args.publish,
                               seed=base_seed + i * 17, privacy=args.privacy))

    total = time.time() - t0
    print("\n" + "=" * 72)
    print(f"batch complete in {total/60:.1f} min")
    print("=" * 72)
    for r in results:
        video = f"  {r.video}" if r.video else ""
        print(f"  {'ok  ' if r.ok else 'FAIL'}  {r.name:24s} {r.seconds/60:5.1f} min  "
              f"{_meaning(r.code, args.publish)}{video}")

    made = sum(1 for r in results if r.video and r.video.exists())
    print(f"\n{made}/{len(results)} videos rendered, "
          f"{sum(1 for r in results if r.ok)}/{len(results)} clean exits")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
