"""Run the whole pipeline with a stubbed LLM. Proves the integration and produces a sample."""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as runner
from shorts.config import Config, logger
from stub_llm import StubLLM


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    cfg = Config()
    cfg.dry_run = True
    runner.LLM = lambda _cfg: StubLLM(seed)          # noqa: ARG005

    rng = random.Random(seed)
    work = cfg.work / f"offline-{seed}"
    work.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = runner.build(cfg, rng, work, use_gate=True)
    gate = result["gate"]
    print("\n" + (gate.report() if gate else "(no gate)"))
    print(f"\n  video    {result['video']}")
    print(f"  title    {result['title']}")
    print(f"  script   {result['script']}")
    print(f"  {result['duration']:.1f}s, {result['shots']} shots, style={result['style']}, voice={result['engine']}")
    print(f"  wall     {time.time() - t0:.0f}s")
    return 0 if (gate is None or gate.passed) else 2


if __name__ == "__main__":
    sys.exit(main())
