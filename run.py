#!/usr/bin/env python3
"""Fully-automated AI comedy Shorts pipeline.

    python run.py                 # generate, gate, and publish (private by default)
    python run.py --dry-run       # everything except the upload
    python run.py --no-gate       # skip the quality judge (mechanical checks still run)
    python run.py --seed 42       # reproducible run

Every stage checkpoints into work/<run_id>/, so a failure late in the run leaves the expensive
artefacts (images, audio) on disk for inspection.
"""
from __future__ import annotations

import argparse
import os
import json
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from shorts import prompts, visuals
from shorts.captions import build_ass
from shorts.config import (
    ORPHEUS_VOICES, TARGET_SECONDS, Config, logger,
)
from shorts.fonts import ensure_font
from shorts.images import generate_all
from shorts.llm import LLM, LLMError
from shorts.publish import PublishError, upload
from shorts.qc import run_gate
from shorts.render import concat_shots, finalize, measure_loudness, probe, render_shot
from shorts.store import Entry, Store
from shorts.voice import estimate_word_times, synthesize
from shorts.write import (
    direct, draft_script, generate_premises, pick_category, punch_up, script_text, select_premise,
)


def checkpoint(work: Path, name: str, data: object) -> None:
    path = work / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def build(cfg: Config, rng: random.Random, work: Path, *, use_gate: bool) -> dict:
    llm = LLM(cfg)
    store = Store(cfg.store_path)
    t0 = time.time()

    # ── 1-2: premise ────────────────────────────────────────────────────────
    category = pick_category(rng)
    logger.info("category: %s", category.id)
    premises = generate_premises(llm, category, store.recent_premises())
    fresh = store.filter_new(premises, key=lambda p: p.text())
    if not fresh:
        raise RuntimeError("every generated premise duplicated something already made")
    premise, _ = select_premise(llm, fresh, rng)
    logger.info("premise: %s", premise.text()[:150])
    checkpoint(work, "01_premises", [asdict(p) for p in fresh])
    checkpoint(work, "02_premise", asdict(premise))

    # ── 3-4: script ─────────────────────────────────────────────────────────
    beats, the_joke = draft_script(llm, premise)
    beats = punch_up(llm, beats, the_joke, rng)
    checkpoint(work, "03_beats", {"beats": beats, "the_joke": the_joke})

    # ── 5: performance ──────────────────────────────────────────────────────
    voice_name = rng.choice(ORPHEUS_VOICES[:3])       # the male personas read best deadpan
    lines = direct(llm, beats, voice_name)
    checkpoint(work, "05_lines", [asdict(l) for l in lines])

    # ── 8: speech first, because everything downstream is timed to it ───────
    audio_path, audio_duration, engine = synthesize(cfg, lines, voice_name, work)
    checkpoint(work, "08_timing", {
        "duration": audio_duration, "engine": engine, "voice": voice_name,
        "lines": [{"index": l.index, "role": l.role, "start": round(l.start, 3),
                   "duration": round(l.duration, 3), "direction": l.direction,
                   "pause_before_ms": l.pause_before_ms} for l in lines],
    })

    # ── 6: shots, timed against the real performance ────────────────────────
    style = visuals.choose_style(premise.mechanism, rng)
    shots, character, contract, negative = visuals.build_shot_list(
        llm, lines, style, audio_duration, rng)
    return _finish(
        cfg, rng, work, lines=lines, shots=shots, style=style, character=character,
        contract=contract, negative=negative, premise=premise, category=category,
        audio_path=audio_path, audio_duration=audio_duration, engine=engine,
        store=store, llm=llm, use_gate=use_gate, t0=t0,
    )


def _finish(cfg: Config, rng: random.Random, work: Path, *, lines, shots, style: str,
            character: str, contract: str, negative: str, premise, category,
            audio_path: Path, audio_duration: float, engine: str, store: Store,
            llm, use_gate: bool, t0: float) -> dict:
    """Stages 6(timing) through 12, shared by the LLM path and the brief path.

    Everything from here on is deterministic given the script and the shot list, so both
    `build` (LLM writes it) and `build_from_brief` (a human or Claude writes it) run through
    exactly the same code — the brief path is not a second, divergent pipeline.
    """
    shots = visuals.assign_timing(shots, lines, audio_duration)
    if not shots:
        raise RuntimeError("shot timing produced no shots")
    checkpoint(work, "06_shots", {"style": style, "character": character, "shots": shots})

    # ── 7: images ───────────────────────────────────────────────────────────
    shots = generate_all(cfg, shots, contract, negative, work / "images", rng.randint(1, 10**6))
    usable = [s for s in shots if s.get("ok")]
    if len(usable) < 2:
        raise RuntimeError(f"only {len(usable)} usable images; refusing to render")
    if len(usable) < len(shots):
        # Redistribute the failed shots' time onto their neighbours rather than showing black.
        logger.warning("%d shots lost their image; redistributing their time", len(shots) - len(usable))
        total = sum(s["duration"] for s in shots)
        scale = total / sum(s["duration"] for s in usable)
        cursor = 0.0
        for s in usable:
            s["duration"] *= scale
            s["start"] = cursor
            cursor += s["duration"]
        shots = usable

    # ── 10: render ──────────────────────────────────────────────────────────
    clips_dir = work / "clips"
    clips_dir.mkdir(exist_ok=True)
    clips = []
    for i, shot in enumerate(shots):
        clips.append(render_shot(
            Path(shot["image"]), clips_dir / f"clip_{i:02d}.mp4",
            motion=shot["motion"], duration=shot["duration"], style=style, seed=shot["seed"],
        ))
    silent = concat_shots(clips, work / "silent.mp4", work)

    # ── 9: captions ─────────────────────────────────────────────────────────
    font_family, fonts_dir = ensure_font()
    ass = build_ass(estimate_word_times(lines), work / "captions.ass", font=font_family)

    final = finalize(silent, audio_path, ass, work / "final.mp4",
                     duration=audio_duration, fonts_dir=fonts_dir)
    info = probe(final)
    lufs = measure_loudness(final)
    logger.info("stage 10: rendered %.1fs, %s, %.1f LUFS, %d KB",
                info.get("duration", 0), f"{info.get('width')}x{info.get('height')}",
                lufs or 0.0, final.stat().st_size // 1024)

    script = script_text(lines)

    # ── 11: gate ────────────────────────────────────────────────────────────
    gate = run_gate(
        llm, cfg, lines=lines, shots=shots, video_info=info, audio_duration=audio_duration,
        lufs=lufs, script=script, store=store, premise=premise.text(),
    ) if use_gate else None

    # ── 12: metadata ────────────────────────────────────────────────────────
    meta = dict(getattr(premise, "metadata", None) or {})
    meta.setdefault("title", premise.situation[:70])
    meta.setdefault("description_hook", "")
    meta.setdefault("hashtags", category.tags)
    meta.setdefault("tags", ["comedy", "shorts"])

    # A brief can supply its own metadata; only ask the model when it did not.
    if llm is not None and not meta.get("_authored"):
        try:
            meta = llm.complete_json(prompts.render(
                "09_metadata", SCRIPT=script, PREMISE=premise.prompt_block(),
                CATEGORY_TAGS=category.tags, VOICE="",
            ), temperature=0.8) or meta
        except LLMError as exc:
            logger.warning("metadata generation failed, using fallback: %s", exc)

    description = f"{meta.get('description_hook', '').strip()}\n\n{meta.get('hashtags', category.tags)}".strip()
    checkpoint(work, "12_meta", meta)

    elapsed = time.time() - t0
    calls = llm.calls if llm is not None else 0
    logger.info("build complete in %.0fs (%d LLM calls)", elapsed, calls)

    return {
        "video": final, "script": script, "premise": premise, "category": category,
        "title": meta.get("title", "")[:100], "description": description,
        "tags": meta.get("tags", ["comedy", "shorts"]), "gate": gate,
        "duration": audio_duration, "engine": engine, "style": style,
        "shots": len(shots), "llm_calls": calls, "elapsed": elapsed,
    }


def build_from_brief(cfg: Config, rng: random.Random, work: Path, brief_path: Path,
                     *, use_gate: bool) -> dict:
    """Render a hand-authored brief. Needs no LLM key at all.

    Stages 1-6 are supplied by the brief (see shorts/brief.py). Everything downstream is
    identical to the LLM path.
    """
    from shorts.brief import load as load_brief

    store = Store(cfg.store_path)
    t0 = time.time()

    brief = load_brief(brief_path)
    logger.info("premise: %s", brief.premise.text()[:150])

    dup, score, entry = store.is_duplicate(brief.premise.text())
    if dup:
        raise RuntimeError(
            f"this brief duplicates run {entry.run_id if entry else '?'} "
            f"(similarity {score:.2f}) - write a different premise"
        )

    category = pick_category(rng)
    lines = brief.lines
    checkpoint(work, "02_premise", asdict(brief.premise))
    checkpoint(work, "03_beats", {"beats": [{"role": l.role, "text": l.text} for l in lines],
                                  "source": str(brief_path)})
    checkpoint(work, "05_lines", [asdict(l) for l in lines])

    voice_name = rng.choice(ORPHEUS_VOICES[:3])
    audio_path, audio_duration, engine = synthesize(cfg, lines, voice_name, work)
    checkpoint(work, "08_timing", {
        "duration": audio_duration, "engine": engine, "voice": voice_name,
        "lines": [{"index": l.index, "role": l.role, "start": round(l.start, 3),
                   "duration": round(l.duration, 3), "direction": l.direction,
                   "pause_before_ms": l.pause_before_ms} for l in lines],
    })

    contract = prompts.style_suffix(brief.style)
    _, negative = prompts.style_contract(brief.style)

    meta = dict(brief.metadata)
    meta["_authored"] = True
    brief.premise.metadata = meta          # carried into _finish's metadata step

    llm = LLM(cfg) if cfg.has_llm() else None
    if llm is None and use_gate:
        logger.warning("no LLM key: the comedy judge cannot run. Mechanical checks only - "
                       "the script's quality rests on whoever wrote the brief.")

    return _finish(
        cfg, rng, work, lines=lines, shots=[dict(s) for s in brief.shots], style=brief.style,
        character=brief.character_sheet, contract=contract, negative=negative,
        premise=brief.premise, category=category, audio_path=audio_path,
        audio_duration=audio_duration, engine=engine, store=store, llm=llm,
        use_gate=use_gate, t0=t0,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doctor", action="store_true",
                    help="check every provider and report what this run can do, then exit")
    ap.add_argument("--brief", type=Path, default=None,
                    help="render a hand-authored brief (see briefs/) instead of writing one "
                         "with an LLM. Works with no API key.")
    ap.add_argument("--dry-run", action="store_true", help="build everything, upload nothing")
    ap.add_argument("--no-gate", action="store_true", help="skip the LLM quality judge")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--privacy", choices=["private", "unlisted", "public"], default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.doctor:
        from shorts.doctor import run as doctor_run
        return doctor_run(cfg)
    if args.privacy:
        cfg.privacy = args.privacy
    cfg.dry_run = args.dry_run

    seed = args.seed if args.seed is not None else random.randrange(10**9)
    rng = random.Random(seed)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{seed % 10000:04d}"
    work = cfg.work / run_id
    work.mkdir(parents=True, exist_ok=True)

    caps = cfg.capabilities()
    # A brief supplies stages 1-6, so it needs no LLM. Only the generate-it-yourself path does.
    if not cfg.has_llm() and not args.brief:
        logger.error(
            "No LLM API key found. Either set one - GEMINI_API_KEY (free tier: 500 requests/day), "
            "OPENROUTER_API_KEY, or GROQ_API_KEY (see docs/SETUP.md) - or render a hand-written "
            "brief with --brief briefs/<name>.json, which needs no key at all.\n"
            "The keyless Pollinations text tier is not sufficient: its anonymous allowance is a "
            "few requests before it returns 401, and a single video needs about 14 calls."
        )
        return 4

    logger.info("run %s | seed %d", run_id, seed)
    logger.info("providers: llm=%s | images=%s | voice=%s", caps["llm"], caps["images"], caps["voice"])
    if "degraded" in caps["images"] or "degraded" in caps["voice"]:
        logger.warning("running on degraded providers - see docs/SETUP.md for the free keys "
                       "that unlock the good paths")

    try:
        if args.brief:
            result = build_from_brief(cfg, rng, work, args.brief, use_gate=not args.no_gate)
        else:
            result = build(cfg, rng, work, use_gate=not args.no_gate)
    except Exception as exc:  # noqa: BLE001
        logger.error("build failed: %s", exc)
        traceback.print_exc()
        (work / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1

    gate = result["gate"]
    if gate:
        print("\n" + gate.report() + "\n")

    print(f"  video     {result['video']}")
    print(f"  title     {result['title']}")
    print(f"  script    {result['script'][:200]}")
    print(f"  duration  {result['duration']:.1f}s across {result['shots']} shots, style={result['style']}")
    print(f"  voice     {result['engine']}")
    print(f"  cost      {result['llm_calls']} LLM calls, {result['elapsed']:.0f}s wall clock\n")

    def remember(video_id: str | None) -> None:
        """Record the attempt whether or not it published.

        Recording only on success would let a rejected premise be regenerated on the next run,
        and the run after that - the dedup store is what stops the pipeline rediscovering the
        same joke forever. A dry run is excluded: it is a rehearsal, not an attempt.
        """
        if cfg.dry_run:
            return
        Store(cfg.store_path).append(Entry(
            run_id=run_id, ts=time.time(), category=result["category"].id,
            premise=result["premise"].text(), script=result["script"],
            title=result["title"], video_id=video_id, published=bool(video_id),
        ))

    if gate and not gate.passed:
        logger.error("quality gate rejected this run; nothing will be published")
        remember(None)
        return 2

    if cfg.dry_run:
        logger.info("dry run - not uploading. Artefacts in %s", work)
        return 0

    creds = Path(os.environ.get("YOUTUBE_CREDENTIALS_PATH", "credentials.json"))
    try:
        video_id = upload(cfg, result["video"], title=result["title"],
                          description=result["description"], tags=result["tags"],
                          creds_path=creds)
    except PublishError as exc:
        logger.error("publish failed: %s", exc)
        remember(None)      # the premise was still spent - do not regenerate it next run
        return 3

    remember(video_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
