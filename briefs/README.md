# Briefs — writing the comedy by hand

A brief supplies stages 1–6 (premise, script, delivery, shot list) directly, so the pipeline
renders with **no LLM API key at all**. Use it when:

- the key's daily quota is spent, or you have no key;
- you want to write the jokes yourself and keep the machinery;
- an assistant (Claude, or anything else) is acting as the writer and hands over JSON.

```bash
python run.py --brief briefs/fridge_baseline.json --dry-run
python batch.py                       # every brief, serially, dry run
python batch.py --publish             # upload the ones that pass the gate
python batch.py --only kettle --limit 1
```

## Render serially, not in parallel

`batch.py` runs one brief at a time deliberately. Free image tiers rate-limit **per IP**, so
parallel renders fight each other. Measured on the keyless tier:

| | result |
|---|---|
| 3 renders in parallel | 1 image in 7 minutes, the rest in 429 backoff |
| 1 render, paced | ~1 image every 60–180s |

Parallelism made it roughly five times *slower*. `shorts/images.py` also enforces a
process-wide 16-second gap between anonymous requests for the same reason.

## Fewer shots renders faster

A shot may span several lines — an uncovered line holds the previous image. Four shots across
six lines is perfectly good pacing and needs 43% fewer image generations, which is the difference
between a 12-minute render and a 25-minute one on a slow provider.

Two shots are non-negotiable:

- **the hook (line 0)** — it is the first frame and decides whether anyone watches;
- **the punch** — the visual gag must land *on* the punch. If its image appears earlier, the
  joke is spoiled before the line arrives.

`kettle_twice.json` is the worked example of a sparse shot list.

## Schema

```jsonc
{
  "premise": {
    "situation": "one sentence — the specific situation",
    "turn":      "one sentence — where it escalates to",
    "detail":    "the concrete unguessable detail",
    "mechanism": "bad system | unspoken rule | sincere wrong effort | escalating commitment | misplaced confidence",
    "target":    "self | friend | stranger | institution | object"
  },
  "style": "flat_absurd | grain_docu | neon_late",
  "character_sheet": "the recurring person, appended to every shot featuring someone",

  "beats": [ {"role": "hook|setup|escalate|turn|punch|tag", "text": "..."} ],

  "direction": [
    {"index": 0, "pause_before_ms": 0, "direction": "deadpan", "emphasis": ["WORD"]}
  ],

  "shots": [
    {"line_index": 0, "shot_size": "wide|medium|close|extreme-close|over-shoulder",
     "motion": "push-in|pull-out|drift-left|drift-right|static-float",
     "prompt": "one dense sentence: shot size, subject, action, expression, environment, light",
     "why_this_image": "what this shot does for the joke"}
  ],

  "metadata": {"title": "... #shorts", "description_hook": "...", "hashtags": "...", "tags": []}
}
```

Validation is strict and happens on load, so a malformed brief fails immediately with a specific
reason rather than producing a subtly broken video six stages later. It enforces:

- a `hook` beat and a `punch` beat, with the punch (or a `tag`) last;
- every `emphasis` word actually present in its line — captions highlight by exact match;
- `pause_before_ms` clamped to 650ms, and zero on the hook (never make a scroller wait);
- no two adjacent shots sharing a camera move, or it reads as a slideshow;
- `line_index` values that exist, and known styles, motions and roles.

## Writing rules

The craft rules live in [`../prompts/00_voice.md`](../prompts/00_voice.md) — read it before
writing. The short version:

- **Be specific.** Generic is the enemy. "A jar of mustard" beats "some food". The unguessable
  detail *is* the joke mechanism.
- **Escalate monotonically.** If a beat is not bigger than the one before it, cut it.
- **Never explain the joke.** End on the image. No "and that's when I realised".
- **Avoid the AI tells**: "it's not X, it's Y", tidy triples, "little did I know", em-dash piles.
- **Write for the mouth.** Read it aloud. If you stumble, rewrite it.

## Quality gate

With no LLM key the pairwise comedy judge cannot run, so the gate falls back to mechanical checks
only (duration, dimensions, loudness, shot count, image variety, dedup) and says so in its report.
**The humour then rests entirely on whoever wrote the brief** — there is no safety net for a
script that simply is not funny. With a key set, briefs get judged against the baseline like any
other script.
