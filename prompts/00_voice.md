# The writers' room — index

**This file is no longer injected into anything.** It was split in two, because it was two
documents wearing one coat:

| file | what it holds | injected as |
|---|---|---|
| `00_craft.md` | persona-independent craft: the specificity rule, the banned AI tells, the definition of funny, the ending rule | first half of `{{VOICE}}` |
| `voices/*.md` | one narrator each: who they are, how they talk, their punch shape and forbidden punch shape | second half of `{{VOICE}}` |

`shorts/prompts.voice_block(persona)` composes the two. Every stage prompt still uses `{{VOICE}}`
and none of them needed changing.

## Why the split

Every video the pipeline produced used one narrator and ended on a flat declarative sentence — all
eleven hand-authored briefs included. That is simultaneously the funniness ceiling and a direct
match for YouTube's Inauthentic Content policy, which demonetises "a highly similar storyline
template across multiple videos". Varying the narrator, and specifically the *shape of the last
line*, is the cheapest structural variance available: it costs no extra LLM calls and no extra
images.

The research points the same way — multi-persona generation beats single-prompt generation
(HumorGen, arXiv 2604.09629), while selection stays pairwise (see `02_tournament.md`).

## The personas

| id | register | punch shape |
|---|---|---|
| `implicated` | tired, warm, inside it — the original voice | flat declarative deflation |
| `forensic` | clinical, over-precise, counts things | a measurement stated as a verdict |
| `true_believer` | wholly sincere, has a system, defends it | doubling down — a forward commitment |

Each persona also declares a **forbidden** punch shape, so the voices cannot converge back onto
the same ending. `implicated` may not end on a rhetorical question; `forensic` may not borrow
`implicated`'s flat deflation; `true_believer` may never have a retrospective realisation.

## Rules that still hold

- **Judges get no persona and no craft block.** `prompts.render(..., VOICE="")` for every judging
  stage. A judge carrying the writing persona prefers its own voice and the tournament becomes
  noise. This is enforced by `tests/test_units.py:test_prompts`.
- **Persona is chosen in code before generation, never by a judge.** The tournament is a *quality*
  selector; it has no visibility of the sequence of previous videos, so it cannot be a diversity
  selector. Diversity is a code decision, quality is a tournament decision.

## Adding a persona

Drop a file in `voices/`. It is picked up automatically by `prompts.available_personas()`. Give it
a distinct punch shape and name a forbidden one — a persona that merely sounds different but ends
the same way does not buy the variance this exists for.
