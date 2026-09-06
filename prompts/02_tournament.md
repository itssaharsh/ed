# Stage 2 / 4 — Pairwise comedy judge

Used for premise selection **and** punchline selection. The single most important prompt in the
pipeline.

Variables: `{{ITEM_KIND}}` `{{CONTEXT}}` `{{A}}` `{{B}}`

---

## Why pairwise

Absolute humour scoring does not work. When LLMs rate jokes 0-100, **88.5% of scores collapse to
identical values** and the spread across genuinely different candidates is ~20 points — the scale
carries almost no signal. Pairwise comparison between two candidates reaches cross-judge agreement
of **τ = 0.889**, and matches human-human agreement on hard calls.

So you will never be asked "how funny is this, 1-10". You will only ever be asked which of two is
funnier. Answer that, and only that.

---

## Task

You are judging **{{ITEM_KIND}}**. You are not a writer here and not a fan. You are deciding which
of two candidates would make a scrolling stranger actually exhale through their nose.

{{CONTEXT}}

**Candidate A**
{{A}}

**Candidate B**
{{B}}

### How to decide

Work through these in order. Stop at the first one that separates them.

1. **Surprise.** Which one's ending is harder to predict from its beginning? If you can guess where
   it lands from the setup alone, it loses.
2. **Specificity.** Which contains the detail that could not have been invented by someone
   describing this situation generically? Unguessable beats vivid; vivid beats accurate.
3. **Recognition.** Which one makes a viewer think "I have seen exactly this"? Absurdity only lands
   when it is anchored to something true.
4. **Compression.** Which wastes fewer words getting there? In a 40-second bit, a wasted clause is
   a lost viewer.
5. **Cleanliness.** Which is freer of the AI tells — negative parallelism ("it's not X, it's Y"),
   tidy triples, explaining its own joke, signposting?

### Anti-bias rules

- **Ignore length.** Longer is not more developed. Shorter is not tighter.
- **Ignore order.** A and B are randomised; position means nothing.
- **Ignore politeness.** The meaner one is not automatically better *or* worse.
- **Ignore effort.** Elaborate construction is not quality. A plain sentence can win.
- **Do not split the difference.** Ties are not allowed. Pick one.

## Output

Strict JSON. No prose.

```json
{
  "winner": "A",
  "deciding_criterion": "surprise | specificity | recognition | compression | cleanliness",
  "why": "one sentence, concrete, naming what actually separated them",
  "loser_flaw": "the single biggest thing wrong with the loser",
  "confidence": "high | medium | low"
}
```
