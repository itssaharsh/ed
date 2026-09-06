# Stage 1 — Premise generation (divergent)

Goal: **quantity and spread.** Do not self-edit here. Selection happens in stage 2, and it is
brutal. Your job is to give it genuinely different things to choose between.

Variables: `{{CATEGORY}}` `{{CATEGORY_BRIEF}}` `{{RECENT_PREMISES}}` `{{N}}`

---

{{VOICE}}

---

## Task

Write **{{N}} different premises** for a 35-45 second spoken comedy bit.

Category: **{{CATEGORY}}** — {{CATEGORY_BRIEF}}

A premise is one sentence describing *the specific situation*, plus one sentence on *where it goes*.
It is not a joke yet. It is the thing the joke will be about.

### Already used — do not repeat these or anything adjacent

{{RECENT_PREMISES}}

Adjacent means: same setup, same target, same escalation, or the same joke wearing a hat. If your
premise could be swapped into one of the above without anyone noticing, it does not count as new.

### Spread requirement

Across your {{N}} premises, deliberately vary:
- **Target** — yourself / a friend / a stranger / an institution / an object behaving badly
- **Scale** — a two-second moment vs. a months-long pattern
- **Register** — warm and fond vs. genuinely irritated vs. clinically baffled
- **Mechanism** — misplaced confidence / bad system design / sincere effort in the wrong direction /
  a rule everyone follows but nobody agreed to / escalating commitment to a bad decision

At least 3 must be about something with **no screen in it**. At least 2 must be about a
**physical object**. At least 2 must have **no other people** in them.

### Quality bar per premise

- It must contain **one concrete, unguessable detail**. That detail is the seed of the joke.
- It must be **observable** — something you could film, not a mood.
- It must be **true enough to recognise** even when exaggerated.

Bad premise: "dating apps are exhausting" — vague, no detail, no target, everyone's said it.
Good premise: "A man maintains a spreadsheet ranking the women he matches with, and the columns
are all things like 'replies fast' — until you see the last column is 'would help me move'."

## Output

Strict JSON. No prose, no markdown fence.

```json
{
  "premises": [
    {
      "id": 1,
      "situation": "one sentence — the specific situation",
      "turn": "one sentence — where it escalates to",
      "detail": "the single concrete unguessable detail",
      "mechanism": "misplaced confidence | bad system | sincere wrong effort | unspoken rule | escalating commitment",
      "target": "self | friend | stranger | institution | object",
      "has_screen": true,
      "has_other_people": true
    }
  ]
}
```
