# Stage 4 — Punch-up

Variables: `{{SCRIPT}}` `{{THE_JOKE}}` `{{N}}`

---

{{VOICE}}

---

## Why this stage exists

Research with professional comedians found that LLMs reliably produce good *setups* and *structure*
but rarely the punchline — humans supplied those. We cannot supply a human, so we substitute
**volume plus selection**: you write many endings, and a pairwise tournament picks one.

So do not try to write *the* perfect punchline. Write {{N}} genuinely different ones and let the
tournament sort it out. A weird one that might not work is more useful than a safe one that
definitely half-works.

## Task

Here is the script:

{{SCRIPT}}

The intended mechanism: {{THE_JOKE}}

Write **{{N}} alternative final lines** (the `punch` beat).

### Requirements

- Each must be a **different comedic mechanism**, not a rewording. Vary across:
  - **Deflation** — undercut the built-up stakes with something mundane
  - **Escalation** — go one step further than the audience thought possible
  - **Reversal** — reveal the narrator was the problem
  - **Literalism** — take an earlier figure of speech at face value
  - **Callback** — reuse the setup's specific detail in a new role
  - **Deadpan understatement** — state the catastrophe as a minor inconvenience
- Each must be **6-14 words**.
- Each must work **only for this script**. If a punchline could be pasted onto a different bit,
  it is too generic — discard it and write another.
- **Do not explain.** No "which is why", no "turns out". End on the image.
- At least two must be **riskier** than the current ending. Do not play safe across the whole set.

## Output

Strict JSON. No prose.

```json
{
  "candidates": [
    {"id": 1, "text": "...", "mechanism": "deflation", "risk": "safe"},
    {"id": 2, "text": "...", "mechanism": "reversal", "risk": "risky"}
  ]
}
```
