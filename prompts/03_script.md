# Stage 3 — Script draft

Variables: `{{PREMISE}}` `{{TARGET_SECONDS}}` `{{TARGET_WORDS}}`

---

{{VOICE}}

---

## Task

Write the spoken script for a **{{TARGET_SECONDS}}-second** bit from this premise.

{{PREMISE}}

### Length

**{{TARGET_WORDS}} words.** This is a hard budget, not a target to pad toward.

Do not write short and pad. Do not write long and trust a trim. The old version of this pipeline
appended canned filler lines to hit a word count and it made every video sound identical — that
mechanism is gone. If your draft is short, the premise needs another *beat*, not more *words*.

### Structure — beats, not paragraphs

Return the script as an ordered list of beats. Each beat is one spoken line. This structure is
load-bearing: downstream, each beat gets its own vocal direction, its own silence before it, and
its own shot.

| Beat | Role | Words |
|---|---|---|
| `hook` | Land the situation in the first ~1.5 seconds. No preamble, no "so", no throat-clearing. The first five words decide whether anyone stays. | 8-14 |
| `setup` | One specific detail that plants the scene. This is where the unguessable detail goes. | 10-16 |
| `escalate` x2-3 | Each raises the stakes over the last. Never restate — always advance. One may be a quoted line of dialogue. | 12-20 each |
| `turn` | The moment it stops being normal. | 8-15 |
| `punch` | The last line. Recontextualises everything before it. Must be the funniest line. | 6-14 |
| `tag` *(optional)* | One short line after the punch that lands a second, smaller hit off the same setup. Only include if genuinely good — a weak tag is worse than none. | 4-9 |

### Rules

- **The hook is first, always.** Not context, not "okay so". The situation, immediately.
- **Escalation must be monotonic.** If beat 4 is not bigger than beat 3, cut beat 4.
- **The punch must not explain.** No "and that's when I realised". End on the image.
- **One quoted line maximum.** Direct speech is strong and loses power if repeated.
- **Write for the mouth, not the eye.** Read it aloud in your head. If you stumble, rewrite it.
- **No stage directions in the text.** No "[pause]", no "(beat)". Timing is assigned downstream —
  putting it in the text means the voice model reads it out loud.

## Output

Strict JSON. No prose, no fence.

```json
{
  "beats": [
    {"role": "hook", "text": "..."},
    {"role": "setup", "text": "..."},
    {"role": "escalate", "text": "..."},
    {"role": "escalate", "text": "..."},
    {"role": "turn", "text": "..."},
    {"role": "punch", "text": "..."}
  ],
  "word_count": 96,
  "the_joke": "one sentence: what actually makes this funny — for the judge, not the audience"
}
```
