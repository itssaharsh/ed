# Stage 5 — Vocal direction

Turns beats into a performance. This is the stage that fixes "bad voice".

Variables: `{{BEATS}}` `{{VOICE_NAME}}`

---

## The instrument

Target engine is **Orpheus** (`canopylabs/orpheus-v1-english`), voice `{{VOICE_NAME}}`.

- **Bracketed direction** at the start of a line sets its delivery: `[deadpan] He said it twice.`
- **Inline non-verbals** are performed, not spoken: `<sigh>`, `<laugh>`, `<giggle>`, `<groan>`.
- **200 characters per line, hard cap.** Lines over that are split, which breaks the performance.
- Directions work best as **1-2 words**, adjective or adverb.
- Removing punctuation gives the model more freedom in choosing intonation — use for the most
  expressive lines only.
- Omit the direction entirely for a plain conversational read. **This is the right default.**

## The most important rule

**Do not direct every line.** A script where every line is `[sarcastic]` or `[dramatic]` sounds
like a cartoon and is exhausting within ten seconds. Direction is punctuation, not paint.

**At most 3 of the lines get a bracketed direction.** Everything else runs plain. The contrast is
what makes the directed lines land — a single `[deadpan]` after four plain lines does more than
six stacked directions.

Non-verbals are rarer still: **at most one** in the whole script, and only if the script earns it.
An unearned `<laugh>` is the single most artificial-sounding thing a voice model can do.

## Silence is the other half

`pause_before_ms` is the silence inserted *before* the line. This is where comic timing lives —
more than any tag.

| Position | Pause | Why |
|---|---|---|
| Hook | **0** | Never make them wait at the start. They will scroll. |
| Between escalations | 80-150 | Keep momentum. Do not let it breathe. |
| Before the turn | 200-350 | Small lift; something is coming. |
| **Before the punch** | **400-650** | The load-bearing silence of the entire video. |
| Before a tag | 250-400 | Let the punch land and settle first. |

Do not exceed 650ms anywhere. On a Short, a long silence reads as buffering and loses the viewer.

## Emphasis

Mark 1-3 words per line as emphasised. These drive **caption highlighting** downstream, so choose
the word that carries the *meaning*, not just a loud word. The unguessable noun usually wins.

## Task

Direct these beats:

{{BEATS}}

## Output

Strict JSON. No prose. Preserve beat order exactly.

```json
{
  "lines": [
    {
      "index": 0,
      "role": "hook",
      "text": "the spoken text, unchanged from the script",
      "direction": null,
      "pause_before_ms": 0,
      "emphasis": ["WORD"]
    },
    {
      "index": 5,
      "role": "punch",
      "text": "...",
      "direction": "deadpan",
      "pause_before_ms": 520,
      "emphasis": ["SPREADSHEET"]
    }
  ],
  "voice_note": "one sentence on the overall performance intent"
}
```

Constraints, enforced downstream — violating them causes a re-ask:
- `text` must be **character-identical** to the script beat. You are directing, not rewriting.
- No line over **200 characters**.
- At most **3** non-null `direction` values.
- At most **1** non-verbal tag across all lines.
- Every `emphasis` word must appear verbatim in that line's `text`.
