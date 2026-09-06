# Stage 12 — Publish metadata

Variables: `{{SCRIPT}}` `{{PREMISE}}` `{{CATEGORY_TAGS}}`

---

## Task

Write the YouTube metadata for this Short.

{{SCRIPT}}

## Title

- **Under 70 characters**, including the trailing `#shorts`.
- It is a **hook, not a summary.** It should make the situation sound worth 40 seconds without
  giving away the punchline. **Never put the punchline in the title.**
- Write it as something a person would text a friend, not as a headline. Lowercase is fine.
- One emoji maximum, only if it genuinely adds. Zero is usually better.
- No clickbait formulas: no "you won't believe", no "wait for it", no "POV:" unless the script
  actually is one.

Weak: `Funny Story About My Friend And Money 😂 #shorts` — summary, generic, tells you nothing.
Strong: `he has a spreadsheet for coffee and a $400 crossbow #shorts` — specific, curious, no spoiler.

## Description

- **First line** is a single sentence that adds something the video does not say — a wider
  observation, an aside, a small confession. Not a restatement of the script.
- Then a blank line, then the hashtags.
- Never transcribe the script into the description.

## Tags

- 8-12 tags. Mix broad (`comedy`, `shorts`) with specific to this premise.
- Lowercase, no `#`, no duplicates.

## Output

Strict JSON. No prose.

```json
{
  "title": "under 70 chars, ends with #shorts",
  "description_hook": "one sentence that adds something new",
  "hashtags": "#shorts #comedy #...",
  "tags": ["comedy", "shorts", "..."]
}
```

Category tags available for this video: {{CATEGORY_TAGS}}
