# Stage 6 — Shot list and image prompts

This is the stage that fixes "stock footage badly fitted". Nothing is searched for. Every frame is
generated to match a specific moment.

Variables: `{{LINES}}` `{{STYLE_NAME}}` `{{STYLE_CONTRACT}}` `{{TOTAL_SECONDS}}`

---

## Two rules that matter more than the rest

**1. One world.** Every shot in this video must look like it came from the same production — same
style, same palette, same lens language, same character. Visual incoherence between shots is the
single biggest tell of cheap AI video. You are given a **style contract** and it is not negotiable;
it gets appended to every prompt automatically. Your job is to describe *what happens in the shot*,
never *what style it is in*.

**2. Cut on the joke.** A shot change is a comic beat. The image must change at the moment the
meaning changes — most importantly, **the punch line gets its own shot, and that shot must not be
visible before the punch begins.** Revealing the visual gag early kills it.

## The style contract for this video

Style: **{{STYLE_NAME}}**

{{STYLE_CONTRACT}}

## Shot grammar

- **One shot per line**, unless a line is long enough (>3.5s) to earn a second shot inside it.
- **Vary the shot size** between adjacent shots. Two consecutive mediums read as a slideshow.
  Cycle among: wide establishing / medium / close-up / extreme close-up on an object / over-shoulder.
- **The reaction shot is your best tool.** A person receiving the absurdity is usually funnier than
  the absurdity itself.
- **Visualise literally, not metaphorically.** If the line says someone has a spreadsheet for
  their coffee budget, show *the spreadsheet*. Literal-minded framing of an absurd statement is
  the joke. Do not illustrate the *feeling* of the line.
- **Escalate visually alongside the script.** The last shots should be visibly more absurd than
  the first. If the script escalates and the images do not, the video plateaus.
- **The hook shot must be arresting on frame one.** It is the thumbnail and the scroll-stopper.
  A face, a strong silhouette, or one baffling object. Never an empty room.

## Composition — this is a 9:16 Short with captions over it

- Compose for **vertical**. The subject occupies the upper-middle of the frame.
- **Keep the band from 55% to 70% of frame height visually quiet** — captions live there. No faces,
  no critical detail, no busy texture in that band.
- Leave headroom at the top. The top ~8% and bottom ~22% are covered by YouTube's UI.

## Text in images

Image models cannot spell. **Never request readable text, signs, labels, logos, or UI with words.**
If the joke needs a spreadsheet, describe "a spreadsheet with dense unreadable rows of numbers",
never "a spreadsheet that says COFFEE BUDGET". Gibberish lettering is an instant tell.

## Prompt writing

Each `prompt` field should be **one dense sentence, 15-35 words**, in this order:

`[shot size] of [subject] [doing what] , [expression/pose] , [environment] , [light]`

- Concrete nouns. No adjectives about quality ("beautiful", "amazing", "high quality").
- Name the **expression** explicitly — flat stare, dawning horror, unearned confidence. Faces carry
  the comedy.
- Name the **light** — it is what makes a still read as cinematic rather than as clip-art.
- **Do not include style, medium, artist, camera, or film-stock words.** Those come from the style
  contract. Repeating them causes drift between shots.

## Task

Build the shot list for these directed lines. Total runtime ≈ {{TOTAL_SECONDS}}s.

{{LINES}}

## Output

Strict JSON. No prose.

```json
{
  "character_sheet": "one sentence describing the recurring person, if any — age, build, hair, clothing. Reused verbatim in every prompt featuring them so they stay the same person across shots. Empty string if the video has no recurring character.",
  "shots": [
    {
      "line_index": 0,
      "shot_size": "wide | medium | close | extreme-close | over-shoulder",
      "prompt": "one dense sentence per the format above",
      "motion": "push-in | pull-out | drift-left | drift-right | static-float",
      "why_this_image": "one sentence: what this shot does for the joke"
    }
  ]
}
```

`motion` guidance: `push-in` builds pressure — use it into the turn and the punch. `pull-out`
reveals — use it when the shot's joke is context the viewer cannot see yet. `drift` is neutral
motion for setup lines. `static-float` (a near-still handheld hold) is for the punch when the image
itself is the gag and movement would distract. **Never use the same motion on two adjacent shots.**
