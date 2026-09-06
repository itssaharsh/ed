# Style contracts

One is selected per video and injected into every image prompt in that video, unchanged. This is
what makes ten separately-generated images look like one production.

**Policy note.** All three presets are deliberately **stylised, not photoreal.** YouTube's
"Modified or Synthetic" disclosure applies to content that could be mistaken for real footage of a
real person or event; illustrated, cartoon and obviously-stylised imagery is exempt, and photoreal
synthetic media has been auto-detected and labelled since May 2026. Stylisation is also simply
better: an obvious art style reads as an intentional choice, while near-miss realism reads as
failed realism — and it hides the anatomy and face errors that free image models still make.

---

## `flat_absurd` — flat vector comedy (default)

> Flat vector illustration, bold clean outlines, limited palette of six flat colours, no gradients,
> geometric simplified shapes, exaggerated cartoon proportions with oversized heads, thick uniform
> line weight, plain colour-block background, centred deadpan composition, editorial-illustration
> style, high contrast, no texture, no shading detail.

**Prompt suffix:** `flat vector illustration, bold black outlines, flat colour blocks, mustard and teal palette, no gradients`

- **Palette:** mustard yellow, burnt orange, deep teal, off-white, charcoal, one hot-pink accent.
- **Negative:** photorealistic, 3d render, gradients, soft shading, realistic skin, text, letters,
  words, signage, watermark, extra fingers, blurry, noisy, cluttered background.
- **Why it works:** cheapest to generate cleanly, most forgiving of model errors (no faces to
  mangle), reads instantly at thumbnail size, and the flatness makes the absurdity land harder.
- **Best for:** corporate satire, unspoken social rules, object-behaving-badly premises.

## `grain_docu` — muted 90s documentary

> Muted 16mm film illustration, desaturated earth tones, heavy visible film grain, soft diffused
> light, slightly washed highlights, naturalistic but painterly rendering, shallow depth of field,
> off-centre documentary framing, subdued colour, gentle vignette.

**Prompt suffix:** `muted 16mm film still, desaturated ochre and olive, heavy grain, soft diffused light`

- **Palette:** ochre, dust brown, faded olive, cream, slate grey.
- **Negative:** vibrant saturated colour, digital sharpness, hdr, glossy, text, letters, words,
  signage, watermark, deformed hands, extra limbs.
- **Why it works:** the sincerity of the format against absurd content is itself the joke — it
  looks like a real documentary about something profoundly stupid.
- **Best for:** sincere-effort-in-the-wrong-direction, escalating-commitment premises.

## `neon_late` — late-night saturated

> Saturated neon-lit illustration, deep shadows with strong magenta and cyan rim light, glossy
> reflective surfaces, night interior, dramatic single-source lighting, cinematic contrast,
> stylised painterly rendering, moody atmosphere.

**Prompt suffix:** `neon-lit illustration, magenta and cyan rim light, deep shadows, night interior, high contrast`

- **Palette:** magenta, cyan, deep indigo, black, one sodium-orange accent.
- **Negative:** daylight, flat even lighting, pastel, washed out, text, letters, words, signage,
  watermark, deformed face, extra fingers.
- **Why it works:** high contrast survives compression and small screens; the drama of the lighting
  against a mundane subject creates the incongruity.
- **Best for:** 3am decisions, doomscrolling, dating, dread-adjacent premises.

---

## Selection

Chosen by premise `mechanism` (see `prompts/01_ideate.md`), not at random, so the look matches the
joke:

| mechanism | style |
|---|---|
| bad system, unspoken rule | `flat_absurd` |
| sincere wrong effort, escalating commitment | `grain_docu` |
| misplaced confidence | `neon_late` |

## Universal negative prompt

Appended to every image request regardless of style:

> text, letters, words, numbers, signage, labels, logos, watermark, signature, subtitles, caption,
> extra fingers, extra limbs, deformed hands, mutated face, disfigured, low quality, jpeg artifacts,
> collage, split frame, border, frame, multiple panels

## Adding a style


Keep the contract to **one paragraph of comma-separated visual attributes** and never include
subject matter. Subject comes from the shot list; the contract only ever describes *how it looks*.
If a contract mentions a person, a place, or an action, it will fight the shot list and shots will
drift apart.
