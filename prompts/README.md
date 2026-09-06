# Prompt files

Every LLM instruction in this pipeline lives here as a file, not inline in Python. Prompts are the
part of this system most likely to need tuning, and they are the part most worth reviewing as
prose. Editing a `.md` here changes behaviour without touching code.

| File | Stage | Purpose |
|---|---|---|
| `00_voice.md` | shared | Narrator persona + the banned AI-comedy tells. Prepended to writing stages, **never** to judging stages. |
| `01_ideate.md` | 1 | Generate N divergent premises. Quantity, not quality. |
| `02_tournament.md` | 2, 4 | Pairwise comedy judge. Used for premises and punchlines. |
| `03_script.md` | 3 | Premise → beat-structured script. |
| `04_punchup.md` | 4 | N alternative punchlines with distinct mechanisms. |
| `05_delivery.md` | 5 | Beats → vocal direction + designed silences. |
| `06_shotlist.md` | 6 | Lines → shots → image prompts. |
| `07_styles.md` | 6 | Style contracts that keep shots visually coherent. |
| `08_qc.md` | 11 | Fail-closed quality gate. |
| `09_metadata.md` | 12 | Title, description, tags. |

## Conventions

- `{{VARIABLE}}` is substituted by `shorts/prompts.py`. A missing variable raises — it never
  silently renders as an empty string.
- `{{VOICE}}` expands to the body of `00_voice.md` (below its `---` separator).
- Every prompt ends with a **strict JSON output block**. Parsing is strict; prose responses are
  retried with a stricter instruction, then fail.

## The two design rules

**1. Judge by comparison, never by score.** Absolute humour ratings from an LLM are noise:
88.5% of 0-100 joke scores collapse to identical values. Pairwise comparison reaches cross-judge
τ=0.889. Any new judging prompt must be pairwise.

**2. Writing prompts get the persona; judging prompts do not.** A judge carrying the comedian
persona rates its own style highly. Judges stay neutral.

## Editing

Prompts are behaviour. Change one and the output changes across every future video, so:

- Change **one** prompt at a time, then run `python run.py --dry-run --keep` and read the artefacts
  in `work/<run_id>/` before letting it publish.
- The QC gate's baseline script (`assets/baseline_script.json`) is the quality floor. Raising it
  raises the bar for everything; do it deliberately.
