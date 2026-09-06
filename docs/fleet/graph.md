# Unit: graph — graphify knowledge graph for this repo

Owns: `graphify-out/**` (except `cost.json`), one added line in `.gitignore`.

graphify is PyPI package `graphifyy` (command `graphify`): local tree-sitter
code extraction plus a docs pass that uses the Claude Code session it is
installed into as a skill — no API key.

1. Install without touching the repo: `uv tool install graphifyy` (or
   `pipx install graphifyy`, or `pip install --user graphifyy`), then
   `graphify --version` and `graphify --help`. **Read the help before using
   any flag**; do not assume the flags this file names exist.
2. `graphify install` at USER level only (do not commit anything under
   `.claude/`), then read the SKILL.md it wrote.
3. Build the graph for the repo root the way the skill instructs, including
   the docs pass (`CLAUDE.md`, `docs/`, `prompts/*.md`, `briefs/README.md`
   are the memory that matters) and excluding `.venv`, `work`, `ed/` (v1 is
   noise), `__pycache__`. Output `graphify-out/` at the root.
4. Sanity: `graphify query "which module keeps the quality gate fail-closed?"`
   and `graphify explain "tournament"`; paste both answers in the results
   file. Fewer than ~40 nodes means something was excluded wrongly — fix it.
5. Add `graphify-out/cost.json` to `.gitignore`. Commit
   `fleet(graph): graphify knowledge graph`, push.

End with `FLEET-RESULT: BUILT … — nodes=N edges=M`.
