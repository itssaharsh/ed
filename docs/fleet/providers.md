# Unit: providers — LLM, voice, image, publish, doctor, config

Owns: `shorts/config.py`, `shorts/llm.py`, `shorts/voice.py`, `shorts/images.py`,
`shorts/doctor.py`, `shorts/publish.py`, `docs/SETUP.md`, `docs/RESEARCH.md`,
`requirements.txt`, new `tests/test_providers_*.py`.

1. Read `docs/RESEARCH.md` and `docs/SETUP.md` first (what was probed, when),
   then the owned code and `ed/config.py` + `ed/youtube.py` (reference only:
   how v1 authenticated and why it 404'd — every model id inline).
2. For each provider path (Gemini text, Groq Orpheus voice, edge-tts
   fallback, Cloudflare FLUX, Pollinations keyless, YouTube upload): what
   env var it needs, what happens when it is missing (must degrade honestly
   or refuse — never a stack trace, never a silent placeholder), retry /
   backoff / rate-limit handling (429, 5xx, timeouts), and the process-wide
   16 s gap for anonymous image requests. Model ids: all in `config.py`, and
   current — check the provider docs you can reach; if the sandbox cannot
   reach them, write "unverified as of <date>" in `RESEARCH.md` instead of
   guessing.
3. `--doctor`: one cheap live call per provider, seconds not minutes, exact
   report of what will run / fall back / is missing. Make it exit non-zero
   when the LLM key is absent (an LLM key is mandatory — CLAUDE.md).
4. `publish.py`: credentials only from `YOUTUBE_CREDENTIALS_PATH`; refreshed
   token surfaced for the workflow (`GITHUB_OUTPUT`) without ever printing
   it; privacy default `private`; the 6/day quota reasoning documented; a
   dry run never touches the API.
5. Tests: fake HTTP (no network) for each failure mode you handle; mutation-
   check them. Run `run.py --doctor` here and paste the report (it will show
   missing keys — that is the correct output for this sandbox).
6. `requirements.txt`: pinned, real versions (`pip index versions`).

End with `FLEET-RESULT`.
