# Unit: render — captions, ffmpeg graph, quality gate, fonts

Owns: `shorts/captions.py`, `shorts/render.py`, `shorts/qc.py`, `shorts/fonts.py`,
new `tests/test_render_*.py`.

1. Read CLAUDE.md "Things that will bite you" (no `zoompan`, no `drawtext`,
   libass captions, the gate fails closed) and the owned code.
2. Render path: `scale:eval=frame` + time-varying `crop` camera moves,
   concat, loudness normalisation, final probe. Verify with the sandbox's
   `imageio-ffmpeg` binary that every filter used exists (`ffmpeg -filters`,
   `-h filter=ass`); measure one 4 s clip render time and record it.
3. Captions: ASS karaoke from word timings; the font is fetched at runtime by
   `fonts.py` (`assets/fonts/*.ttf` is gitignored) — make the fetch robust
   (size or checksum check, cached, clear error when offline).
4. Gate (`qc.py`): mechanical checks first (duration, 1080×1920, loudness,
   shot count, word count, image variety), then the judge; if the judge is
   unreachable → do not publish. Prove the fail-closed behaviour with a test
   that stubs an unreachable judge and asserts no publish. Never add a
   "publish anyway" path.
5. Run `tests/run_offline.py` and probe the produced `final.mp4`
   (`ffprobe` dimensions/duration/streams); paste the probe.
6. Tests for every fix, mutation-checked.

End with `FLEET-RESULT`.
