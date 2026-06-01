# YouTube Shorts pipeline

Quick local runner. Keep secrets out of git. The workflow in `.github/workflows/main.yml` runs the pipeline on cron in GitHub Actions.

One-line run (replace the keys inline):

```bash
GEMINI_API_KEY="your_gemini_key" PEXELS_API_KEY="your_pexels_key" \
  YOUTUBE_CLIENT_SECRETS_JSON="$(cat client_secrets.json)" \
  YOUTUBE_CREDENTIALS_JSON="$(cat credentials.json)" \
  bash run_once.sh
```

Notes:
- Do NOT commit `credentials.json` or `client_secrets.json`. Add them to GitHub Secrets instead.
- The `run_once.sh` script will create a virtual environment and install dependencies.
