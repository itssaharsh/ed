#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# If the GitHub-style secrets are present in env, materialize them as files for local run
if [ -n "${YOUTUBE_CREDENTIALS_JSON:-}" ] && [ ! -f credentials.json ]; then
  printf '%s' "$YOUTUBE_CREDENTIALS_JSON" > credentials.json
fi
if [ -n "${YOUTUBE_CLIENT_SECRETS_JSON:-}" ] && [ ! -f client_secrets.json ]; then
  printf '%s' "$YOUTUBE_CLIENT_SECRETS_JSON" > client_secrets.json
fi

if [ -z "${GEMINI_API_KEY:-}" ] || [ -z "${PEXELS_API_KEY:-}" ]; then
  echo "ERROR: Please provide GEMINI_API_KEY and PEXELS_API_KEY in environment variables." >&2
  echo "Example (single-line): GEMINI_API_KEY=... PEXELS_API_KEY=... bash run_once.sh" >&2
  exit 1
fi

python3 main.py
