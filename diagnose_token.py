"""
diagnose_token.py
-----------------
Reads credentials.json (passed via YOUTUBE_CREDENTIALS_PATH or auto-detected in CWD),
then directly probes Google's OAuth2 token endpoints to determine whether the
refresh_token is still valid — without requiring the full google-api-python-client stack.

Usage:
  python diagnose_token.py
  python diagnose_token.py path/to/credentials.json
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from datetime import datetime, timezone


GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


def load_credentials(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8").strip()
    data = json.loads(raw)
    return data


def probe_token_info(access_token: str) -> dict | None:
    """Hit /tokeninfo with the current access_token (may already be expired)."""
    try:
        url = f"{GOOGLE_TOKENINFO_URL}?access_token={urllib.parse.quote(access_token)}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return {"_http_error": e.code, "_body": body}
    except Exception as e:
        return {"_error": str(e)}


def probe_refresh(client_id: str, client_secret: str, refresh_token: str) -> dict:
    """Attempt to exchange the refresh_token for a new access_token."""
    payload = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }).encode()
    try:
        req = urllib.request.Request(
            GOOGLE_TOKEN_URL,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"_status": resp.status, **json.loads(resp.read().decode())}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            body_json = json.loads(body)
        except Exception:
            body_json = {"raw": body}
        return {"_http_error": e.code, **body_json}
    except Exception as e:
        return {"_error": str(e)}


def main() -> None:
    # Resolve credentials path
    if len(sys.argv) > 1:
        creds_path = Path(sys.argv[1])
    else:
        env_path = os.environ.get("YOUTUBE_CREDENTIALS_PATH")
        creds_path = Path(env_path) if env_path else Path("credentials.json")

    print(f"=== diagnose_token.py ===")
    print(f"Reading: {creds_path.resolve()}")
    print()

    if not creds_path.exists():
        print("ERROR: credentials.json not found.")
        print("  This file only exists after running get_credentials.py locally.")
        print("  The YOUTUBE_CREDENTIALS_JSON GitHub secret is never written to disk here.")
        sys.exit(1)

    creds = load_credentials(creds_path)

    # ── Field audit ──────────────────────────────────────────────────────────
    REQUIRED = ["token", "refresh_token", "token_uri", "client_id", "client_secret", "scopes"]
    print("--- Field audit ---")
    all_present = True
    for field in REQUIRED:
        val = creds.get(field)
        if val:
            display = f"[PRESENT, len={len(str(val))}]"
        else:
            display = "*** MISSING or EMPTY ***"
            all_present = False
        print(f"  {field}: {display}")

    # Expiry check
    expiry_raw = creds.get("expiry")
    print(f"  expiry: {expiry_raw}")
    if expiry_raw:
        try:
            # google-auth stores expiry as "2026-01-01T00:00:00Z" or with microseconds
            expiry_raw_clean = expiry_raw.rstrip("Z").split(".")[0]
            expiry_dt = datetime.fromisoformat(expiry_raw_clean).replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = expiry_dt - now
            if delta.total_seconds() < 0:
                print(f"  ⚠  Access token is EXPIRED by {abs(int(delta.total_seconds()))} seconds. Refresh needed.")
            else:
                print(f"  ✓  Access token is still valid for {int(delta.total_seconds())} seconds.")
        except Exception as e:
            print(f"  Could not parse expiry: {e}")

    print()
    if not all_present:
        print("FATAL: credentials.json is missing required fields. Re-run get_credentials.py.")
        sys.exit(1)

    # ── Token info probe ─────────────────────────────────────────────────────
    access_token = creds.get("token", "")
    if access_token:
        print("--- Probing /tokeninfo with current access_token ---")
        info = probe_token_info(access_token)
        if "_http_error" in info:
            print(f"  HTTP {info['_http_error']}: {info.get('_body', info)}")
        elif "_error" in info:
            print(f"  Network error: {info['_error']}")
        else:
            print(f"  scope: {info.get('scope', 'N/A')}")
            print(f"  email: {info.get('email', 'N/A')}")
            print(f"  expires_in: {info.get('expires_in', 'N/A')} seconds")
            print(f"  azp (client_id match): {info.get('azp', 'N/A')}")
        print()

    # ── Refresh probe ────────────────────────────────────────────────────────
    client_id = creds.get("client_id", "")
    client_secret = creds.get("client_secret", "")
    refresh_token = creds.get("refresh_token", "")

    print("--- Probing /token refresh endpoint (this is the exact call that failed in CI) ---")
    result = probe_refresh(client_id, client_secret, refresh_token)

    if "_http_error" in result:
        error_code = result.get("error", "unknown")
        error_desc = result.get("error_description", "no description")
        print(f"  ✗ HTTP {result['_http_error']} — error: '{error_code}' — description: '{error_desc}'")
        print()
        if error_code == "invalid_grant":
            print("  CONFIRMED ROOT CAUSE: refresh_token has been permanently revoked by Google.")
            print()
            print("  Most likely reasons (in order of probability):")
            print("  1. This refresh_token was created while your app was in 'Testing' mode.")
            print("     Even though the app is now 'In production', OLD tokens from testing are dead.")
            print("     FIX: Re-run get_credentials.py → a fresh Production-mode token will be issued.")
            print("  2. Refresh token was not used for > 6 months.")
            print("     FIX: Same — re-run get_credentials.py.")
            print("  3. client_id/client_secret mismatch between the secret and current OAuth client.")
            print("     FIX: Re-download client_secrets.json from Google Cloud Console and re-auth.")
        elif error_code == "invalid_client":
            print("  ROOT CAUSE: client_id or client_secret is wrong.")
            print("  Re-download client_secrets.json from Google Cloud Console.")
    elif "_error" in result:
        print(f"  Network error: {result['_error']}")
    else:
        new_token = result.get("access_token", "")
        scope = result.get("scope", "")
        expires = result.get("expires_in", "?")
        print(f"  ✓ Refresh SUCCEEDED — new access_token obtained.")
        print(f"  scope: {scope}")
        print(f"  expires_in: {expires} seconds")
        print()
        print("  The refresh_token itself is VALID. The CI failure must have been transient.")
        print("  Re-save credentials.json content to YOUTUBE_CREDENTIALS_JSON GitHub secret.")

    print()
    print("=== Diagnosis complete ===")


if __name__ == "__main__":
    main()
