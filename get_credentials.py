"""Run a one-time OAuth flow to get YouTube API `credentials.json`.

Usage:
  python3 get_credentials.py

This opens a local browser to complete OAuth and writes `credentials.json`.
Do NOT commit `credentials.json` to your repository; instead, copy its contents
into the `YOUTUBE_CREDENTIALS_JSON` GitHub secret for scheduled runs.
"""
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse, unquote

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def main() -> None:
    secrets = Path("client_secrets.json")
    if not secrets.exists():
        print("client_secrets.json not found in current folder. Place your client JSON here.")
        return

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
    # Allow forcing the console (copy-paste) flow when the environment can't use a loopback server.
    if os.getenv("FORCE_CONSOLE") == "1":
        print("FORCE_CONSOLE=1: using console OAuth flow. Open the URL, authorize, then paste the code.")
        auth_url, _ = flow.authorization_url(access_type="offline", include_granted_scopes="true")
        print(auth_url)
        result_url = os.getenv("AUTH_RESULT_URL")
        if result_url:
            print("Using AUTH_RESULT_URL from environment.")
            parsed_result = urlparse(result_url.strip())
            result_qs = parse_qs(parsed_result.query)
            code_values = result_qs.get("code", [])
            if not code_values:
                raise ValueError("AUTH_RESULT_URL must be the redirected localhost URL that contains ?code=... in the query string")
            code = code_values[0]
            redirect_uri = f"{parsed_result.scheme}://{parsed_result.netloc}{parsed_result.path}"
        else:
            auth_code_env = os.getenv("AUTH_CODE")
            if auth_code_env:
                code = auth_code_env.strip()
                print("Using AUTH_CODE from environment.")
            else:
                code = input("Enter the authorization code: ").strip()
            parsed = urlparse(auth_url)
            qs = parse_qs(parsed.query)
            redirect_uri = None
            if "redirect_uri" in qs:
                redirect_uri = unquote(qs["redirect_uri"][0])
        if redirect_uri:
            flow.fetch_token(code=code, redirect_uri=redirect_uri)
        else:
            flow.fetch_token(code=code)
        creds = flow.credentials
    else:
        try:
            creds = flow.run_local_server(port=0)
        except Exception:
            print("run_local_server failed (likely redirect_uri mismatch). Falling back to console flow.")
            print("Open the URL below in your browser, complete sign-in, then paste the authorization code here.")
            auth_url, _ = flow.authorization_url(access_type="offline", include_granted_scopes="true")
            print(auth_url)
            result_url = os.getenv("AUTH_RESULT_URL")
            if result_url:
                print("Using AUTH_RESULT_URL from environment.")
                parsed_result = urlparse(result_url.strip())
                result_qs = parse_qs(parsed_result.query)
                code_values = result_qs.get("code", [])
                if not code_values:
                    raise ValueError("AUTH_RESULT_URL did not contain a code parameter")
                code = code_values[0]
                redirect_uri = f"{parsed_result.scheme}://{parsed_result.netloc}{parsed_result.path}"
            else:
                code = input("Enter the authorization code: ").strip()
                parsed = urlparse(auth_url)
                qs = parse_qs(parsed.query)
                redirect_uri = None
                if "redirect_uri" in qs:
                    redirect_uri = unquote(qs["redirect_uri"][0])
            if redirect_uri:
                flow.fetch_token(code=code, redirect_uri=redirect_uri)
            else:
                flow.fetch_token(code=code)
            creds = flow.credentials
    out = Path("credentials.json")
    out.write_text(creds.to_json(), encoding="utf-8")
    print(f"Saved {out.resolve()}. Do NOT commit this file; add its contents to the GitHub secret YOUTUBE_CREDENTIALS_JSON.")


if __name__ == "__main__":
    main()
