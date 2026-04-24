"""
OAuth setup for Gmail API.

Run this once to authenticate with Google and save credentials:
    python scripts/setup_gmail.py

Creates .secrets/token.json which is used by test_gmail_live.py.
"""

import json
import os
import sys
from pathlib import Path

# Scopes needed: read-only Gmail access (we never send real emails)
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
]

SECRETS_DIR = Path(__file__).parent.parent / ".secrets"


def main():
    # Check for google auth libraries
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("ERROR: google-auth-oauthlib not installed.")
        print("Run: pip install --user google-auth-oauthlib")
        return 1

    SECRETS_DIR.mkdir(parents=True, exist_ok=True)

    credentials_path = SECRETS_DIR / "credentials.json"
    token_path = SECRETS_DIR / "token.json"

    if not credentials_path.exists():
        print(f"ERROR: OAuth credentials file not found at:\n  {credentials_path}")
        print()
        print("To create it:")
        print("  1. Go to https://console.cloud.google.com/apis/credentials")
        print("  2. Create an OAuth 2.0 Client ID (Desktop app)")
        print("  3. Download the JSON file")
        print(f"  4. Save it as: {credentials_path}")
        return 1

    print("Starting OAuth flow — a browser window will open...")
    print(f"Scopes requested: {SCOPES}")
    print()

    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
    creds = flow.run_local_server(port=0)

    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or SCOPES),
    }

    with open(token_path, "w") as f:
        json.dump(token_data, f, indent=2)

    print(f"✅ Token saved to {token_path}")
    print("You can now run: python scripts/test_gmail_live.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
