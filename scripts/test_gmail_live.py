"""
End-to-end test: fetch REAL emails from Gmail, run them through the
Experience Packs graph, coach the agent, and verify packs are created.

Usage:
    python scripts/test_gmail_live.py                      # last 60 min of emails
    python scripts/test_gmail_live.py --minutes 1440       # last 24 hours
    python scripts/test_gmail_live.py --max-emails 3       # limit to 3 emails
    python scripts/test_gmail_live.py --email you@gmail.com  # filter by recipient

Requirements:
    1. .secrets/token.json  (run scripts/setup_gmail.py first)
    2. OPENAI_API_KEY in .env or environment
"""

import argparse
import base64
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Gmail fetch (standalone — no dependency on deleted gmail tools)
# ---------------------------------------------------------------------------

SECRETS_DIR = Path(__file__).parent.parent / ".secrets"


def get_gmail_service():
    """Build Gmail API service from saved token."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        print("ERROR: Google API libraries not installed.")
        print("Run: pip install --user google-api-python-client google-auth-oauthlib")
        sys.exit(1)

    token_path = SECRETS_DIR / "token.json"
    if not token_path.exists():
        print(f"ERROR: No token found at {token_path}")
        print("Run: python scripts/setup_gmail.py")
        sys.exit(1)

    with open(token_path) as f:
        token_data = json.load(f)

    creds = Credentials(
        token=token_data["token"],
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=token_data.get("client_id"),
        client_secret=token_data.get("client_secret"),
        scopes=token_data.get("scopes", ["https://www.googleapis.com/auth/gmail.readonly"]),
    )

    # Refresh if expired
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_data["token"] = creds.token
        with open(token_path, "w") as f:
            json.dump(token_data, f, indent=2)

    return build("gmail", "v1", credentials=creds)


def extract_body(payload):
    """Recursively extract text from a Gmail message payload."""
    if payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
    if payload.get("parts"):
        texts = []
        for part in payload["parts"]:
            mime = part.get("mimeType", "")
            if "text/plain" in mime or "multipart" in mime:
                texts.append(extract_body(part))
        return "\n".join(t for t in texts if t)
    return ""


def fetch_emails(service, minutes_since=60, email_filter=None, max_emails=5):
    """Fetch recent emails from Gmail and return them in the graph's input format."""
    after_ts = int((datetime.now() - timedelta(minutes=minutes_since)).timestamp())
    query = f"after:{after_ts}"
    if email_filter:
        query += f" to:{email_filter}"
    query += " is:unread"

    print(f"[Gmail] query: {query}")
    results = service.users().messages().list(userId="me", q=query, maxResults=max_emails).execute()
    messages = results.get("messages", [])
    print(f"   Found {len(messages)} message(s)\n")

    emails = []
    for msg_ref in messages:
        msg = service.users().messages().get(userId="me", id=msg_ref["id"]).execute()
        headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
        body = extract_body(msg["payload"])

        # Truncate very long emails for the demo
        if len(body) > 3000:
            body = body[:3000] + "\n\n[... truncated ...]"

        email_input = {
            "author": headers.get("From", "unknown"),
            "to": headers.get("To", "unknown"),
            "subject": headers.get("Subject", "(no subject)"),
            "email_thread": body,
        }
        emails.append(email_input)

    return emails


# ---------------------------------------------------------------------------
# Run the graph
# ---------------------------------------------------------------------------


def run_graph_on_email(email_input, thread_id, store, checkpointer):
    """Invoke the experience graph on a single email using shared store."""
    # Import here so OPENAI_API_KEY must be set before this point
    from email_assistant.email_assistant_experience import (
        overall_workflow,
        State,
        StateInput,
    )

    # Compile with shared store + checkpointer
    graph = overall_workflow.compile(store=store, checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}}

    print(f"> Running graph for: {email_input['subject']}")
    print(f"  From: {email_input['author']}")

    # Show email body right after From
    body = email_input.get("email_thread", "")
    if body:
        preview = body[:500] + "..." if len(body) > 500 else body
        print(f"  [Email Body]")
        for line in preview.splitlines():
            print(f"     {line}")
    print()

    try:
        result = graph.invoke({"email_input": email_input}, config=config)
    except Exception as e:
        # interrupt() raises GraphInterrupt when HITL is needed
        # In a real setup you'd use Agent Inbox; here we handle it gracefully
        if "GraphInterrupt" in type(e).__name__ or "interrupt" in str(e).lower():
            print(f"  [PAUSED] Graph paused for human review (interrupt)")
            print(f"     In production, open Agent Inbox to respond.")
            print()

            # Get the state so far to show the trace
            state = graph.get_state(config)
            result = state.values
        else:
            raise

    # Show trace
    trace = result.get("trace", [])
    if trace:
        print(f"  [Trace] Reasoning Trace ({len(trace)} step(s)):")
        for t in trace:
            print(f"     Step {t.step_id}: {t.objective}")
            print(f"       -> Chose: {t.chosen_option}")
            print(f"       -> Rationale: {t.rationale}")
            print(f"       -> Confidence: {t.confidence}")
        print()

    classification = result.get("classification_decision", "N/A")
    print(f"  [Classification] {classification}")

    # Show experience packs used
    packs = result.get("experience_packs", [])
    if packs:
        print(f"  [Packs] Experience Packs injected: {len(packs)}")
        for p in packs:
            print(f"     - {p.trigger_context[:80]}")
    else:
        print(f"  [Packs] No Experience Packs matched (library empty or no relevant packs)")

    print()
    return result


# ---------------------------------------------------------------------------
# Interactive coaching (terminal-based for this test script)
# ---------------------------------------------------------------------------


def run_coaching(email_input, trace, store):
    """Simplified terminal coaching — creates an ExperiencePack from feedback."""
    from email_assistant.experience_packs import (
        ExperiencePack,
        ExperienceLibrary,
        COACHING_PACK_PROMPT,
    )
    from langchain.chat_models import init_chat_model

    print("=" * 60)
    print("COACHING SESSION")
    print("=" * 60)
    print()
    print(f"Email: {email_input['subject']}")
    print(f"From:  {email_input['author']}")
    print()

    if trace:
        print("Trace summary:")
        for t in trace:
            print(f"  Step {t.step_id}: chose '{t.chosen_option}' "
                  f"(confidence {t.confidence}) — {t.rationale[:100]}")
        print()

    feedback = input("Coach feedback (or press Enter to skip): ").strip()
    if not feedback:
        print("   Skipped coaching.\n")
        return None

    # Generate pack via LLM
    trace_json = json.dumps([t.model_dump() for t in trace[-3:]], indent=2, default=str)
    email_json = json.dumps(email_input, indent=2)

    prompt = COACHING_PACK_PROMPT.format(
        trace_json=trace_json,
        email_json=email_json,
        feedback=feedback,
    )

    llm = init_chat_model("openai:gpt-4.1", temperature=0.0).with_structured_output(
        ExperiencePack
    )
    new_pack = llm.invoke(prompt)

    lib = ExperienceLibrary(store)
    lib.add_pack(new_pack)

    print(f"\n[OK] Created Experience Pack: {new_pack.pack_id}")
    print(f"   Trigger: {new_pack.trigger_context}")
    print(f"   Directive: {new_pack.directive}")
    print(f"   Confidence: {new_pack.confidence}")
    print()

    return new_pack


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Test Experience Packs with real Gmail emails")
    parser.add_argument("--minutes", type=int, default=60, help="Fetch emails from last N minutes (default: 60)")
    parser.add_argument("--max-emails", type=int, default=5, help="Max emails to process (default: 5)")
    parser.add_argument("--email", type=str, default=None, help="Filter by recipient email address")
    parser.add_argument("--coach", action="store_true", help="Enable interactive coaching after each email")
    args = parser.parse_args()

    # Verify API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
        return 1

    # Connect to Gmail
    print("[*] Connecting to Gmail API...\n")
    service = get_gmail_service()

    # Fetch emails
    emails = fetch_emails(service, args.minutes, args.email, args.max_emails)
    if not emails:
        print("No emails found. Try increasing --minutes or removing --email filter.")
        return 0

    # Shared store + checkpointer across all emails in this run
    from langgraph.store.memory import InMemoryStore
    from langgraph.checkpoint.memory import MemorySaver
    from email_assistant.experience_packs import (
        seed_store_from_file, sync_store_to_file, PACKS_FILE_DEFAULT,
    )

    store = InMemoryStore()
    checkpointer = MemorySaver()

    # Load any previously saved packs from disk into the store
    existing = seed_store_from_file(store)
    if existing:
        print(f"[Packs] Loaded {len(existing)} existing pack(s) from {PACKS_FILE_DEFAULT}\n")

    print(f"{'='*60}")
    print(f"Processing {len(emails)} email(s) through Experience Packs graph")
    print(f"{'='*60}\n")

    store_ref = store

    for i, email_input in enumerate(emails):
        print(f"--- Email {i+1}/{len(emails)} ---")
        result = run_graph_on_email(email_input, thread_id=f"gmail-test-{i}",
                                    store=store, checkpointer=checkpointer)

        if args.coach:
            trace = result.get("trace", [])
            run_coaching(email_input, trace, store)

    # Summary
    if store_ref and args.coach:
        from email_assistant.experience_packs import ExperienceLibrary
        lib = ExperienceLibrary(store_ref)
        all_packs = lib.list_packs()
        print(f"\n{'='*60}")
        print(f"[Packs] Experience Pack Library: {len(all_packs)} pack(s)")
        print(f"{'='*60}")
        for p in all_packs:
            print(f"  [{p.pack_id}] {p.trigger_context[:60]}")
            print(f"    -> {p.directive[:80]}")
            print(f"    usage: {p.usage_count}, confidence: {p.confidence}")
            print()

    # Persist packs to disk so they survive across runs
    saved = sync_store_to_file(store_ref)
    if saved:
        print(f"[Saved] {len(saved)} pack(s) written to {PACKS_FILE_DEFAULT}")

    print("[OK] Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
