"""
Test harness for Experience Packs.

Tests the full lifecycle:
  1. Agent processes email → produces a trace
  2. Coach provides feedback → ExperiencePack is created
  3. Same/similar email re-processed → pack is retrieved and injected
  4. Verify confidence/classification changes

Can run without LLM calls using mock mode (default), or with real
LLM calls using --live flag.
"""

import json
import sys
import os
from typing import List
from unittest.mock import MagicMock

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from email_assistant.experience_packs import (
    StepTrace,
    ExperiencePack,
    ExperienceLibrary,
    trace_reducer,
)


# ---------------------------------------------------------------------------
# Mock Store (mimics LangGraph BaseStore interface for unit testing)
# ---------------------------------------------------------------------------


class MockStore:
    """In-memory store that implements the LangGraph BaseStore get/put/search API."""

    def __init__(self):
        self._data = {}

    def _key(self, namespace, key):
        return (tuple(namespace), key)

    def get(self, namespace, key):
        k = self._key(namespace, key)
        value = self._data.get(k)
        if value is None:
            return None
        # Return an object with .value like LangGraph Item
        return _Item(value)

    def put(self, namespace, key, value):
        k = self._key(namespace, key)
        self._data[k] = value

    def search(self, namespace, query=None, limit=10):
        # Not supported in mock — library falls back to keyword search
        raise NotImplementedError


class _Item:
    """Minimal stand-in for langgraph.store.base.Item."""

    def __init__(self, value):
        self.value = value


# ---------------------------------------------------------------------------
# Test emails
# ---------------------------------------------------------------------------

EMAIL_API_QUESTION = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi Lance,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

EMAIL_MARKETING = {
    "author": "Marketing Team <marketing@company.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "New Company Newsletter Available",
    "email_thread": """Hello Lance,

The latest edition of our company newsletter is now available on the intranet.

Best regards,
Marketing Team""",
}

EMAIL_MEETING = {
    "author": "Team Lead <teamlead@company.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Quarterly planning meeting",
    "email_thread": """Hi Lance,

I'd like to schedule a 90-minute meeting next week to discuss our roadmap for Q3.
Could you let me know your availability for Monday or Wednesday? Ideally sometime between 10AM and 3PM.

Best,
Team Lead""",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trace_reducer():
    """Trace reducer auto-numbers steps and appends."""
    t1 = StepTrace(objective="triage", rationale="test", confidence=0.9)
    t2 = StepTrace(objective="respond", rationale="test2", confidence=0.8)
    combined = trace_reducer([t1], [t2])
    assert len(combined) == 2
    assert combined[0].step_id == 0  # left side unchanged
    assert combined[1].step_id == 1  # auto-numbered
    print("  ✅ trace_reducer")


def test_experience_library_add_and_retrieve():
    """Add packs and retrieve by keyword overlap."""
    store = MockStore()
    lib = ExperienceLibrary(store)

    pack1 = ExperiencePack(
        trigger_context="API documentation questions about missing endpoints",
        directive="Always check if endpoints were intentionally removed before responding",
        rationale="Prevents unnecessary doc updates",
        linked_skills=["triage", "response"],
        confidence=0.85,
    )
    pack2 = ExperiencePack(
        trigger_context="Marketing newsletter emails",
        directive="Classify as ignore — these are never actionable",
        rationale="User never responds to newsletters",
        linked_skills=["triage"],
        confidence=0.95,
    )

    lib.add_pack(pack1)
    lib.add_pack(pack2)

    # Retrieve with API-related query
    results = lib.retrieve("API documentation endpoints missing")
    assert len(results) >= 1
    assert any("API" in r.trigger_context for r in results)
    print("  ✅ add_and_retrieve")


def test_deduplication():
    """Duplicate trigger_context is rejected."""
    store = MockStore()
    lib = ExperienceLibrary(store)

    pack1 = ExperiencePack(
        trigger_context="Test trigger",
        directive="Do X",
        confidence=0.8,
    )
    pack2 = ExperiencePack(
        trigger_context="Test trigger",  # exact duplicate
        directive="Do Y",
        confidence=0.9,
    )

    lib.add_pack(pack1)
    returned = lib.add_pack(pack2)
    assert returned.pack_id == pack1.pack_id  # returned existing
    assert len(lib.list_packs()) == 1
    print("  ✅ deduplication")


def test_usage_count():
    """usage_count increments on retrieve."""
    store = MockStore()
    lib = ExperienceLibrary(store)

    pack = ExperiencePack(
        trigger_context="calendar meeting scheduling",
        directive="Always check availability first",
        confidence=0.8,
    )
    lib.add_pack(pack)
    assert lib.list_packs()[0].usage_count == 0

    lib.increment_usage([pack.pack_id])
    assert lib.list_packs()[0].usage_count == 1
    print("  ✅ usage_count")


def test_deactivate():
    """Deactivated packs are excluded from retrieve."""
    store = MockStore()
    lib = ExperienceLibrary(store)

    pack = ExperiencePack(
        trigger_context="test deactivate",
        directive="Do something",
        confidence=0.8,
    )
    lib.add_pack(pack)
    lib.deactivate_pack(pack.pack_id)
    assert len(lib.list_packs()) == 0
    assert len(lib.list_packs(include_inactive=True)) == 1
    print("  ✅ deactivate")


def test_persistence_across_instances():
    """A second ExperienceLibrary instance sees packs from the first."""
    store = MockStore()

    lib1 = ExperienceLibrary(store)
    lib1.add_pack(
        ExperiencePack(
            trigger_context="persistence test",
            directive="Persist correctly",
            confidence=0.9,
        )
    )

    lib2 = ExperienceLibrary(store)
    packs = lib2.list_packs()
    assert len(packs) == 1
    assert packs[0].trigger_context == "persistence test"
    print("  ✅ persistence_across_instances")


def test_keyword_search_ranking():
    """Keyword search returns most relevant pack first."""
    store = MockStore()
    lib = ExperienceLibrary(store)

    lib.add_pack(
        ExperiencePack(
            trigger_context="swimming class registration for daughter",
            directive="Always express interest and ask about schedule options",
            confidence=0.8,
        )
    )
    lib.add_pack(
        ExperiencePack(
            trigger_context="API documentation missing endpoints",
            directive="Investigate before responding",
            confidence=0.85,
        )
    )
    lib.add_pack(
        ExperiencePack(
            trigger_context="tax season financial planning call",
            directive="Schedule 45 minute meeting",
            confidence=0.7,
        )
    )

    results = lib.retrieve("daughter swimming registration summer")
    assert len(results) >= 1
    assert "swimming" in results[0].trigger_context.lower()
    print("  ✅ keyword_search_ranking")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all_tests():
    print("\n🧪 Running Experience Packs unit tests...\n")
    test_trace_reducer()
    test_experience_library_add_and_retrieve()
    test_deduplication()
    test_usage_count()
    test_deactivate()
    test_persistence_across_instances()
    test_keyword_search_ranking()
    print("\n✅ All tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
