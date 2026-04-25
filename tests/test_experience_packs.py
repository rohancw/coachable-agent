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
    save_traces_to_file,
    load_traces_from_file,
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
        return _Item(value, key)

    def put(self, namespace, key, value):
        k = self._key(namespace, key)
        self._data[k] = value

    def search(self, namespace, query=None, limit=10):
        """Return all items in the namespace (keyword filtering is done by ExperienceLibrary)."""
        ns = tuple(namespace)
        results = []
        for (stored_ns, key), value in self._data.items():
            if stored_ns == ns and isinstance(value, dict):
                results.append(_Item(value, key))
        return results[:limit]


class _Item:
    """Minimal stand-in for langgraph.store.base.Item."""

    def __init__(self, value, key=""):
        self.value = value
        self.key = key


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
    print("  [ok] trace_reducer")


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
    print("  [ok] add_and_retrieve")


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
    print("  [ok] deduplication")


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
    print("  [ok] usage_count")


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
    print("  [ok] deactivate")


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
    print("  [ok] persistence_across_instances")


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
    print("  [ok] keyword_search_ranking")


def test_pack_specificity_fields():
    """New fields: applicability_criteria, negative_examples, source_trace_ids."""
    pack = ExperiencePack(
        trigger_context="Emails from known personal contacts",
        directive="Always notify",
        applicability_criteria=["sender is a known personal contact"],
        negative_examples=["sender is an automated mailing list"],
        source_trace_ids=["1", "2"],
        confidence=0.9,
    )
    assert pack.applicability_criteria == ["sender is a known personal contact"]
    assert pack.negative_examples == ["sender is an automated mailing list"]
    assert pack.source_trace_ids == ["1", "2"]

    # Ensure they serialize/deserialize correctly
    dumped = pack.model_dump()
    restored = ExperiencePack.model_validate(dumped)
    assert restored.applicability_criteria == pack.applicability_criteria
    assert restored.negative_examples == pack.negative_examples
    assert restored.source_trace_ids == pack.source_trace_ids
    print("  [ok] pack_specificity_fields")


def test_individual_pack_storage():
    """Each pack is stored under its own key, not as a single blob."""
    store = MockStore()
    lib = ExperienceLibrary(store)

    pack1 = ExperiencePack(
        trigger_context="Test pack one",
        directive="Do X",
        confidence=0.8,
    )
    pack2 = ExperiencePack(
        trigger_context="Test pack two",
        directive="Do Y",
        confidence=0.9,
    )

    lib.add_pack(pack1)
    lib.add_pack(pack2)

    # Verify each pack is stored under its own key
    from email_assistant.experience_packs import EXPERIENCE_NAMESPACE
    item1 = store.get(EXPERIENCE_NAMESPACE, pack1.pack_id)
    item2 = store.get(EXPERIENCE_NAMESPACE, pack2.pack_id)
    assert item1 is not None
    assert item2 is not None
    assert item1.value["pack_id"] == pack1.pack_id
    assert item2.value["pack_id"] == pack2.pack_id

    # Verify list_packs finds both
    all_packs = lib.list_packs()
    assert len(all_packs) == 2
    print("  [ok] individual_pack_storage")


def test_trace_persistence():
    """Traces are saved to and loaded from JSONL."""
    import tempfile
    import os

    traces = [
        StepTrace(objective="triage", rationale="test", confidence=0.9),
        StepTrace(objective="respond", rationale="test2", confidence=0.8),
    ]
    email_input = {"subject": "Test email", "author": "test@test.com"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name

    try:
        save_traces_to_file(traces, email_input, "run-1", path=path)
        save_traces_to_file(traces, email_input, "run-2", path=path)

        records = load_traces_from_file(path=path)
        assert len(records) == 2
        assert records[0]["run_id"] == "run-1"
        assert records[1]["run_id"] == "run-2"
        assert len(records[0]["traces"]) == 2
        print("  [ok] trace_persistence")
    finally:
        os.unlink(path)


def test_migration_from_blob():
    """Packs stored as a single blob are migrated to individual keys."""
    store = MockStore()

    # Simulate old-style blob storage
    from email_assistant.experience_packs import EXPERIENCE_NAMESPACE, EXPERIENCE_KEY
    pack1 = ExperiencePack(
        trigger_context="Old pack one",
        directive="Do X",
        confidence=0.8,
    )
    pack2 = ExperiencePack(
        trigger_context="Old pack two",
        directive="Do Y",
        confidence=0.9,
    )
    store.put(EXPERIENCE_NAMESPACE, EXPERIENCE_KEY, [pack1.model_dump(), pack2.model_dump()])

    # Creating ExperienceLibrary triggers migration
    lib = ExperienceLibrary(store)

    # Old blob should be cleared
    blob = store.get(EXPERIENCE_NAMESPACE, EXPERIENCE_KEY)
    assert blob.value == []

    # Packs should be individually stored and retrievable
    all_packs = lib.list_packs()
    assert len(all_packs) == 2
    print("  [ok] migration_from_blob")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all_tests():
    print("\nRunning Experience Packs unit tests...\n")
    test_trace_reducer()
    test_experience_library_add_and_retrieve()
    test_deduplication()
    test_usage_count()
    test_deactivate()
    test_persistence_across_instances()
    test_keyword_search_ranking()
    test_pack_specificity_fields()
    test_individual_pack_storage()
    test_trace_persistence()
    test_migration_from_blob()
    print("\n[ok] All tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
