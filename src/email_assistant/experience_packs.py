"""
Experience Packs: Structured reasoning traces, reusable coaching lessons,
and a retrieval library that integrates with LangGraph Store.

All models follow LangGraph conventions:
- Nodes return partial state dicts (never mutate state directly)
- Persistence uses LangGraph Store (same as existing memory)
- Pydantic v2 models throughout
"""

import json
import operator
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

class StepTrace(BaseModel):
    """Structured reasoning trace recorded at each decision point."""

    step_id: int = 0
    objective: str = Field(..., description="What the agent is trying to achieve right now")
    instructions_received: str = Field(default="", description="Key instructions in play")
    options_considered: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    chosen_option: str = Field(default="")
    rationale: str = Field(..., description="Why this option was chosen")
    confidence: float = Field(..., ge=0, le=1)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExperiencePack(BaseModel):
    """One reusable coaching lesson — a directive the agent should follow
    when it encounters a similar situation in the future."""

    pack_id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    trigger_context: str = Field(
        ..., description="Summary of the situation that triggers this pack"
    )
    directive: str = Field(
        ..., description="How the agent should behave next time"
    )
    rationale: str = Field(default="")
    linked_skills: List[str] = Field(default_factory=list)
    applicability_criteria: List[str] = Field(
        default_factory=list,
        description="Conditions under which this pack should apply",
    )
    negative_examples: List[str] = Field(
        default_factory=list,
        description="Situations where this pack should NOT apply",
    )
    source_trace_ids: List[str] = Field(
        default_factory=list,
        description="Step IDs of the traces that generated this pack",
    )
    confidence: float = Field(default=0.5, ge=0, le=1)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    active: bool = True


# ---------------------------------------------------------------------------
# Trace reducer  (for Annotated[List[StepTrace], trace_reducer])
# ---------------------------------------------------------------------------

def trace_reducer(left: List[StepTrace], right: List[StepTrace]) -> List[StepTrace]:
    """Append new traces, auto-numbering step_id."""
    combined = list(left)
    next_id = (max((t.step_id for t in combined), default=0)) + 1
    for t in right:
        t.step_id = next_id
        combined.append(t)
        next_id += 1
    return combined


# ---------------------------------------------------------------------------
# Experience Library — persistence via LangGraph Store
# ---------------------------------------------------------------------------

EXPERIENCE_NAMESPACE = ("email_assistant", "experience_packs")
EXPERIENCE_KEY = "library"  # legacy key for migration


class ExperienceLibrary:
    """Manages Experience Packs backed by LangGraph Store.

    Each pack is stored individually under its pack_id so that
    semantic search can operate at pack granularity.

    Usage inside graph nodes that receive ``store: BaseStore``:

        lib = ExperienceLibrary(store)
        lib.add_pack(pack)
        relevant = lib.retrieve(email_body)
    """

    def __init__(self, store):
        self.store = store
        self._maybe_migrate()

    # -- migration from single-blob to individual packs --------------------

    def _maybe_migrate(self):
        """If packs are stored under the old single-blob key, split them."""
        item = self.store.get(EXPERIENCE_NAMESPACE, EXPERIENCE_KEY)
        if item and item.value and isinstance(item.value, list) and len(item.value) > 0:
            for raw in item.value:
                pack = ExperiencePack.model_validate(raw)
                self.store.put(
                    EXPERIENCE_NAMESPACE,
                    pack.pack_id,
                    pack.model_dump(),
                )
            # Clear the old blob
            self.store.put(EXPERIENCE_NAMESPACE, EXPERIENCE_KEY, [])

    # -- persistence helpers ------------------------------------------------

    def _load_packs(self) -> List[ExperiencePack]:
        """Load all individual packs from the store."""
        packs = []
        try:
            results = self.store.search(EXPERIENCE_NAMESPACE, query="", limit=100)
            for item in results:
                if isinstance(item.value, dict) and "pack_id" in item.value:
                    packs.append(ExperiencePack.model_validate(item.value))
        except Exception:
            # Fallback: try the old single-blob key
            item = self.store.get(EXPERIENCE_NAMESPACE, EXPERIENCE_KEY)
            if item and item.value and isinstance(item.value, list):
                for raw in item.value:
                    packs.append(ExperiencePack.model_validate(raw))
        return packs

    def _save_pack(self, pack: ExperiencePack):
        """Save a single pack under its own key."""
        self.store.put(
            EXPERIENCE_NAMESPACE,
            pack.pack_id,
            pack.model_dump(),
        )

    # -- public API ---------------------------------------------------------

    def add_pack(self, pack: ExperiencePack) -> ExperiencePack:
        """Add a pack, deduplicating against existing packs by trigger similarity."""
        packs = self._load_packs()

        # Simple dedup: reject if an active pack has identical trigger_context
        for existing in packs:
            if existing.active and existing.trigger_context.strip().lower() == pack.trigger_context.strip().lower():
                print(f"  [!] Duplicate trigger -- skipping pack {pack.pack_id}")
                return existing

        self._save_pack(pack)
        print(f"  [+] Added Experience Pack {pack.pack_id}")
        return pack

    def retrieve(self, query: str, k: int = 3) -> List[ExperiencePack]:
        """Retrieve the most relevant active packs for a given email/query.

        Uses LangGraph Store's built-in semantic search when available,
        falling back to keyword overlap scoring.
        """
        packs = self._load_packs()
        active_packs = [p for p in packs if p.active]
        if not active_packs:
            return []

        # Try store.search with the actual query
        try:
            results = self.store.search(EXPERIENCE_NAMESPACE, query=query, limit=k * 2)
            if results:
                found = []
                for item in results:
                    if isinstance(item.value, dict) and "pack_id" in item.value:
                        pack = ExperiencePack.model_validate(item.value)
                        if pack.active:
                            found.append(pack)
                if found:
                    return found[:k]
        except Exception:
            pass

        # Fallback: simple keyword overlap scoring
        return self._keyword_search(query, active_packs, k)

    def increment_usage(self, pack_ids: List[str]):
        """Bump usage_count for the given pack IDs."""
        for pack_id in pack_ids:
            item = self.store.get(EXPERIENCE_NAMESPACE, pack_id)
            if item and item.value and isinstance(item.value, dict):
                pack = ExperiencePack.model_validate(item.value)
                pack.usage_count += 1
                self._save_pack(pack)

    def deactivate_pack(self, pack_id: str):
        """Retire a pack (soft delete)."""
        item = self.store.get(EXPERIENCE_NAMESPACE, pack_id)
        if item and item.value and isinstance(item.value, dict):
            pack = ExperiencePack.model_validate(item.value)
            pack.active = False
            self._save_pack(pack)

    def list_packs(self, include_inactive: bool = False) -> List[ExperiencePack]:
        packs = self._load_packs()
        if include_inactive:
            return packs
        return [p for p in packs if p.active]

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _keyword_search(
        query: str, packs: List[ExperiencePack], k: int
    ) -> List[ExperiencePack]:
        """Rank packs by token overlap with the query."""
        query_tokens = set(query.lower().split())
        scored = []
        for p in packs:
            pack_tokens = set(
                (p.trigger_context + " " + p.directive).lower().split()
            )
            overlap = len(query_tokens & pack_tokens)
            scored.append((overlap, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:k] if _ > 0]


# ---------------------------------------------------------------------------
# Coaching prompt (used by the coaching node to generate an ExperiencePack)
# ---------------------------------------------------------------------------

COACHING_PACK_PROMPT = """You are an AI coaching assistant.  Given the structured reasoning trace
from a previous email handling attempt and the human coach's feedback, create ONE ExperiencePack
that captures the lesson so the agent behaves better next time.

## Trace (last 3 steps)
{trace_json}

## Email
{email_json}

## Coach Feedback
{feedback}

Produce a single ExperiencePack JSON object with these fields:
- trigger_context: a concise summary of the situation type
- directive: clear, actionable instruction for the agent
- rationale: why this matters
- linked_skills: list of relevant skill names (e.g. "triage", "calendar", "response")
- applicability_criteria: list of conditions when this pack should apply (e.g. "sender is a known personal contact")
- negative_examples: list of situations where this pack should NOT apply (e.g. "sender is an automated mailing list")
- source_trace_ids: list of step IDs from the trace that motivated this pack (as strings, e.g. ["1", "2"])
- confidence: a float 0-1 reflecting how confident you are in this lesson
"""


# ---------------------------------------------------------------------------
# JSON file persistence for standalone / test usage
# ---------------------------------------------------------------------------

PACKS_FILE_DEFAULT = "experience_packs.json"


def save_packs_to_file(packs: List[ExperiencePack], path: str = PACKS_FILE_DEFAULT):
    """Write packs list to a JSON file on disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in packs], f, indent=2, default=str)


def load_packs_from_file(path: str = PACKS_FILE_DEFAULT) -> List[ExperiencePack]:
    """Load packs from a JSON file. Returns empty list if file doesn't exist."""
    import os
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [ExperiencePack.model_validate(p) for p in raw]


def seed_store_from_file(store, path: str = PACKS_FILE_DEFAULT):
    """Load packs from disk and write them individually into a LangGraph Store
    so the graph can retrieve them at runtime."""
    packs = load_packs_from_file(path)
    for pack in packs:
        store.put(
            EXPERIENCE_NAMESPACE,
            pack.pack_id,
            pack.model_dump(),
        )
    return packs


def sync_store_to_file(store, path: str = PACKS_FILE_DEFAULT):
    """Read packs from a LangGraph Store and write them to disk."""
    lib = ExperienceLibrary(store)
    packs = lib.list_packs(include_inactive=True)
    save_packs_to_file(packs, path)
    return packs


# ---------------------------------------------------------------------------
# Trace persistence (JSONL — one record per run)
# ---------------------------------------------------------------------------

TRACE_FILE_DEFAULT = "trace_history.jsonl"


def save_traces_to_file(
    traces: List[StepTrace],
    email_input: dict,
    run_id: str,
    path: str = TRACE_FILE_DEFAULT,
):
    """Append a run's traces to a JSONL file on disk."""
    record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "email_subject": email_input.get("subject", ""),
        "email_author": email_input.get("author", ""),
        "traces": [t.model_dump() for t in traces],
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_traces_from_file(path: str = TRACE_FILE_DEFAULT):
    """Load all trace records from a JSONL file."""
    import os
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
