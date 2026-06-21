"""Deterministic tests for the progressive context levels (build spec §7/§13).
Runs against MockBackend and a temp SQLite store — no GPU, no network."""

import pytest
from session.store import SessionStore, Turn
from llm.backends.mock_backend import MockBackend
from orchestrator.memory import (
    RawPassthroughProvider,
    SlidingWindowProvider,
    RollingSummaryProvider,
)
from orchestrator.context_budget import ContextBudgetAllocator
from orchestrator.orchestrator import Orchestrator
from config.settings import settings


def make_turns(n: int) -> list[Turn]:
    return [
        Turn(role="user" if i % 2 == 0 else "assistant",
             content=f"Turn number {i} content padding text", tokens=0, ts="")
        for i in range(n)
    ]


# ── Level 0 — raw passthrough ──────────────────────────────────────────

async def test_level0_sends_full_history():
    turns = make_turns(20)
    context_turns, summary = await RawPassthroughProvider().prepare("s1", {}, turns)
    assert len(context_turns) == 20
    assert summary == ""


async def test_level0_allocator_still_enforces_hard_guard():
    # Even at Level 0, a tiny budget must drop oldest turns rather than overflow.
    turns = make_turns(50)
    context_turns, summary = await RawPassthroughProvider().prepare("s1", {}, turns)
    allocator = ContextBudgetAllocator(model_ctx=200, max_tokens=50, safety_margin=20)
    messages = allocator.assemble(
        system_prompt="sys", turns=context_turns, current_message="hi", summary=summary
    )
    assert len(messages) < 52  # system + current + fewer than all 50 turns
    # oldest turns are the ones dropped — the last *history* turn (before the
    # current message, which is always appended last) should be near the end.
    history_contents = [m["content"] for m in messages[1:-1]]
    assert "Turn number 49" in history_contents[-1]


# ── Level 1 — sliding window ───────────────────────────────────────────

async def test_level1_truncates_to_window():
    turns = make_turns(20)
    provider = SlidingWindowProvider(window_turns=5)
    context_turns, summary = await provider.prepare("s1", {}, turns)
    assert len(context_turns) == 5
    assert summary == ""
    assert context_turns[0]["content"] == "Turn number 15 content padding text"
    assert context_turns[-1]["content"] == "Turn number 19 content padding text"


async def test_level1_no_truncation_when_under_window():
    turns = make_turns(3)
    provider = SlidingWindowProvider(window_turns=5)
    context_turns, summary = await provider.prepare("s1", {}, turns)
    assert len(context_turns) == 3
    assert summary == ""


# ── Level 2 — rolling summary ──────────────────────────────────────────

@pytest.fixture
async def store(tmp_path):
    s = SessionStore(f"sqlite:///{tmp_path / 'test.db'}")
    await s.init()
    return s


async def test_level2_no_summarization_below_trigger(store):
    session_id = "low-trigger"
    await store.get_or_create(session_id)
    turns = make_turns(10)
    provider = RollingSummaryProvider(
        backend=MockBackend(), store=store, summarizer_model="mock",
        window_turns=2, trigger_tokens=1_000_000,  # effectively unreachable
    )
    session = await store.get_or_create(session_id)
    context_turns, summary = await provider.prepare(session_id, session, turns)
    assert summary == ""
    assert len(context_turns) == 2  # recent window still applied
    persisted = await store.get_or_create(session_id)
    assert persisted["summarized_count"] == 0


async def test_level2_triggers_summarize_at_threshold(store):
    session_id = "trigger-now"
    await store.get_or_create(session_id)
    turns = make_turns(5)  # window=2 -> older=[0,1,2], recent=[3,4]
    provider = RollingSummaryProvider(
        backend=MockBackend(), store=store, summarizer_model="mock",
        window_turns=2, trigger_tokens=0,  # summarize ASAP
    )
    session = await store.get_or_create(session_id)
    context_turns, summary = await provider.prepare(session_id, session, turns)

    assert len(context_turns) == 2
    assert context_turns[0]["content"] == "Turn number 3 content padding text"
    assert summary.startswith("[MOCK SUMMARY]")

    persisted = await store.get_or_create(session_id)
    assert persisted["summarized_count"] == 3
    assert persisted["summary"] == summary


async def test_level2_does_not_resummarize_already_folded_turns(store):
    session_id = "idempotent"
    await store.get_or_create(session_id)
    turns = make_turns(5)
    provider = RollingSummaryProvider(
        backend=MockBackend(), store=store, summarizer_model="mock",
        window_turns=2, trigger_tokens=0,
    )

    session = await store.get_or_create(session_id)
    _, first_summary = await provider.prepare(session_id, session, turns)

    # Same turns, same session state — nothing new to fold in.
    session = await store.get_or_create(session_id)
    _, second_summary = await provider.prepare(session_id, session, turns)

    assert second_summary == first_summary
    persisted = await store.get_or_create(session_id)
    assert persisted["summarized_count"] == 3


async def test_level2_folds_new_turns_into_existing_summary(store):
    session_id = "cumulative"
    await store.get_or_create(session_id)
    provider = RollingSummaryProvider(
        backend=MockBackend(), store=store, summarizer_model="mock",
        window_turns=2, trigger_tokens=0,
    )

    turns = make_turns(5)
    session = await store.get_or_create(session_id)
    await provider.prepare(session_id, session, turns)

    # Conversation continues — two more turns arrive.
    turns = make_turns(7)
    session = await store.get_or_create(session_id)
    context_turns, summary = await provider.prepare(session_id, session, turns)

    assert len(context_turns) == 2
    assert context_turns[-1]["content"] == "Turn number 6 content padding text"
    persisted = await store.get_or_create(session_id)
    assert persisted["summarized_count"] == 5  # turns[:-2] of 7 == 5
    # The new summarize() call should have folded in the prior cached summary.
    assert "[Previous summary]" in summary or summary.startswith("[MOCK SUMMARY]")


# ── Orchestrator dispatch ──────────────────────────────────────────────

def test_orchestrator_dispatches_provider_by_level(monkeypatch):
    orch = Orchestrator(backend=MockBackend(), store=None)

    monkeypatch.setattr(settings, "CONTEXT_LEVEL", 0)
    assert isinstance(orch._get_provider("mock-model"), RawPassthroughProvider)

    monkeypatch.setattr(settings, "CONTEXT_LEVEL", 1)
    assert isinstance(orch._get_provider("mock-model"), SlidingWindowProvider)

    monkeypatch.setattr(settings, "CONTEXT_LEVEL", 2)
    assert isinstance(orch._get_provider("mock-model"), RollingSummaryProvider)

    monkeypatch.setattr(settings, "CONTEXT_LEVEL", 3)
    with pytest.raises(ValueError):
        orch._get_provider("mock-model")
