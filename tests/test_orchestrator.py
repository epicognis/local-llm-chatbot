"""Orchestrator wiring: per-model runtime options, calibration, failure semantics."""

import pytest

from config.settings import settings
from llm.backends.base import BackendError
from llm.backends.mock_backend import MockBackend
from orchestrator.context_budget import ContextBudgetAllocator, TokenCalibrator
from orchestrator.orchestrator import Orchestrator
from session.store import SessionStore


@pytest.fixture
async def store(tmp_path):
    s = SessionStore(f"sqlite:///{tmp_path / 'orch.db'}")
    await s.init()
    return s


async def run_turn(orch, session_id="s", message="hello", model="Gemma 4 12B"):
    return [e async for e in orch.chat_stream(session_id, message, model)]


async def test_num_ctx_sent_from_registry_default_ctx(store):
    backend = MockBackend(token_delay=0)
    await run_turn(Orchestrator(backend, store), model="GPT-OSS 20B")
    opts = backend.calls[0]["opts"]
    assert opts["num_ctx"] == 131072  # never Ollama's silent 4096 default
    assert opts["think"] == "low"     # GPT-OSS can't disable reasoning; "low" is minimal
    assert opts["num_predict"] == settings.DEFAULT_MAX_TOKENS


async def test_think_flag_only_sent_to_thinking_capable_models(store):
    backend = MockBackend(token_delay=0)
    await run_turn(Orchestrator(backend, store), model="Llama 3.1 8B")
    assert "think" not in backend.calls[0]["opts"]


async def test_history_alternates_and_is_replayed_next_turn(store):
    backend = MockBackend(token_delay=0)
    orch = Orchestrator(backend, store)
    await run_turn(orch, message="first")
    await run_turn(orch, message="second")
    roles = [m["role"] for m in backend.calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert [t.role for t in await store.get_turns("s")] == ["user", "assistant"] * 2


async def test_failed_turn_persists_nothing(store):
    class Broken(MockBackend):
        def _chat_stream(self, messages, model, **opts):
            async def gen():
                raise BackendError("boom")
                yield  # pragma: no cover
            return gen()

    orch = Orchestrator(Broken(), store)
    with pytest.raises(BackendError):
        await run_turn(orch)
    assert await store.get_turns("s") == []
    assert await store.list_sessions() == []


async def test_level2_summarize_gets_runtime_opts_and_status_event(store, monkeypatch):
    monkeypatch.setattr(settings, "CONTEXT_LEVEL", 2)
    monkeypatch.setattr(settings, "SLIDING_WINDOW_TURNS", 2)
    monkeypatch.setattr(settings, "SUMMARY_TRIGGER_TOKENS", 0)
    backend = MockBackend(token_delay=0)
    orch = Orchestrator(backend, store)
    await run_turn(orch, message="one")
    await run_turn(orch, message="two")
    events = await run_turn(orch, message="three")
    assert {"type": "status", "text": "Summarizing earlier conversation…"} in events
    assert len(backend.summarize_calls) == 1
    sent = backend.calls[-1]["messages"]
    assert sent[1]["content"].startswith("[Earlier conversation summary]")


async def test_calibration_learns_from_prompt_eval_count(store):
    class Undercounted(MockBackend):
        """Reports 2x the heuristic's prompt tokens, like a tokenizer-hungry model."""
        def _chat_stream(self, messages, model, **opts):
            stats = opts.get("stats")
            gen = super()._chat_stream(messages, model, **opts)

            async def wrap():
                async for t in gen:
                    yield t
                from orchestrator.context_budget import estimate_messages
                stats["prompt_eval_count"] = 2 * estimate_messages(messages)
            return wrap()

    orch = Orchestrator(Undercounted(token_delay=0), store)
    await run_turn(orch)
    assert orch.calibrator.ratio("gemma4:12b") == pytest.approx(2.0)


def test_calibrator_never_loosens_below_heuristic():
    cal = TokenCalibrator()
    assert cal.observe("m", estimated=100, actual=50) == 1.0
    assert cal.observe("m", estimated=100, actual=200) == pytest.approx(1.0 + 0.3 * 1.0)
    assert cal.observe("m", estimated=100, actual=None) == pytest.approx(1.3)


def test_calibrated_ratio_tightens_the_hard_guard():
    turns = [{"role": "user", "content": "x" * 400}] * 10  # 100 est. tokens each
    loose = ContextBudgetAllocator(model_ctx=2000, max_tokens=100, safety_margin=100)
    tight = ContextBudgetAllocator(model_ctx=2000, max_tokens=100, safety_margin=100, ratio=2.0)
    n_loose = len(loose.assemble("sys", turns, "q"))
    n_tight = len(tight.assemble("sys", turns, "q"))
    assert n_tight < n_loose


def test_oversized_message_is_refused():
    alloc = ContextBudgetAllocator(model_ctx=1000, max_tokens=100, safety_margin=100)
    with pytest.raises(ValueError, match="too long"):
        alloc.assemble("sys", [], "x" * 10_000)


async def test_partial_reply_survives_level_triggered_cancellation(store):
    """Starlette cancels a disconnected response via an anyio cancel scope, which
    re-cancels every await — including the persistence in chat_stream's finally.
    The shielded write must still land."""
    import anyio

    orch = Orchestrator(MockBackend(token_delay=0.01), store)
    seen = 0
    with anyio.CancelScope() as scope:
        async for ev in orch.chat_stream("s", "hello", "Gemma 4 12B"):
            if ev["type"] == "token":
                seen += 1
                if seen == 3:
                    scope.cancel()
    turns = await store.get_turns("s")
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[1].content.startswith("[MOCK] You said:")
