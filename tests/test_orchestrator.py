"""Orchestrator wiring: per-model runtime options, failure/cancel semantics,
calibration, readiness checks."""

import anyio
import pytest

from config.settings import Settings
from llm.backends.base import BackendError
from llm.backends.mock_backend import MockBackend
from llm.registry import Registry, UnknownModelError
from orchestrator.context_budget import ContextBudget, ContextOverflowError
from orchestrator.orchestrator import Orchestrator
from session.store import SessionStore


def make(tmp_path, backend=None, **settings_kw):
    settings = Settings(llm_backend="mock", session_db_url=f"sqlite:///{tmp_path / 'o.db'}",
                        **{"context_level": 0, **settings_kw})
    backend = backend or MockBackend()
    store = SessionStore(settings.session_db_url)
    orch = Orchestrator(settings=settings, registry=Registry(), backend=backend, store=store,
                        budget=ContextBudget(safety_margin=settings.context_safety_margin,
                                             level=settings.context_level))
    return orch, backend, store


async def turn(orch, message="hello", model="Gemma 4 12B", session_id="s", **options):
    return [e async for e in orch.generate(session_id=session_id, message=message,
                                           model_name=model, options=options)]


async def test_num_ctx_and_think_value_from_registry(tmp_path):
    orch, backend, _ = make(tmp_path)
    await turn(orch, model="GPT-OSS 20B")
    call = backend.calls[-1]
    assert call["num_ctx"] == 65536    # never Ollama's silent 4096 default
    assert call["think"] == "low"      # GPT-OSS can't disable reasoning; "low" is minimal
    await turn(orch, model="Gemma 4 12B")
    assert backend.calls[-1]["think"] is False
    await turn(orch, model="Llama 3.1 8B")
    assert backend.calls[-1]["think"] is None   # no reasoning mode -> field omitted


async def test_thinking_mode_only_for_capable_models(tmp_path):
    orch, backend, _ = make(tmp_path)
    await turn(orch, model="Qwen3 8B", think=True)
    assert backend.calls[-1]["think"] is True
    await turn(orch, model="Llama 3.1 8B", think=True)
    assert backend.calls[-1]["think"] is None


async def test_history_alternates_and_is_replayed(tmp_path):
    orch, backend, store = make(tmp_path)
    await turn(orch, message="first")
    await turn(orch, message="second")
    assert [m["role"] for m in backend.calls[-1]["messages"]] == ["system", "user", "assistant", "user"]
    assert [t["role"] for t in store.get_turns("s")] == ["user", "assistant"] * 2


async def test_failed_turn_persists_nothing(tmp_path):
    class Broken(MockBackend):
        async def chat(self, messages, model, **kw):
            raise BackendError("boom")
            yield  # pragma: no cover

    orch, _, store = make(tmp_path, backend=Broken())
    with pytest.raises(BackendError):
        await turn(orch)
    assert store.get_turns("s") == [] and store.list_sessions() == []


async def test_unknown_model_and_oversized_message_fail_before_storage(tmp_path):
    orch, _, store = make(tmp_path)
    with pytest.raises(UnknownModelError):
        await turn(orch, model="Nope")
    with pytest.raises(ContextOverflowError):
        await turn(orch, message="x" * 400_000)
    assert store.list_sessions() == []


async def test_missing_model_fails_fast(tmp_path):
    class NotPulled(MockBackend):
        async def warm_state(self, model, num_ctx=None):
            return "missing"

    orch, backend, _ = make(tmp_path, backend=NotPulled())
    with pytest.raises(BackendError, match="ollama pull gemma4:12b"):
        await turn(orch)
    assert backend.calls == []


async def test_cold_model_announces_loading(tmp_path):
    class Cold(MockBackend):
        async def warm_state(self, model, num_ctx=None):
            return "reload"

    orch, _, _ = make(tmp_path, backend=Cold())
    evs = await turn(orch)
    assert evs[0]["status"] == "loading" and "32,768" in evs[0]["text"]


async def test_partial_reply_survives_cancellation(tmp_path):
    """Starlette cancels a disconnected response via an anyio cancel scope, which
    re-cancels every await. The (synchronous) write in chat's finally must land."""
    orch, _, store = make(tmp_path, backend=MockBackend(token_delay=0.01))
    seen = 0
    with anyio.CancelScope() as scope:
        async for ev in orch.generate(session_id="c", message="hello",
                                      model_name="Gemma 4 12B", options={}):
            if "delta" in ev:
                seen += 1
                if seen == 3:
                    scope.cancel()
    turns = store.get_turns("c")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[1]["content"].startswith("[mock:gemma4:12b]")


async def test_calibration_learns_per_model_from_prompt_eval_count(tmp_path):
    class Hungry(MockBackend):
        """Reports ~2x the heuristic's prompt tokens, like a token-hungry tokenizer."""
        async def chat(self, messages, model, **kw):
            async for ev in super().chat(messages, model, **kw):
                if ev.done:
                    ev.prompt_eval_count = sum(len(m["content"]) for m in messages) // 2
                yield ev

    orch, _, _ = make(tmp_path, backend=Hungry())
    for _ in range(5):
        await turn(orch, model="Gemma 4 12B")
    assert orch.budget.chars_per_token("gemma4:12b") < 3.0
    assert orch.budget.chars_per_token("gpt-oss:20b") == 4.0


async def test_level0_overflow_refuses_with_guidance(tmp_path):
    orch, _, store = make(tmp_path)
    store.append_exchange("big", "Gemma 4 12B", "x" * 80_000, "y" * 80_000)
    with pytest.raises(ContextOverflowError, match="CONTEXT_LEVEL=2"):
        await turn(orch, session_id="big", num_ctx=8192)
