"""Chatbot-side Level 2 wiring (recursive summarization).

Library internals are covered by recursive_summarization/tests; these check the
host integration end-to-end through /chat on the mock backend: compaction
engages, fold runs in steady state, recompaction resets drift, no turn is ever
lost, the summarize call reuses the generation's runtime options, and the final
prompt always passes the hard window guard.
"""

from fastapi.testclient import TestClient

from api.main import create_app
from config.settings import Settings
from llm.backends.mock_backend import MockBackend, mock_summarize
from orchestrator.summarization import deserialize_state, serialize_state
from recursive_summarization import CompactedContext, CompactionStats, Role, RunningSummary, Turn


def run_level2(tmp_path, n_turns, backend=None, **cfg):
    settings = Settings(
        llm_backend="mock",
        context_level=2,
        session_db_url=f"sqlite:///{tmp_path / 'l2.db'}",
        **{"summary_target_budget_tokens": 120, "summary_keep_recent_turns": 2,
           "summary_input_budget_tokens": 200, **cfg},
    )
    app = create_app(settings, backend=backend)
    sid = "lvl2"
    statuses = []
    with TestClient(app) as client:
        for i in range(n_turns):
            body = {"session_id": sid, "message": f"message {i} " + "word " * 30,
                    "model": "GPT-OSS 20B"}
            with client.stream("POST", "/chat", json=body) as res:
                text = "".join(res.iter_text())
            statuses.append("summarizing" in text)
        raw_state = app.state.store.get_compaction_state(sid)
        data = client.get(f"/sessions/{sid}").json()
    return raw_state, data, statuses, app


def test_mock_summarize_extracts_and_compresses():
    prompt = ("Summarize.\n\n--- CONVERSATION ---\nuser: hello there this is a reasonably "
              "long conversation about budgets\n--- END ---\n\nSummary:")
    out = mock_summarize(prompt)
    assert "hello" in out and "Summarize" not in out


def test_short_session_is_zero_call_passthrough(tmp_path):
    backend = MockBackend()
    raw, data, statuses, _ = run_level2(tmp_path, 2, backend=backend,
                                        summary_target_budget_tokens=0)
    assert backend.complete_calls == []
    assert not any(statuses)
    assert data["summary"] is None


def test_level2_engages_and_keeps_full_transcript(tmp_path):
    raw, data, statuses, _ = run_level2(tmp_path, 8)
    assert data["summary"], "expected a running summary after a long session"
    assert data["turn_count"] == 16  # compaction is prompt-only; history is complete
    assert any(statuses), "UI should be told when summarization runs"


def test_fold_engages_in_steady_state_without_losing_turns(tmp_path):
    raw, data, _, _ = run_level2(tmp_path, 10, summary_recompact_cadence=0)
    compacted, covered, fold_count = deserialize_state(raw)
    assert compacted.summary and compacted.summary.text
    assert fold_count > 0, "expected the incremental fold path to run"
    # The current message isn't folded yet; everything before it is accounted for,
    # either verbatim in the recency window or counted in the summary.
    assert covered == data["turn_count"] - 2
    assert compacted.summary.turns_folded + len(compacted.recent_turns) == covered


def test_recompaction_cadence_caps_fold_count(tmp_path):
    raw, _, _, _ = run_level2(tmp_path, 12, summary_recompact_cadence=2)
    compacted, _, fold_count = deserialize_state(raw)
    assert compacted.summary and fold_count <= 2


def test_summarize_call_reuses_generation_runtime_options(tmp_path):
    seen = []

    class Recording(MockBackend):
        async def complete(self, prompt, model, **opts):
            seen.append((model, opts))
            return await super().complete(prompt, model, **opts)

    run_level2(tmp_path, 6, backend=Recording())
    assert seen, "expected summarize calls"
    model, opts = seen[0]
    assert model == "gpt-oss:20b"
    assert opts["num_ctx"] == 65536   # same runner as generation -> no reload
    assert opts["think"] == "low"     # not false: that costs GPT-OSS ~2k reasoning tokens


def test_level2_prompt_always_fits_the_window(tmp_path):
    # A tiny window: recency alone overflows it, so the final hard guard must trim.
    settings = Settings(llm_backend="mock", context_level=2, context_safety_margin=64,
                        summary_keep_recent_turns=4, summary_target_budget_tokens=0,
                        session_db_url=f"sqlite:///{tmp_path / 'g.db'}")
    backend = MockBackend()
    app = create_app(settings, backend=backend)
    with TestClient(app) as client:
        for i in range(6):
            with client.stream("POST", "/chat", json={
                "session_id": "g", "message": f"m{i} " + "w" * 900, "model": "Gemma 4 12B",
                "options": {"num_ctx": 2048, "max_tokens": 256},
            }) as res:
                text = "".join(res.iter_text())
                assert '"error"' not in text
    budget = app.state.orchestrator.budget
    for call in backend.calls:
        assert budget.total_tokens(call["messages"], "gemma4:12b") <= 2048 - 256 - 64


def test_compaction_state_round_trip():
    c = CompactedContext(
        summary=RunningSummary(text="notes", turns_folded=6, token_count=12, recursion_depth=1),
        recent_turns=[Turn(role=Role.USER, text="hi", metadata={"k": "v"}),
                      Turn(role=Role.ASSISTANT, text="yo")],
        stats=CompactionStats(),
    )
    back, covered, fold = deserialize_state(serialize_state(c, covered_count=8, fold_count=3))
    assert (covered, fold) == (8, 3)
    assert back.summary.text == "notes" and back.summary.turns_folded == 6
    assert [t.text for t in back.recent_turns] == ["hi", "yo"]
    assert back.recent_turns[0].metadata == {"k": "v"}
