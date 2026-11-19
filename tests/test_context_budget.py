"""Allocator: hard window guard at every level, Level-0 passthrough, Level-1
sliding window, Level-2 summary reservation, per-model calibration."""

import pytest

from orchestrator.context_budget import ContextBudget, ContextOverflowError


def _history(n, content="word " * 20):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": content} for i in range(n)]


def test_level0_passthrough_keeps_all_history():
    b = ContextBudget(safety_margin=64, level=0)
    a = b.assemble(system_prompt="sys", history=_history(4), current_message="hello",
                   num_ctx=4096, max_tokens=256)
    assert len(a.messages) == 6  # system + 4 history + current
    assert a.dropped_turns == 0


def test_hard_guard_refuses_overflow_at_level0():
    b = ContextBudget(safety_margin=64, level=0)
    with pytest.raises(ContextOverflowError, match="no longer fits"):
        b.assemble(system_prompt="sys", history=_history(20, content="x" * 10_000),
                   current_message="hi", num_ctx=2048, max_tokens=256)


def test_anchors_too_big_refused():
    b = ContextBudget(safety_margin=64, level=0)
    with pytest.raises(ContextOverflowError, match="too long"):
        b.assemble(system_prompt="x" * 100_000, history=[], current_message="hi",
                   num_ctx=1024, max_tokens=256)


def test_non_positive_budget_refused():
    b = ContextBudget(safety_margin=768)
    with pytest.raises(ContextOverflowError):
        b.assemble(system_prompt="s", history=[], current_message="hi", num_ctx=1024, max_tokens=512)


def test_level1_sliding_drops_oldest_and_never_exceeds_budget():
    b = ContextBudget(safety_margin=64, level=1)
    history = [{"role": "user", "content": f"turn {i} " + "y" * 400} for i in range(40)]
    a = b.assemble(system_prompt="sys", history=history, current_message="hi",
                   num_ctx=2048, max_tokens=256)
    assert a.dropped_turns > 0
    assert a.prompt_tokens_est <= a.budget
    assert a.messages[0]["role"] == "system"
    assert a.messages[-1]["content"] == "hi"
    assert a.messages[-2]["content"].startswith("turn 39 ")  # most recent kept


def test_summary_is_reserved_before_recent_turns():
    b = ContextBudget(safety_margin=64)
    a = b.assemble(system_prompt="sys", history=_history(40, content="z" * 400),
                   current_message="hi", num_ctx=2048, max_tokens=256,
                   summary="[summary] facts", sliding=True)
    assert a.messages[1] == {"role": "system", "content": "[summary] facts"}
    assert a.prompt_tokens_est <= a.budget


def test_oversized_summary_is_dropped_not_overflowed():
    b = ContextBudget(safety_margin=64)
    a = b.assemble(system_prompt="sys", history=[], current_message="hi",
                   num_ctx=2048, max_tokens=256, summary="s" * 50_000, sliding=True)
    assert a.summary_dropped is True
    assert a.prompt_tokens_est <= a.budget


def test_calibration_is_per_model_and_only_tightens():
    b = ContextBudget(chars_per_token=4.0)
    # A model whose tokenizer uses 2 chars/token: estimates must grow.
    b.calibrate("dense", prompt_chars=800, prompt_eval_count=400)
    assert b.chars_per_token("dense") < 4.0
    assert b.estimate_tokens("x" * 400, "dense") > b.estimate_tokens("x" * 400)
    # A reading that suggests 8 chars/token must NOT loosen the guard.
    b.calibrate("loose", prompt_chars=800, prompt_eval_count=100)
    assert b.chars_per_token("loose") == 4.0
    # Other models are unaffected.
    assert b.chars_per_token("other") == 4.0


def test_calibrated_ratio_tightens_sliding_window():
    history = [{"role": "user", "content": "x" * 400}] * 20
    loose = ContextBudget(safety_margin=100, level=1)
    tight = ContextBudget(safety_margin=100, level=1)
    for _ in range(10):
        tight.calibrate("m", prompt_chars=400, prompt_eval_count=200)
    n_loose = len(loose.assemble(system_prompt="s", history=history, current_message="q",
                                 num_ctx=2000, max_tokens=100, model="m").messages)
    n_tight = len(tight.assemble(system_prompt="s", history=history, current_message="q",
                                 num_ctx=2000, max_tokens=100, model="m").messages)
    assert n_tight < n_loose
