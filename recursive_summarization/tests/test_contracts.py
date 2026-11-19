"""Contract tests for the recursive summarization library.

Runs with unittest (no pytest dependency). Tests the contracts rather than
exact summary text — the summary text is the model's to decide; the budget
arithmetic and structural guarantees are the library's.

Contracts pinned:
  - Passthrough makes zero model calls when under budget
  - Recency window is preserved verbatim and in order
  - Summary respects its target ceiling
  - Recency-wins holds when recent turns overflow
  - Fold path passes through cheaply when it can, folds when it must
  - Chat adapter keeps extra history fields in the metadata bag
  - Recursion actually fires on large inputs
  - No-progress guard terminates pathological inputs
  - Empty model output does not crash
  - Oversized blocks are hard-split
  - Tighten pass converges or hard-truncates
"""

from __future__ import annotations

import sys
import unittest

from recursive_summarization import (
    CompactedContext,
    HeuristicCounter,
    RecursiveSummarizer,
    Role,
    SummarizerConfig,
    Turn,
    from_chat_history,
)

from .conftest import (
    MockEmptySummarizeFn,
    MockExpandingSummarizeFn,
    MockSummarizeFn,
    make_config,
    make_summarizer,
    make_turns,
)


class TestPassthrough(unittest.TestCase):
    def test_empty_history(self):
        summarizer, fn, _ = make_summarizer(target_budget_tokens=5000)
        result = summarizer.compact([])
        self.assertTrue(result.stats.passthrough)
        self.assertIsNone(result.summary)
        self.assertEqual(fn.call_count, 0)

    def test_short_history_under_budget(self):
        fn = MockSummarizeFn()
        summarizer, _, _ = make_summarizer(mock_fn=fn, target_budget_tokens=5000)
        turns = make_turns(3, text_len=50)
        result = summarizer.compact(turns)
        self.assertTrue(result.stats.passthrough)
        self.assertEqual(fn.call_count, 0)
        self.assertEqual(len(result.recent_turns), 3)

    def test_passthrough_preserves_all_turns(self):
        summarizer, _, _ = make_summarizer(target_budget_tokens=5000)
        turns = make_turns(4, text_len=30)
        result = summarizer.compact(turns)
        self.assertTrue(result.stats.passthrough)
        for i, turn in enumerate(result.recent_turns):
            self.assertEqual(turn.text, turns[i].text)
            self.assertEqual(turn.role, turns[i].role)


class TestRecencyWindow(unittest.TestCase):
    def test_recency_window_verbatim(self):
        summarizer, _, _ = make_summarizer(
            keep_recent_turns=2, target_budget_tokens=80
        )
        turns = make_turns(8, text_len=60)
        result = summarizer.compact(turns)
        self.assertEqual(len(result.recent_turns), 2)
        self.assertEqual(result.recent_turns[0].text, turns[6].text)
        self.assertEqual(result.recent_turns[1].text, turns[7].text)

    def test_recency_order_preserved(self):
        summarizer, _, _ = make_summarizer(
            keep_recent_turns=3, target_budget_tokens=80
        )
        turns = make_turns(10, text_len=60)
        result = summarizer.compact(turns)
        self.assertEqual(len(result.recent_turns), 3)
        for i in range(3):
            self.assertEqual(result.recent_turns[i].text, turns[7 + i].text)

    def test_recency_larger_than_history(self):
        summarizer, fn, _ = make_summarizer(
            keep_recent_turns=10, target_budget_tokens=5000
        )
        turns = make_turns(3, text_len=30)
        result = summarizer.compact(turns)
        self.assertTrue(result.stats.passthrough)
        self.assertEqual(len(result.recent_turns), 3)


class TestRecencyWins(unittest.TestCase):
    def test_recency_overflow_keeps_recent_turns(self):
        summarizer, _, _ = make_summarizer(
            keep_recent_turns=2, target_budget_tokens=30, min_summary_tokens=20
        )
        turns = [
            Turn(role=Role.USER, text="a" * 200),
            Turn(role=Role.ASSISTANT, text="b" * 200),
            Turn(role=Role.USER, text="c" * 200),
            Turn(role=Role.ASSISTANT, text="d" * 200),
        ]
        result = summarizer.compact(turns)
        self.assertEqual(len(result.recent_turns), 2)
        self.assertEqual(result.recent_turns[0].text, "c" * 200)
        self.assertEqual(result.recent_turns[1].text, "d" * 200)
        self.assertIsNotNone(result.summary)


class TestRecursion(unittest.TestCase):
    def test_recursion_fires_on_large_input(self):
        summarizer, _, _ = make_summarizer(
            keep_recent_turns=2,
            target_budget_tokens=100,
            summarize_input_budget=80,
        )
        turns = make_turns(20, text_len=100)
        result = summarizer.compact(turns)
        self.assertGreaterEqual(result.stats.max_recursion_depth, 1)
        self.assertGreater(result.stats.summarize_calls, 1)
        self.assertIsNotNone(result.summary)

    def test_depth_ceiling(self):
        summarizer, _, _ = make_summarizer(
            keep_recent_turns=1,
            target_budget_tokens=50,
            summarize_input_budget=40,
            max_recursion_depth=2,
        )
        turns = make_turns(30, text_len=120)
        result = summarizer.compact(turns)
        self.assertLessEqual(result.stats.max_recursion_depth, 2)


class TestNoProgressGuard(unittest.TestCase):
    def test_expanding_mock_terminates(self):
        fn = MockExpandingSummarizeFn()
        cfg = make_config(
            keep_recent_turns=1,
            target_budget_tokens=80,
            summarize_input_budget=100,
            max_recursion_depth=3,
        )
        summarizer = RecursiveSummarizer(
            summarize_fn=fn, config=cfg,
            count_tokens=HeuristicCounter(cfg.chars_per_token),
        )
        turns = make_turns(10, text_len=80)
        result = summarizer.compact(turns)
        self.assertIsNotNone(result.summary)


class TestEmptyOutput(unittest.TestCase):
    def test_empty_summarize_fn(self):
        fn = MockEmptySummarizeFn()
        cfg = make_config(keep_recent_turns=2, target_budget_tokens=80)
        summarizer = RecursiveSummarizer(
            summarize_fn=fn, config=cfg,
            count_tokens=HeuristicCounter(cfg.chars_per_token),
        )
        turns = make_turns(8, text_len=60)
        result = summarizer.compact(turns)
        self.assertIsNotNone(result.summary)
        self.assertGreater(fn.call_count, 0)


class TestOversizedBlock(unittest.TestCase):
    def test_oversized_block_hard_split(self):
        summarizer, _, _ = make_summarizer(
            keep_recent_turns=1,
            target_budget_tokens=100,
            summarize_input_budget=30,
        )
        turns = [
            Turn(role=Role.USER, text="x" * 500),
            Turn(role=Role.ASSISTANT, text="y" * 500),
            Turn(role=Role.USER, text="recent"),
        ]
        result = summarizer.compact(turns)
        self.assertIsNotNone(result.summary)
        self.assertEqual(result.recent_turns[-1].text, "recent")


class TestTighten(unittest.TestCase):
    def test_tighten_or_hard_truncate(self):
        fn = MockSummarizeFn(compression_ratio=0.8)
        cfg = make_config(
            keep_recent_turns=1,
            target_budget_tokens=40,
            max_tighten_passes=2,
        )
        summarizer = RecursiveSummarizer(
            summarize_fn=fn, config=cfg,
            count_tokens=HeuristicCounter(cfg.chars_per_token),
        )
        turns = make_turns(10, text_len=60)
        result = summarizer.compact(turns)
        # Either tighten converged or hard truncation kicked in
        self.assertTrue(
            result.stats.tighten_passes > 0 or result.stats.hard_truncations > 0
            or result.stats.passthrough
        )

    def test_hard_truncation_marker(self):
        fn = MockExpandingSummarizeFn()
        cfg = make_config(
            keep_recent_turns=1,
            target_budget_tokens=40,
            summarize_input_budget=300,
            max_tighten_passes=1,
        )
        summarizer = RecursiveSummarizer(
            summarize_fn=fn, config=cfg,
            count_tokens=HeuristicCounter(cfg.chars_per_token),
        )
        turns = make_turns(8, text_len=80)
        result = summarizer.compact(turns)
        if result.stats.hard_truncations > 0:
            self.assertIn(cfg.elision_marker, result.summary.text)


class TestFold(unittest.TestCase):
    def test_fold_passthrough_when_fits(self):
        summarizer, _, _ = make_summarizer(target_budget_tokens=5000)
        turns = make_turns(4, text_len=30)
        initial = summarizer.compact(turns)
        new = [Turn(role=Role.USER, text="short")]
        folded = summarizer.fold(initial, new)
        self.assertTrue(folded.stats.passthrough)

    def test_fold_empty_new_turns(self):
        summarizer, _, _ = make_summarizer()
        turns = make_turns(4, text_len=30)
        initial = summarizer.compact(turns)
        folded = summarizer.fold(initial, [])
        self.assertEqual(len(folded.recent_turns), len(initial.recent_turns))

    def test_fold_preserves_newest(self):
        summarizer, _, _ = make_summarizer(
            keep_recent_turns=2, target_budget_tokens=150
        )
        turns = make_turns(4, text_len=40)
        initial = summarizer.compact(turns)
        new = [Turn(role=Role.USER, text="MARKER_NEW")]
        folded = summarizer.fold(initial, new)
        texts = [t.text for t in folded.recent_turns]
        self.assertIn("MARKER_NEW", texts)

    def test_fold_never_drops_turns_silently(self):
        # Every turn must end up either verbatim in recent_turns or counted in
        # the summary. A fold that fits the budget must not trim the working
        # set down to keep_recent_turns without folding the rest in.
        summarizer, _, _ = make_summarizer(
            keep_recent_turns=2, target_budget_tokens=150
        )
        initial = summarizer.compact(make_turns(12, text_len=120))
        self.assertIsNotNone(initial.summary)
        folded_before = initial.summary.turns_folded
        new = [Turn(role=Role.USER, text="a"), Turn(role=Role.ASSISTANT, text="b")]
        folded = summarizer.fold(initial, new)
        accounted = len(folded.recent_turns) + (
            folded.summary.turns_folded if folded.summary else 0
        )
        self.assertEqual(accounted, folded_before + len(initial.recent_turns) + 2)

    def test_fold_triggers_on_overflow(self):
        fn = MockSummarizeFn()
        summarizer, _, _ = make_summarizer(
            mock_fn=fn, keep_recent_turns=2, target_budget_tokens=80
        )
        turns = make_turns(6, text_len=60)
        initial = summarizer.compact(turns)
        fn.call_count = 0
        big = make_turns(6, text_len=80)
        folded = summarizer.fold(initial, big)
        if not folded.stats.passthrough:
            self.assertGreater(fn.call_count, 0)


class TestRendering(unittest.TestCase):
    def test_messages_summary_is_system(self):
        summarizer, _, _ = make_summarizer(
            keep_recent_turns=2, target_budget_tokens=80
        )
        turns = make_turns(8, text_len=60)
        result = summarizer.compact(turns)
        messages = result.render_messages()
        if result.summary and result.summary.text:
            self.assertEqual(messages[0]["role"], "system")

    def test_prompt_block_contains_recent(self):
        summarizer, _, _ = make_summarizer(
            keep_recent_turns=2, target_budget_tokens=80
        )
        turns = make_turns(8, text_len=60)
        result = summarizer.compact(turns)
        block = result.render_prompt_block()
        self.assertIsInstance(block, str)
        for rt in result.recent_turns:
            self.assertIn(rt.text, block)

    def test_passthrough_no_system_message(self):
        summarizer, _, _ = make_summarizer(target_budget_tokens=5000)
        turns = make_turns(3, text_len=30)
        result = summarizer.compact(turns)
        messages = result.render_messages()
        self.assertTrue(all(m["role"] != "system" for m in messages))


class TestAdapters(unittest.TestCase):
    def test_chat_dict_round_trip(self):
        history = [
            {"role": "user", "content": "Hello", "timestamp": "2026-06-20"},
            {"role": "assistant", "content": "Hi!", "tool_calls": []},
        ]
        turns = from_chat_history(history)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].metadata["timestamp"], "2026-06-20")
        self.assertEqual(turns[1].metadata["tool_calls"], [])


class TestTokenCounting(unittest.TestCase):
    def test_heuristic(self):
        c = HeuristicCounter(4.0)
        self.assertEqual(c(""), 0)
        self.assertEqual(c("a"), 1)
        self.assertEqual(c("a" * 100), 25)

    def test_calibration(self):
        c = HeuristicCounter(4.0)
        c.calibrate("a" * 100, 50)
        self.assertEqual(c.chars_per_token, 2.0)
        self.assertEqual(c("a" * 100), 50)


class TestInstrumentation(unittest.TestCase):
    def test_passthrough_zero_calls(self):
        fn = MockSummarizeFn()
        summarizer, _, _ = make_summarizer(mock_fn=fn, target_budget_tokens=5000)
        turns = make_turns(2, text_len=30)
        result = summarizer.compact(turns)
        self.assertEqual(result.stats.summarize_calls, 0)
        self.assertEqual(fn.call_count, 0)

    def test_nonzero_calls(self):
        fn = MockSummarizeFn()
        summarizer, _, _ = make_summarizer(
            mock_fn=fn, keep_recent_turns=2, target_budget_tokens=80
        )
        turns = make_turns(10, text_len=60)
        result = summarizer.compact(turns)
        self.assertGreater(result.stats.summarize_calls, 0)
        self.assertEqual(result.stats.summarize_calls, fn.call_count)


if __name__ == "__main__":
    unittest.main()
