#!/usr/bin/env python3
"""
核心逻辑的自测 —— 不依赖网络，CI 里跑得很快。

覆盖的是“曾经踩过的坑”，不是形式上凑覆盖率：
每个 test 对应一次真实事故或一次误报。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.limits import derive_recommended, is_placeholder, vote_limits  # noqa: E402


class TestPlaceholderDetect(unittest.TestCase):
    def test_floor_09_is_placeholder(self):
        """deepseek-v4.1-flash 事故：943718 = floor(1048576*0.9)。"""
        ctx = 1048576
        self.assertEqual(math.floor(ctx * 0.9), 943718)
        self.assertTrue(is_placeholder(ctx, 943718))

    def test_output_ge_context_is_placeholder(self):
        """输出 >= 上下文，共享窗口下不可能。"""
        self.assertTrue(is_placeholder(1048576, 1048576))
        self.assertTrue(is_placeholder(256000, 300000))

    def test_real_values_pass(self):
        """真实值不能被误判。"""
        self.assertFalse(is_placeholder(1048576, 384000))
        self.assertFalse(is_placeholder(1048576, 393216))
        self.assertFalse(is_placeholder(1000000, 128000))
        self.assertFalse(is_placeholder(400000, 128000))

    def test_none_safe(self):
        self.assertFalse(is_placeholder(None, 128000))
        self.assertFalse(is_placeholder(128000, None))


class TestVoteLimits(unittest.TestCase):
    def _obs(self, *rows):
        return [
            {"provider": p, "first_party": fp, "context": c, "output": o}
            for p, fp, c, o in rows
        ]

    def test_outlier_does_not_win(self):
        """1 个离群低值不能把一致的共识带偏（曾经用 min 就踩了这个）。"""
        obs = self._obs(
            ("a", False, 1048576, 384000),
            ("b", False, 1048576, 384000),
            ("c", False, 1048576, 384000),
            ("d", False, 1048000, 384000),   # 离群
        )
        r = vote_limits(obs)
        self.assertEqual(r["context"], 1048576)
        self.assertEqual(r["max_output"], 384000)

    def test_524288_not_merged_with_500000(self):
        """容差过宽会把 524288 和 500000 并簇，产出 out>=ctx 的荒谬结果。"""
        obs = self._obs(
            ("a", False, 524288, 500000),
            ("b", False, 524288, 500000),
            ("c", False, 500000, 128000),
        )
        r = vote_limits(obs)
        # 524288 票数多 → 选中它；其下 500000 不是占位（500000/524288 = 95%）
        self.assertEqual(r["context"], 524288)
        self.assertLess(r["max_output"], r["context"])

    def test_placeholders_dropped_when_all_are_placeholders(self):
        """全部是占位值时，宁可留 None 也不给假上限。"""
        obs = self._obs(
            ("a", False, 256000, 230400),
            ("b", False, 256000, 230400),
        )
        r = vote_limits(obs)
        self.assertEqual(r["context"], 256000)
        self.assertIsNone(r["max_output"])

    def test_first_party_preferred(self):
        """第一方口径优先于聚合商。"""
        obs = self._obs(
            ("aggregator1", False, 1000000, 384000),
            ("aggregator2", False, 1000000, 384000),
            ("deepseek", True, 1000000, 393216),
        )
        r = vote_limits(obs)
        self.assertEqual(r["max_output"], 393216)
        self.assertEqual(r["confidence"], "first_party")

    def test_first_party_loses_to_big_consensus(self):
        """第一方孤票但聚合端有压倒性共识时，以共识为准并降级 confidence。"""
        obs = [{"provider": f"agg{i}", "first_party": False,
                "context": 1048576, "output": 384000} for i in range(20)]
        obs.append({"provider": "auth", "first_party": True, "context": 200000, "output": 64000})
        r = vote_limits(obs)
        self.assertEqual(r["context"], 1048576)
        self.assertEqual(r["confidence"], "consensus")

    def test_empty_observations(self):
        r = vote_limits([])
        self.assertIsNone(r["context"])
        self.assertEqual(r["confidence"], "inferred")


class TestDeriveRecommended(unittest.TestCase):
    def test_never_starves_input(self):
        """无论上限多大，输入预算都不低于 50%。"""
        for ctx in (8192, 32768, 200000, 1048576, 2000000):
            for out in (None, 4096, 128000, ctx, ctx * 2):
                rec = derive_recommended({"context": ctx, "max_output": out})
                ratio = rec["input_budget"] / ctx
                self.assertGreaterEqual(ratio, 0.5, f"ctx={ctx} out={out} ratio={ratio:.2%}")

    def test_budget_fits_window(self):
        """input_budget + output_budget 不得超过 context。"""
        for ctx in (8192, 131072, 1048576):
            rec = derive_recommended({"context": ctx, "max_output": ctx // 2})
            self.assertLessEqual(rec["input_budget"] + rec["output_budget"], ctx)

    def test_output_capped_even_if_model_allows_more(self):
        """真实上限 393216 时，推荐输出仍按 128K 封顶。"""
        rec = derive_recommended({"context": 1048576, "max_output": 393216})
        self.assertEqual(rec["output_budget"], 131072)
        self.assertEqual(rec["input_budget"], 1048576 - 131072)

    def test_output_respects_small_model_limit(self):
        """小模型上限低于封顶时听模型的。"""
        rec = derive_recommended({"context": 32768, "max_output": 4096})
        self.assertEqual(rec["output_budget"], 4096)

    def test_input_only_semantics(self):
        rec = derive_recommended({"context": 200000, "max_output": 64000}, semantics="input_only")
        self.assertEqual(rec["input_budget"], 200000)
        self.assertEqual(rec["output_budget"], 64000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
