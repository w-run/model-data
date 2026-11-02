#!/usr/bin/env python3
"""可信度引擎与容量记法的单元测试。

每个用例都对应一次真实事故或一次误判，不是凑覆盖率。
"""

from __future__ import annotations

import math
import unittest
from datetime import date, timedelta

from model_data import identity, notation
from model_data.sources.collect import placeholder_reason
from model_data.trust.engine import Observation, aggregate, aggregate_pair, cluster_by_value

TODAY = date(2026, 11, 2)


def obs(source, tier, field, value, *, valid=True, prov="none", at=None):
    return Observation(source=source, tier=tier, field_name=field, value=value,
                       valid=valid, provenance=prov, observed_at=at)


class TestPlaceholder(unittest.TestCase):
    """OpenRouter 的 floor(0.9*ctx) 填充是最初的问题来源。"""

    def test_floor_09_detected(self):
        self.assertEqual(math.floor(1048576 * 0.9), 943718)
        self.assertEqual(placeholder_reason(1048576, 943718), "floor(0.9*context)")

    def test_output_ge_context(self):
        self.assertEqual(placeholder_reason(256000, 256000), "output>=context")

    def test_real_values_pass(self):
        self.assertIsNone(placeholder_reason(1048576, 393216))
        self.assertIsNone(placeholder_reason(1048576, 384000))
        self.assertIsNone(placeholder_reason(400000, 128000))


class TestClustering(unittest.TestCase):
    """384K 的两种写法要合并，512K 和 500K 必须分开。"""

    def test_384k_forms_merge(self):
        self.assertEqual(len(cluster_by_value([393216, 384000])), 1)

    def test_512k_vs_500k_split(self):
        self.assertEqual(len(cluster_by_value([524288, 500000])), 2)

    def test_1m_vs_1mi_split(self):
        self.assertEqual(len(cluster_by_value([1048576, 1000000])), 2)

    def test_outlier_kept_apart(self):
        clusters = cluster_by_value([1048576, 1048576, 1048000])
        self.assertEqual(len(clusters), 2)


class TestAuthoritative(unittest.TestCase):
    """第一方证据不该被聚合商的票数稀释。"""

    def test_first_party_wins_over_many_aggregators(self):
        rows = [obs("models.dev:deepseek", "first_party", "context", 1048576, prov="doc_url")]
        for i in range(20):
            rows.append(obs(f"models.dev:agg{i}", "aggregator", "context", 1000000))
        ev, _ = aggregate_pair(rows, today=TODAY)
        self.assertEqual(ev.value, 1048576)
        self.assertGreaterEqual(ev.confidence, 0.90)
        self.assertEqual(ev.level, "verified")

    def test_no_authority_falls_back_to_consensus(self):
        rows = [obs(f"a{i}", "aggregator", "context", 200000) for i in range(5)]
        ev, _ = aggregate_pair(rows, today=TODAY)
        self.assertEqual(ev.value, 200000)
        self.assertLess(ev.confidence, 0.90)


class TestPairing(unittest.TestCase):
    """ctx 和 out 必须同源，否则会拼出物理上不存在的组合。"""

    def test_no_cross_source_mixing(self):
        rows = [
            obs("models.dev:aimlapi", "aggregator", "context", 32768),
            obs("openrouter", "openrouter", "context", 1024000),
            obs("openrouter", "openrouter", "max_output", 819200),
        ]
        ctx, out = aggregate_pair(rows, today=TODAY)
        # 跨源混合的典型症状：ctx 来自 A 家、out 来自 B 家。
        # 正确行为是 out 要么与 ctx 同源，要么因为没有同源声明而留空。
        if ctx and out and out.value is not None:
            self.assertLess(out.value, ctx.value,
                            f"拼出了 {ctx.value}/{out.value} 这种不存在的组合")

    def test_output_comes_from_same_cluster(self):
        rows = [
            obs("src-a", "first_party", "context", 1048576),
            obs("src-a", "first_party", "max_output", 393216),
            obs("src-b", "aggregator", "context", 32768),
            obs("src-b", "aggregator", "max_output", 8000),
        ]
        ctx, out = aggregate_pair(rows, today=TODAY)
        self.assertEqual(ctx.value, 1048576)
        self.assertEqual(out.value, 393216)


class TestNotation(unittest.TestCase):
    """十进制整 K 记法。"""

    def test_power_of_two_to_decimal(self):
        self.assertEqual(notation.round_k(1048576), 1024000)
        self.assertEqual(notation.round_k(393216), 384000)
        self.assertEqual(notation.round_k(131072), 128000)
        self.assertEqual(notation.round_k(8192), 8000)

    def test_already_decimal_untouched(self):
        self.assertEqual(notation.round_k(1000000), 1000000)
        self.assertEqual(notation.round_k(384000), 384000)

    def test_non_round_k_untouched(self):
        """非整 K 不强套，强行取整反而丢信息。"""
        self.assertEqual(notation.round_k(943718), 943718)

    def test_never_rounds_up(self):
        for v in (1048576, 393216, 131072, 8192, 262144):
            self.assertLessEqual(notation.round_k(v), v)

    def test_clamp_keeps_headroom(self):
        """取整后 ctx 与 out 相等时必须压低下输出。"""
        ctx, out = notation.clamp_output(256000, 256000)
        self.assertLess(out, ctx)
        ctx, out = notation.clamp_output(1024000, 1024000)
        self.assertLess(out, ctx)

    def test_clamp_noop_when_ok(self):
        self.assertEqual(notation.clamp_output(1024000, 384000), (1024000, 384000))


class TestIdentity(unittest.TestCase):
    """身份对齐：写法归一 + 日期折叠，但不合并版本。"""

    def test_norm_key_unifies_spelling(self):
        self.assertEqual(identity.norm_key("claude-sonnet-4.5"),
                         identity.norm_key("claude-sonnet-4-5"))

    def test_version_not_merged(self):
        """v4 与 v4.1 是不同模型，不能折叠到同一族。"""
        self.assertNotEqual(identity.norm_key("deepseek-v4-flash"),
                            identity.norm_key("deepseek-v4.1-flash"))

    def test_snapshot_folds_to_base(self):
        self.assertEqual(identity.date_base("deepseek-v4.1-flash-1017"),
                         "deepseek-v4.1-flash")

    def test_variant_suffix_stripped(self):
        self.assertEqual(identity.strip_variant("gpt-5.2:batch"), "gpt-5.2")


class TestFreshness(unittest.TestCase):
    def test_old_observation_depreciated(self):
        old = obs("x", "first_party", "context", 100000,
                  at=(TODAY - timedelta(days=400)).isoformat())
        fresh = obs("y", "first_party", "context", 100000, at=TODAY.isoformat())
        self.assertLess(old.weight(today=TODAY), fresh.weight(today=TODAY))

    def test_recent_full_weight(self):
        o = obs("x", "first_party", "context", 100000,
                at=(TODAY - timedelta(days=3)).isoformat(), prov="doc_url")
        self.assertGreaterEqual(o.weight(today=TODAY), 0.95)


class TestEdgeCases(unittest.TestCase):
    def test_zero_and_negative_filtered(self):
        rows = [obs("x", "aggregator", "max_output", 0),
                obs("y", "aggregator", "max_output", -5)]
        self.assertIsNone(aggregate(rows, field_name="max_output"))

    def test_all_invalid_returns_none(self):
        rows = [obs("x", "openrouter", "max_output", 943718, valid=False)]
        self.assertIsNone(aggregate(rows, field_name="max_output"))

    def test_empty(self):
        self.assertEqual(aggregate_pair([]), (None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
