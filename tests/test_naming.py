#!/usr/bin/env python3
"""id 规范化、厂商归一、配额记法、计费模型的测试。"""

from __future__ import annotations

import unittest

from model_data import billing, currency, idnorm, notation, vendors


class TestCanonicalId(unittest.TestCase):
    """全库统一口径：小写 + 版本用小数点。"""

    def test_lowercase(self):
        self.assertEqual(idnorm.canonical_id("DeepSeek-V4.1-Flash"), "deepseek-v4.1-flash")
        self.assertEqual(idnorm.canonical_id("GLM-5.3"), "glm-5.3")

    def test_dash_to_dot_version(self):
        self.assertEqual(idnorm.canonical_id("claude-sonnet-4-5"), "claude-sonnet-4.5")
        self.assertEqual(idnorm.canonical_id("gpt-3-5-turbo"), "gpt-3.5-turbo")
        self.assertEqual(idnorm.canonical_id("deepseek-v4-1-flash"), "deepseek-v4.1-flash")
        self.assertEqual(idnorm.canonical_id("GLM-5-3"), "glm-5.3")

    def test_versions_converge(self):
        """同一个模型的两种写法必须归一到同一结果。"""
        pairs = [
            ("deepseek-v4.1-flash", "deepseek-v4-1-flash"),
            ("claude-sonnet-4.5", "claude-sonnet-4-5"),
            ("glm-5.3", "GLM-5-3"),
        ]
        for a, b in pairs:
            self.assertEqual(idnorm.canonical_id(a), idnorm.canonical_id(b))

    def test_llama_version_converted(self):
        """llama-3-1-8b 是 Llama 3.1 8B，3-1 是版本号。"""
        self.assertEqual(idnorm.canonical_id("llama-3-1-8b"), "llama-3.1-8b")
        self.assertEqual(idnorm.canonical_id("llama-3-1-70b"), "llama-3.1-70b")

    def test_size_markers_untouched(self):
        """235b / a22b 是规模标记，不是版本号。"""
        self.assertEqual(
            idnorm.canonical_id("qwen3-vl-235b-a22b-instruct"),
            "qwen3-vl-235b-a22b-instruct")

    def test_vendor_prefix_stripped(self):
        """厂商前缀冗余时剥掉：amazon-nova → nova。"""
        self.assertEqual(idnorm.canonical_id("amazon-nova-2-5-sonic"),
                         "nova-2.5-sonic")
        self.assertEqual(idnorm.canonical_id("coding-xiaomi-mimo-v2.5"),
                         "mimo-v2.5")
        self.assertEqual(idnorm.canonical_id("zai-glm-5.3"), "glm-5.3")

    def test_distill_keeps_source_prefix(self):
        """蒸馏模型的厂商前缀不能剥：它是归属信息。"""
        self.assertEqual(
            idnorm.canonical_id("deepseek-r1-distill-qwen-32b"),
            "deepseek-r1-distill-qwen-32b")

    def test_seed_versions_converted(self):
        """doubao-seed-1.6 是豆包的版本号，要转小数点。"""
        self.assertEqual(idnorm.canonical_id("doubao-seed-1-6-vision-250815"),
                         "doubao-seed-1.6-vision-250815")
        self.assertEqual(idnorm.canonical_id("doubao-seed-2-0-pro"),
                         "doubao-seed-2.0-pro")

    def test_dates_untouched(self):
        self.assertEqual(idnorm.canonical_id("gpt-4o-mini-2024-07-18"),
                         "gpt-4o-mini-2024-07-18")

    def test_underscore_and_dup_dashes(self):
        self.assertEqual(idnorm.canonical_id("MiniMax_M3.1__Flash"), "minimax-m3.1-flash")

    def test_empty(self):
        self.assertEqual(idnorm.canonical_id(""), "")
        self.assertEqual(idnorm.canonical_id(None), "")

    def test_same_model(self):
        self.assertTrue(idnorm.is_same_model("DeepSeek-V4.1-Flash", "deepseek-v4.1-flash"))
        self.assertFalse(idnorm.is_same_model("gpt-5.2", "gpt-5.3"))


class TestVendors(unittest.TestCase):
    """厂商别名归一。"""

    def test_deepseek_variants(self):
        for v in ("deepseek", "deepseek-ai", "deepseekai", "DeepSeek-AI", "深度求索"):
            self.assertEqual(vendors.canonical_vendor(v), "deepseek", v)

    def test_zhipu_variants(self):
        for v in ("zhipuai", "zhipu", "z-ai", "zai-org", "bigmodel", "智谱AI"):
            self.assertEqual(vendors.canonical_vendor(v), "zhipuai", v)

    def test_others(self):
        self.assertEqual(vendors.canonical_vendor("kimi"), "moonshotai")
        self.assertEqual(vendors.canonical_vendor("qwen"), "alibaba")
        self.assertEqual(vendors.canonical_vendor("volcengine"), "bytedance")
        self.assertEqual(vendors.canonical_vendor("hunyuan"), "tencent")
        self.assertEqual(vendors.canonical_vendor("meta-llama"), "meta")

    def test_unknown_passthrough(self):
        """认不出就原样保留，不猜。"""
        self.assertEqual(vendors.canonical_vendor("some-new-lab"), "some-new-lab")

    def test_aliases_list(self):
        a = vendors.vendor_aliases("deepseek")
        self.assertIn("deepseek", a)
        self.assertIn("deepseek-ai", a)


class TestNotation(unittest.TestCase):
    def test_pow2_to_decimal(self):
        self.assertEqual(notation.round_k(1048576), 1024000)
        self.assertEqual(notation.round_k(393216), 384000)

    def test_never_rounds_up(self):
        for v in (1048576, 393216, 131072, 262144, 8192):
            self.assertLessEqual(notation.round_k(v), v)

    def test_clamp(self):
        ctx, out = notation.clamp_output(256000, 256000)
        self.assertLess(out, ctx)


class TestCurrency(unittest.TestCase):
    def test_cn_channels(self):
        for ch in ("alibaba-cn", "zhipuai", "siliconflow-cn", "volcengine", "volcengine"):
            self.assertTrue(currency.is_cny_channel(ch), ch)

    def test_intl_channels(self):
        for ch in ("alibaba", "moonshot", "siliconflow", "openai", "anthropic"):
            self.assertFalse(currency.is_cny_channel(ch), ch)


class TestBilling(unittest.TestCase):
    def test_cache_block(self):
        from model_data.trust.engine import Observation
        rows = [
            Observation(source="a", tier="first_party", field_name="price_cache_read",
                        value=0.3, valid=True, currency="USD"),
            Observation(source="a", tier="first_party", field_name="price_cache_write",
                        value=3.75, valid=True, currency="USD"),
        ]
        b = billing.build_cache_block(rows)
        self.assertEqual(b["read"], 0.3)
        self.assertEqual(b["write"], 3.75)

    def test_cache_absent_is_none(self):
        self.assertIsNone(billing.build_cache_block([]))

    def test_tiers_sorted_and_dedup(self):
        from model_data.trust.engine import Observation
        rows = [
            Observation(source="a", tier="aggregator", field_name="price_tier",
                        value={"input": 4, "output": 18,
                               "tier": {"type": "context", "size": 200000}},
                        valid=True, currency="USD"),
            Observation(source="b", tier="aggregator", field_name="price_tier",
                        value={"input": 4, "output": 18,
                               "tier": {"type": "context", "size": 200000}},
                        valid=True, currency="USD"),
            Observation(source="a", tier="aggregator", field_name="price_tier",
                        value={"input": 2, "output": 10,
                               "tier": {"type": "context", "size": 100000}},
                        valid=True, currency="USD"),
        ]
        tiers = billing.build_tiers(rows)
        self.assertEqual(len(tiers), 2)          # 同 over 去重
        self.assertEqual(tiers[0]["over"], 100000)   # 升序
        self.assertEqual(tiers[1]["over"], 200000)

    def test_multipliers_dedup(self):
        from model_data.trust.engine import Observation
        same = {"factor": 2, "when": {"utc_days": {"in": ["mon"]}}}
        rows = [
            Observation(source="a", tier="first_party", field_name="price_multiplier",
                        value=same, valid=True, currency="USD"),
            Observation(source="b", tier="aggregator", field_name="price_multiplier",
                        value=dict(same), valid=True, currency="USD"),
        ]
        mults = billing.build_multipliers(rows)
        self.assertEqual(len(mults), 1)


class TestIdentitySnapshot(unittest.TestCase):
    """日期折叠只处理日期，绝不动版本号。"""

    def test_date_folding(self):
        from model_data.identity import date_base
        self.assertEqual(date_base("claude-opus-4-5-20251101"), "claude-opus-4-5")
        self.assertEqual(date_base("gpt-4o-mini-2024-07-18"), "gpt-4o-mini")
        self.assertEqual(date_base("deepseek-v4.1-flash-1017"), "deepseek-v4.1-flash")

    def test_version_is_not_date(self):
        from model_data.identity import date_base
        for x in ("claude-sonnet-4-5", "gpt-5.2", "deepseek-v4-flash",
                  "claude-opus-4-5", "claude-haiku-4-5"):
            self.assertIsNone(date_base(x), x)

    def test_distinct_markers_not_folded(self):
        from model_data.identity import date_base
        self.assertIsNone(date_base("gemini-2.5-flash-preview-09-2025"))

    def test_alias_index_from_openrouter(self):
        from model_data.identity import build_alias_index
        data = {"data": [
            {"id": "openai/gpt-x", "canonical_slug": "openai/gpt-x-20260101"},
            {"id": "openai/gpt-x:batch", "canonical_slug": "openai/gpt-x-20260101"},
        ]}
        idx = build_alias_index(("openrouter", data))
        self.assertEqual(idx.resolve("gpt-x"), idx.resolve("gpt-x:batch"))
        self.assertEqual(len(idx.siblings("gpt-x")), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
