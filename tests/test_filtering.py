#!/usr/bin/env python3
"""收录过滤与厂商归属推断的测试。"""

from __future__ import annotations

import unittest

from model_data import attribution, filtering

KNOWN = {"alibaba", "anthropic", "openai", "deepseek", "google", "meta",
         "mistral", "xai", "moonshotai", "zhipuai", "amazon", "ai21", "stepfun"}


class TestFiltering(unittest.TestCase):
    """信息库只收录可信赖的模型：排除社区微调、个人发布方、非对话类。"""

    def test_heretic_excluded(self):
        ok, why = filtering.decide(model_id="deepseek-v4.1-flash-1017-heretic",
                                   vendor="pqhaz")
        self.assertFalse(ok)
        self.assertIn("无审查", why)

    def test_abliterated_excluded(self):
        for mid in ("llama-3.1-8b-abliterated",
                    "deepseek-r1-distill-qwen-32b-abliterated",
                    "qwen3.8-27b-uncensored"):
            ok, _ = filtering.decide(model_id=mid)
            self.assertFalse(ok, mid)

    def test_blocked_even_under_official_vendor(self):
        """官方厂商名下的 heretic 变体也要排除。"""
        ok, _ = filtering.decide(model_id="deepseek-v4.1-flash-uncensored",
                                 vendor="deepseek")
        self.assertFalse(ok)

    def test_personal_publisher_excluded(self):
        for v in ("thedrummer", "sao10k", "huihui-ai", "abliteration-ai"):
            ok, why = filtering.decide(model_id="some-model", vendor=v)
            self.assertFalse(ok, v)

    def test_aggregator_excluded(self):
        for v in ("openrouter", "baseten", "@cf", "novita"):
            ok, _ = filtering.decide(model_id="x", vendor=v)
            self.assertFalse(ok, v)

    def test_official_models_kept(self):
        cases = [
            ("deepseek-v4.1-flash", "deepseek"),
            ("claude-sonnet-4.5", "anthropic"),
            ("gpt-5.2", "openai"),
            ("qwen3.8-max", "alibaba"),
            ("glm-5.3", "zhipuai"),
        ]
        for mid, v in cases:
            ok, why = filtering.decide(model_id=mid, vendor=v)
            self.assertTrue(ok, f"{mid} 被误排除: {why}")

    def test_non_chat_excluded(self):
        ok, why = filtering.decide(model_id="bge-m3-embedding")
        self.assertFalse(ok)

    def test_allowlist_wins(self):
        """白名单里的厂商不受黑名单影响。"""
        ok, _ = filtering.decide(model_id="sakana-fugu", vendor="sakana",
                                 allowlist={"sakana"})
        self.assertTrue(ok)


class TestAttribution(unittest.TestCase):
    """从 id 推断厂商，剥区域/渠道前缀。"""

    def test_region_prefix_stripped(self):
        self.assertEqual(attribution.infer_vendor("apac-anthropic-claude-haiku", KNOWN), "anthropic")
        self.assertEqual(attribution.infer_vendor("us-deepseek-r1", KNOWN), "deepseek")

    def test_channel_prefix_stripped(self):
        self.assertEqual(attribution.infer_vendor("databricks-gpt-5-5", KNOWN), "openai")

    def test_prefix_match(self):
        self.assertEqual(attribution.infer_vendor("alibaba-qwen3-32b", KNOWN), "alibaba")
        self.assertEqual(attribution.infer_vendor("amazon-nova-2-pro", KNOWN), "amazon")
        self.assertEqual(attribution.infer_vendor("ai21-jamba-1.5", KNOWN), "ai21")

    def test_version_suffix(self):
        self.assertEqual(attribution.infer_vendor("qwen3.8-max", KNOWN), "alibaba")
        self.assertEqual(attribution.infer_vendor("glm-5.3", KNOWN), "zhipuai")
        self.assertEqual(attribution.infer_vendor("step-3.7-flash", KNOWN), "stepfun")

    def test_unknown_stays_unknown(self):
        """判不出来就留空，不猜。"""
        self.assertIsNone(attribution.infer_vendor("agnes-2.5-pro", KNOWN))
        self.assertIsNone(attribution.infer_vendor("agent-max", KNOWN))


if __name__ == "__main__":
    unittest.main(verbosity=2)
