#!/usr/bin/env python3
"""收录过滤与厂商归属推断的测试。"""

from __future__ import annotations

import unittest

from model_data import attribution, filtering

KNOWN = {"alibaba", "anthropic", "openai", "deepseek", "google", "meta",
         "mistral", "xai", "moonshotai", "zhipuai", "amazon", "ai21", "stepfun"}


ALLOW = {"deepseek", "anthropic", "openai", "alibaba", "zhipuai", "meta"}


class TestFiltering(unittest.TestCase):
    """白名单制：只有登记过的厂商才收录。"""

    def test_heretic_excluded(self):
        ok, why = filtering.decide(model_id="deepseek-v4.1-flash-1017-heretic",
                                   vendor="pqhaz", allowlist=ALLOW)
        self.assertFalse(ok)
        self.assertIn("无审查", why)

    def test_unlisted_vendor_excluded(self):
        """不在白名单的厂商一律不收 —— 这是白名单制的核心。"""
        for v in ("thedrummer", "sao10k", "orcamini", "particle-ai", "some-new-lab"):
            ok, why = filtering.decide(model_id="x-model", vendor=v, allowlist=ALLOW)
            self.assertFalse(ok, v)
            self.assertIn("厂商未收录", why)

    def test_listed_vendor_kept(self):
        for v in ALLOW:
            ok, why = filtering.decide(model_id="some-model", vendor=v, allowlist=ALLOW)
            self.assertTrue(ok, f"{v} 未被收录: {why}")

    def test_unknown_vendor_excluded(self):
        ok, why = filtering.decide(model_id="mystery-model", vendor="unknown",
                                   allowlist=ALLOW)
        self.assertFalse(ok)

    def test_abliterated_excluded(self):
        for mid in ("llama-3.1-8b-abliterated",
                    "deepseek-r1-distill-qwen-32b-abliterated",
                    "qwen3.8-27b-uncensored"):
            ok, _ = filtering.decide(model_id=mid, vendor="meta", allowlist=ALLOW)
            self.assertFalse(ok, mid)

    def test_blocked_even_under_official_vendor(self):
        """官方厂商名下的 heretic 变体也要排除。"""
        ok, _ = filtering.decide(model_id="deepseek-v4.1-flash-uncensored",
                                 vendor="deepseek", allowlist=ALLOW)
        self.assertFalse(ok)

    def test_deny_list_beats_allowlist(self):
        """即便误把聚合商加进白名单，也仍然排除。"""
        for v in ("openrouter", "baseten", "@cf", "novita", "unknown"):
            ok, _ = filtering.decide(model_id="x", vendor=v,
                                     allowlist=ALLOW | {v})
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
            ok, why = filtering.decide(model_id=mid, vendor=v, allowlist=ALLOW)
            self.assertTrue(ok, f"{mid} 被误排除: {why}")

    def test_non_chat_excluded(self):
        ok, why = filtering.decide(model_id="bge-m3-embedding", vendor="meta",
                                   allowlist=ALLOW)
        self.assertFalse(ok)


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
