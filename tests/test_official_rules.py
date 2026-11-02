#!/usr/bin/env python3
"""
厂商官方 ID 规则的测试。

每个用例都对应官方文档的一句原文，或一次真实事故。
"""

from __future__ import annotations

import unittest

from model_data import official_rules as R


class TestClaudeGenerations(unittest.TestCase):
    """
    官方原文：4.6 及以后的无日期 ID 是规范 ID（固定快照）；
    4.6 之前的无日期 ID 是别名，指向最新日期快照。
    """

    def test_46_and_later_dateless_is_canonical(self):
        for mid in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-5",
                    "claude-sonnet-5", "claude-opus-4-8"):
            p = R.parse_claude(mid)
            self.assertIsNotNone(p, mid)
            self.assertTrue(p["dateless_is_canonical"], mid)

    def test_before_46_dateless_is_alias(self):
        for mid in ("claude-opus-4", "claude-sonnet-4", "claude-sonnet-4-5",
                    "claude-haiku-4-5"):
            p = R.parse_claude(mid)
            self.assertIsNotNone(p, mid)
            self.assertFalse(p["dateless_is_canonical"], mid)

    def test_major_omits_minor(self):
        """官方：主版本发布省略 minor 段（Claude Opus 4 没有 -0）。"""
        self.assertEqual(R.canonical_claude_id("claude-opus-4-0"), "claude-opus-4")
        self.assertEqual(R.canonical_claude_id("claude-sonnet-5-0"), "claude-sonnet-5")


class TestClaudeSnapshots(unittest.TestCase):
    def test_date_folds_to_base(self):
        self.assertEqual(R.claude_snapshot_base("claude-sonnet-4-5-20250929"),
                         "claude-sonnet-4-5")
        self.assertEqual(R.claude_snapshot_base("claude-opus-4-20250514"),
                         "claude-opus-4")

    def test_at_separator_is_same_snapshot(self):
        """官方：Google Cloud 用 @ 分隔日期，与 - 是同一快照。"""
        self.assertEqual(R.claude_snapshot_base("claude-haiku-4-5@20251001"),
                         "claude-haiku-4-5")

    def test_dateless_is_not_snapshot(self):
        self.assertIsNone(R.claude_snapshot_base("claude-opus-4-6"))


class TestRejectThirdPartyFormat(unittest.TestCase):
    """
    OpenRouter 等第三方用 claude-4.8-opus-20260528 这种倒序写法，
    结构不符合官方定义，不能与官方 ID 混用。
    """

    def test_reversed_format_rejected(self):
        for mid in ("claude-4.8-opus-20260528", "claude-4-opus",
                    "claude-4.5-sonnet-20250929"):
            self.assertIsNone(R.parse_claude(mid), mid)

    def test_non_claude_rejected(self):
        self.assertIsNone(R.parse_claude("gpt-5.2"))


class TestPrefixStripping(unittest.TestCase):
    """区域/渠道前缀必须先剥，否则会掉进通用规则产生错误的 id。"""

    def test_region_prefix(self):
        self.assertEqual(R.strip_cloud_prefix("apac-anthropic-claude-opus-4-7"),
                         "claude-opus-4-7")
        self.assertEqual(R.strip_cloud_prefix("us.anthropic.claude-opus-4-8"),
                         "claude-opus-4-8")

    def test_channel_prefix(self):
        self.assertEqual(R.strip_cloud_prefix("databricks-claude-opus-4-8"),
                         "claude-opus-4-8")
        self.assertEqual(R.strip_cloud_prefix("github_copilot/claude-opus-4.8"),
                         "claude-opus-4.8")

    def test_dot_form_normalized(self):
        self.assertEqual(R.canonical_claude_id("claude-opus-4.8"),
                         "claude-opus-4-8")


class TestOutputIdConsistency(unittest.TestCase):
    """
    族键与输出 id 必须一致，否则同一模型会拆成多条（曾经造成重复卡片）。
    """

    def test_channel_variants_converge(self):
        from model_data.assemble import canonical_output_id
        forms = [
            "claude-opus-4-8",
            "claude-opus-4.8",
            "databricks-claude-opus-4-8",
            "github_copilot/claude-opus-4.8",
            "apac-anthropic-claude-opus-4-8",
        ]
        outs = {canonical_output_id(f) for f in forms}
        self.assertEqual(outs, {"claude-opus-4-8"}, outs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
