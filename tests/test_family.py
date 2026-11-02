#!/usr/bin/env python3
"""
家族/别名规则的测试。

核心：规则来自主流信息源的 canonical 表达，不自己造。
"""

from __future__ import annotations

import unittest

from model_data import family


class TestFamilyOf(unittest.TestCase):
    """家族 = canonical 去掉尾部完整日期。"""

    def test_full_date_stripped(self):
        self.assertEqual(family.family_of("deepseek-v4-flash-20260423"),
                         "deepseek-v4-flash")
        self.assertEqual(family.family_of("gpt-6.1-terra-20261026"),
                         "gpt-6.1-terra")

    def test_no_date_is_own_family(self):
        self.assertEqual(family.family_of("claude-opus-4-8"), "claude-opus-4-8")


class TestShortDateFallback(unittest.TestCase):
    """
    canonical 不自带完整日期时，按短日期 / -latest 归族。
    这是 models.dev 独有条目必需的兜底。
    """

    def test_short_date(self):
        self.assertEqual(family.family_of_with_short_date("deepseek-v4-flash-0423"),
                         "deepseek-v4-flash")

    def test_latest(self):
        self.assertEqual(family.family_of_with_short_date("deepseek-v4-flash-latest"),
                         "deepseek-v4-flash")

    def test_full_date(self):
        self.assertEqual(
            family.family_of_with_short_date("deepseek-v4-flash-20260731"),
            "deepseek-v4-flash")

    def test_plain_name_untouched(self):
        self.assertEqual(family.family_of_with_short_date("gpt-5.2"), "gpt-5.2")


class TestOpenRouterCanonical(unittest.TestCase):
    """
    OpenRouter 的 canonical_slug 精确表达了「谁是快照、谁是滚动别名」：
        deepseek-v4-flash        → deepseek-v4-flash-20260423
        deepseek-v4-flash-0731   → deepseek-v4-flash-20260731
    两者同族，但各自保留自己的 canonical。
    """

    def setUp(self):
        self.data = {"data": [
            {"id": "deepseek/deepseek-v4-flash",
             "canonical_slug": "deepseek/deepseek-v4-flash-20260423"},
            {"id": "deepseek/deepseek-v4-flash-0731",
             "canonical_slug": "deepseek/deepseek-v4-flash-20260731"},
            {"id": "openai/gpt-6.1-terra",
             "canonical_slug": "openai/gpt-6.1-terra-20261026"},
        ]}
        self.idx = family.build_index(("openrouter", self.data))

    def test_distinct_canonicals(self):
        self.assertNotEqual(self.idx.resolve("deepseek-v4-flash"),
                            self.idx.resolve("deepseek-v4-flash-0731"))

    def test_same_family(self):
        self.assertEqual(self.idx.family("deepseek-v4-flash"),
                         self.idx.family("deepseek-v4-flash-0731"))

    def test_members_include_both_snapshots(self):
        m = self.idx.members("deepseek-v4-flash")
        self.assertIn("deepseek-v4-flash-20260423", m)
        self.assertIn("deepseek-v4-flash-20260731", m)

    def test_aliases_expand(self):
        a = self.idx.aliases_of("deepseek-v4-flash")
        self.assertIn("deepseek-v4-flash-0731", a)


class TestPrimarySelection(unittest.TestCase):
    """
    族代表选择：别名不能抢走主名。

    事故：deepseek-flash（官方别名，指向 -1017）因名字最短被选为 primary，
    结果整族以它命名，deepseek-v4.1-flash 在产出中消失。
    """

    def test_alias_does_not_win(self):
        """过时别名的快照日期更旧，应让位给滚动主名。"""
        # deepseek-flash  → -1017（旧）
        # deepseek-v4-1-flash → -20260910（新）
        old = family.snapshot_date("deepseek-v4-1-flash-1017")
        new = family.snapshot_date("deepseek-v4-1-flash-20260910")
        self.assertLess(old, new, "快照日期应可比较，用于选出更新的别名")

    def test_short_date_snapshot_not_primary(self):
        """-260910 这类短日期快照不能当族代表。"""
        self.assertIsNotNone(family.snapshot_date("deepseek-v4-1-flash-260910"))

    def test_service_tier_folded(self):
        """速度档并入主条目。"""
        self.assertEqual(
            family.family_of_with_short_date("kimi-k3.1-pro-highspeed"),
            family.family_of_with_short_date("kimi-k3.1-pro"))


class TestPrefixStripping(unittest.TestCase):
    """区域/渠道前缀不属于模型身份。"""

    def test_region(self):
        self.assertEqual(family.strip_provider_prefix("apac-anthropic-claude-opus-4-7"),
                         "anthropic-claude-opus-4-7")

    def test_channel(self):
        self.assertEqual(family.strip_provider_prefix("databricks-claude-opus-4-8"),
                         "claude-opus-4-8")

    def test_slash_form(self):
        self.assertEqual(family.strip_provider_prefix("github_copilot/claude-opus-4.8"),
                         "claude-opus-4.8")

    def test_plain_untouched(self):
        self.assertEqual(family.strip_provider_prefix("gpt-5.2"), "gpt-5.2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
