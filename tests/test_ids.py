#!/usr/bin/env python3
"""
ID 归一 —— 全库唯一实现的测试。

这个模块取代了原来 5 个重叠的实现（idnorm / identity / official_rules /
family / attribution 的前缀部分）。每个用例对应一次真实事故。
"""

from __future__ import annotations

import unittest

from model_data import ids


class TestPrefixStripping(unittest.TestCase):
    """渠道/区域前缀要剥，厂商名是模型名一部分时不剥。"""

    def test_channel_prefix(self):
        self.assertEqual(ids.canonical_id("databricks-claude-opus-4-8"),
                         "claude-opus-4.8")
        self.assertEqual(ids.canonical_id("github_copilot/claude-opus-4.8"),
                         "claude-opus-4.8")

    def test_region_prefix(self):
        self.assertEqual(ids.canonical_id("apac-anthropic-claude-opus-4-7"),
                         "claude-opus-4.7")
        self.assertEqual(ids.canonical_id("global-openai-gpt-6.1-terra"),
                         "gpt-6.1-terra")

    def test_vendor_name_kept_when_part_of_model(self):
        """deepseek-v4.1-flash 里的 deepseek 是模型名，不能剥成 v4.1-flash。"""
        self.assertEqual(ids.canonical_id("deepseek-v4.1-flash"),
                         "deepseek-v4.1-flash")
        self.assertEqual(ids.canonical_id("deepseek-r1"), "deepseek-r1")

    def test_vendor_prefix_with_own_series(self):
        """anthropic-claude-… 剥成 claude-…（claude 是专属系列名）。"""
        self.assertEqual(ids.canonical_id("anthropic-claude-opus-4-5"),
                         "claude-opus-4.5")
        self.assertEqual(ids.canonical_id("openai-gpt-6.1-terra"),
                         "gpt-6.1-terra")


class TestPEscape(unittest.TestCase):
    """id 不允许点号时用 p 代替：v4p1 == v4.1。"""

    def test_p_escape(self):
        self.assertEqual(ids.canonical_id("deepseek-v4p1-flash"),
                         "deepseek-v4.1-flash")
        self.assertEqual(ids.canonical_id("glm-4p5"), "glm-4.5")

    def test_converges_with_dot_form(self):
        self.assertEqual(ids.canonical_id("deepseek-v4p1-flash"),
                         ids.canonical_id("deepseek-v4.1-flash"))

    def test_normal_p_untouched(self):
        """普通单词里的 p 不能转：gpt / phi 等。"""
        self.assertEqual(ids.canonical_id("gpt-5.2"), "gpt-5.2")
        self.assertEqual(ids.canonical_id("phi-3"), "phi-3")


class TestVersionDots(unittest.TestCase):
    """全库统一：版本用小数点。"""

    def test_version_pair(self):
        self.assertEqual(ids.canonical_id("llama-3-1-8b"), "llama-3.1-8b")
        self.assertEqual(ids.canonical_id("doubao-seed-2-0-pro"),
                         "doubao-seed-2.0-pro")
        self.assertEqual(ids.canonical_id("gpt-3-5-turbo"), "gpt-3.5-turbo")

    def test_size_markers_untouched(self):
        self.assertEqual(ids.canonical_id("qwen3-vl-235b-a22b"),
                         "qwen3-vl-235b-a22b")

    def test_date_untouched(self):
        self.assertEqual(ids.canonical_id("gpt-4o-mini-2024-07-18"),
                         "gpt-4o-mini-2024-07-18")


class TestClaude(unittest.TestCase):
    """Claude 也走统一口径（小数点），官方写法作为别名保留。"""

    def test_official_form_to_dots(self):
        self.assertEqual(ids.canonical_id("claude-opus-4-5"), "claude-opus-4.5")
        self.assertEqual(ids.canonical_id("claude-haiku-4-5"), "claude-haiku-4.5")

    def test_major_omits_minor(self):
        self.assertEqual(ids.canonical_id("claude-opus-4-0"), "claude-opus-4")

    def test_snapshot_kept(self):
        self.assertEqual(ids.canonical_id("claude-opus-4-5-20251101"),
                         "claude-opus-4.5-20251101")

    def test_date_not_mistaken_for_minor(self):
        """claude-opus-4-20250514 的 20250514 是日期，不是 minor 版本。"""
        self.assertEqual(ids.canonical_id("claude-opus-4-20250514"),
                         "claude-opus-4-20250514")


class TestFamilyKey(unittest.TestCase):
    """家族键 = canonical 去掉日期快照。"""

    def test_snapshot_folded(self):
        self.assertEqual(ids.family_key("deepseek-v4.1-flash-1017"),
                         "deepseek-v4.1-flash")
        self.assertEqual(ids.family_key("claude-opus-4.5-20251101"),
                         "claude-opus-4.5")

    def test_same_family_for_alias_forms(self):
        self.assertEqual(ids.family_key("deepseek-v4p1-flash"),
                         ids.family_key("deepseek-v4.1-flash"))

    def test_no_snapshot(self):
        self.assertEqual(ids.family_key("glm-5.3"), "glm-5.3")


class TestSnapshotDate(unittest.TestCase):
    def test_short_and_full(self):
        self.assertEqual(ids.snapshot_date("deepseek-v4.1-flash-1017"), "1017")
        self.assertEqual(ids.snapshot_date("claude-opus-4.5-20251101"), "20251101")
        self.assertEqual(ids.snapshot_date("glm-5.3"), None)


class TestIdempotent(unittest.TestCase):
    """归一必须幂等，否则族键与输出 id 会不同步。"""

    def test_idempotent(self):
        for x in ("deepseek-v4p1-flash", "claude-opus-4-5",
                  "databricks-claude-opus-4-8", "llama-3-1-8b"):
            once = ids.canonical_id(x)
            self.assertEqual(ids.canonical_id(once), once, x)


if __name__ == "__main__":
    unittest.main(verbosity=2)
