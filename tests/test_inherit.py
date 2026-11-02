#!/usr/bin/env python3
"""变体继承与跨源 ID 对齐的测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inherit import apply_inheritance, base_id  # noqa: E402


class TestBaseId(unittest.TestCase):
    def test_strips_batch(self):
        self.assertEqual(base_id("claude-opus-4.5:batch"), "claude-opus-4.5")
        self.assertEqual(base_id("gpt-5.2:free"), "gpt-5.2")

    def test_vision_is_different_model(self):
        """vision / exp / preview 是不同模型，窗口可能不同，不继承。"""
        self.assertIsNone(base_id("deepseek-v4-flash-vision-exp"))
        self.assertIsNone(base_id("gemini-3.1-flash-image-preview"))

    def test_non_variant(self):
        self.assertIsNone(base_id("gpt-5.2"))
        self.assertIsNone(base_id("deepseek-v4.1-flash"))


class TestInheritance(unittest.TestCase):
    def test_inherits_when_inferred(self):
        models = {
            "claude-haiku-4.5": {
                "id": "claude-haiku-4.5",
                "limit": {"context": 200000, "max_output": 64000,
                          "confidence": "first_party", "votes": 28},
                "recommended": {"input_budget": 136000, "output_budget": 64000},
            },
            "claude-haiku-4.5:batch": {
                "id": "claude-haiku-4.5:batch",
                "limit": {"context": 1048576, "max_output": 943718,
                          "confidence": "inferred", "votes": 0},
                "recommended": {"input_budget": 104858, "output_budget": 943718},
            },
        }
        n = apply_inheritance(models)
        self.assertEqual(n, 1)
        v = models["claude-haiku-4.5:batch"]
        self.assertEqual(v["limit"]["context"], 200000)
        self.assertEqual(v["limit"]["max_output"], 64000)
        self.assertEqual(v["limit"]["confidence"], "first_party")
        # 预算必须跟着重算，不能留着 104858/943718
        self.assertEqual(v["recommended"]["output_budget"], 64000)
        self.assertEqual(v["recommended"]["input_budget"], 136000)

    def test_does_not_override_confident(self):
        """自己有可信数据的变体不该被基础模型覆盖。"""
        models = {
            "x": {"id": "x", "limit": {"context": 100000, "max_output": 8000,
                                       "confidence": "first_party"}, "recommended": {}},
            "x:batch": {"id": "x:batch", "limit": {"context": 200000, "max_output": 64000,
                                                   "confidence": "consensus"}, "recommended": {}},
        }
        self.assertEqual(apply_inheritance(models), 0)
        self.assertEqual(models["x:batch"]["limit"]["context"], 200000)

    def test_no_base_available(self):
        models = {
            "y:batch": {"id": "y:batch", "limit": {"context": 1000, "max_output": 900,
                                                   "confidence": "inferred"}, "recommended": {}},
        }
        self.assertEqual(apply_inheritance(models), 0)


class TestLooseKey(unittest.TestCase):
    def test_not_called_cross_module(self):
        """loose_key 只用于对齐，确认它在 build 里可用且不破坏 id 归一化。"""
        from scripts.build import loose_key
        self.assertEqual(loose_key("claude-sonnet-4.5"), loose_key("claude-sonnet-4-5"))
        self.assertEqual(loose_key("gpt-5.2"), "gpt-5-2")
        # 不同的模型不能被合并
        self.assertNotEqual(loose_key("gpt-5.2"), loose_key("gpt-5.3"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
