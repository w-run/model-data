#!/usr/bin/env python3
"""
W/Run mimi 数据源

从我们的自有聚合网关 https://mimi.wrun.cc/v1/models 读取 mimi-* 系列模型的
规格与能力。这些模型在 OpenRouter / models.dev 等公开目录里都不存在，是它们
唯一的能力数据来源。

该端点默认公开可读；若日后加了鉴权，用环境变量 MIMI_API_KEY 注入。

W/Run 是厂商专属数据源（vendor_id = w-run），所有模型归属 W/Run。

字段映射：
  max_input_tokens   -> context_length
  max_output_tokens  -> max_output_tokens
  input_modalities   -> modalities.input（output 恒为 text）
  supports_tool_call -> capabilities.tool_call
  supports_reasoning -> capabilities.reasoning
  supports_images    -> modalities.input 补 image
"""

import os
import re
from datetime import datetime, timezone
from typing import Any

from .base import BaseSource, register_source

USD_UNIT = "USD/Mtok"
VENDOR_ID = "w-run"
DEFAULT_ENDPOINT = "https://mimi.wrun.cc/v1/models"


@register_source
class MimiSource(BaseSource):
    source_id = "mimi"
    name = "W/Run mimi"
    is_aggregator = False

    def fetch(self, alias_map: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
        endpoint = os.environ.get("MIMI_MODELS_URL", DEFAULT_ENDPOINT)
        try:
            data = self.http_get_json(endpoint, timeout=30)
        except Exception as e:
            raise RuntimeError(f"Mimi fetch failed: {e}") from e

        raw_models = data.get("data", []) or []
        print(f"  {len(raw_models)} models from {endpoint}", flush=True)

        now = datetime.now(timezone.utc).isoformat()
        models: list[dict[str, Any]] = []

        for m in raw_models:
            raw_id = str(m.get("id", "")).strip()
            if not raw_id:
                continue

            model_id = self.normalize_id(raw_id)
            name = str(m.get("name") or raw_id)

            entry: dict[str, Any] = {
                "id": model_id,
                "vendor_id": VENDOR_ID,
                "name": name,
            }

            # 别名：原始 ID 与 name 都保留，方便各项目按自己的写法匹配
            aliases = {model_id, self.normalize_id(name)}
            if self.normalize_id(raw_id) != raw_id.lower():
                aliases.add(raw_id.lower())
            aliases.discard(model_id)
            aliases.discard("")
            if aliases:
                entry["model_aliases"] = sorted(aliases)

            ctx = m.get("max_input_tokens")
            if isinstance(ctx, (int, float)) and ctx > 0:
                entry["context_length"] = int(ctx)

            max_out = m.get("max_output_tokens")
            if isinstance(max_out, (int, float)) and max_out > 0:
                entry["max_output_tokens"] = int(max_out)

            # 模态：以上游 input_modalities 为准，supports_images 兜底
            modalities_in = m.get("input_modalities")
            if not isinstance(modalities_in, list) or not modalities_in:
                modalities_in = ["text"]
                if m.get("supports_images"):
                    modalities_in.append("image")
                if m.get("supports_video"):
                    modalities_in.append("video")
            entry["modalities"] = {"input": list(modalities_in), "output": ["text"]}

            # 能力
            capabilities: dict[str, Any] = {}
            if m.get("supports_tool_call"):
                capabilities["tool_call"] = True
            if m.get("supports_reasoning"):
                capabilities["reasoning"] = True
            capabilities["temperature"] = True
            entry["capabilities"] = capabilities

            # 计费：mimi 系列对终端免费（上游以 credits 计），这里标 0 便于消费方判断
            entry["price"] = {"unit": USD_UNIT, "input": 0.0, "output": 0.0}

            entry["kind"] = str(m.get("kind") or "chat")
            entry["source"] = self.source_id
            entry["updated_at"] = now
            models.append(entry)

        return {VENDOR_ID: models}
