#!/usr/bin/env python3
"""
OpenRouter 数据源

从 OpenRouter API (https://openrouter.ai/api/v1/models) 获取模型信息。
免费，无需 API Key。

OpenRouter 是聚合平台，模型来自多个厂商。
模型 ID 格式: org/model-name (如 openai/gpt-4o)
"""

import re
from datetime import datetime, timezone
from typing import Any

from .base import BaseSource, register_source

USD_UNIT = "USD/Mtok"


@register_source
class OpenRouterSource(BaseSource):
    source_id = "openrouter"
    name = "OpenRouter"
    is_aggregator = True

    def fetch(self, alias_map: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
        """从 OpenRouter API 获取模型信息"""
        try:
            data = self.http_get_json("https://openrouter.ai/api/v1/models", timeout=60)
        except Exception as e:
            raise RuntimeError(f"OpenRouter fetch failed: {e}") from e

        raw_models = data.get("data", [])
        print(f"  {len(raw_models)} models from API", flush=True)

        result: dict[str, list[dict[str, Any]]] = {}
        now = datetime.now(timezone.utc).isoformat()

        for m in raw_models:
            model_id = m.get("id", "")
            if "/" not in model_id:
                continue

            org = model_id.split("/")[0]
            if org == "openrouter" or org.startswith("~"):
                continue

            # 大小写无关匹配
            vendor_id = alias_map.get(org.lower())
            if not vendor_id:
                continue

            # 模型 ID（去掉 org/ 前缀）
            raw_model_id = model_id.split("/", 1)[1]
            has_suffix = ":" in raw_model_id
            base_model_id = raw_model_id.split(":")[0]

            # 模型 name（去掉 "Org: " 前缀和 " (free)" 后缀）
            raw_name = m.get("name", "")
            if ": " in raw_name:
                raw_name = raw_name.split(": ", 1)[1]
            raw_name = re.sub(r"\s*\((free|extended)\)\s*$", "", raw_name, flags=re.IGNORECASE)

            entry: dict[str, Any] = {
                "id": base_model_id,
                "vendor_id": vendor_id,
                "name": raw_name,
            }

            # 别名
            model_aliases: list[str] = []
            seen_aliases: set[str] = {base_model_id.lower()}

            if has_suffix:
                suffix_alias = raw_model_id.lower()
                if suffix_alias not in seen_aliases:
                    model_aliases.append(suffix_alias)
                    seen_aliases.add(suffix_alias)

            # canonical_slug
            cs = m.get("canonical_slug", "") or ""
            if cs:
                cs_stripped = self.strip_org_prefix(cs)
                cs_normalized = self.normalize_id(cs_stripped)
                if cs_normalized and cs_normalized not in seen_aliases:
                    model_aliases.append(cs_normalized)
                    seen_aliases.add(cs_normalized)

            # hugging_face_id
            hf = m.get("hugging_face_id", "") or ""
            if hf:
                hf_stripped = self.strip_org_prefix(hf)
                hf_normalized = self.normalize_id(hf_stripped)
                if hf_normalized and hf_normalized not in seen_aliases:
                    model_aliases.append(hf_normalized)
                    seen_aliases.add(hf_normalized)

            if model_aliases:
                entry["model_aliases"] = model_aliases

            # 上下文长度
            if "context_length" in m:
                entry["context_length"] = m["context_length"]

            # 最大输出 token 数
            top_provider = m.get("top_provider", {})
            if top_provider:
                max_completion = top_provider.get("max_completion_tokens")
                if max_completion and isinstance(max_completion, (int, float)):
                    entry["max_output_tokens"] = int(max_completion)

            # 模态
            arch = m.get("architecture", {})
            if arch:
                modality = arch.get("modality", "")
                if modality and "->" in modality:
                    parts = modality.split("->")
                    input_mods = [x.strip() for x in parts[0].split("+")]
                    output_mods = [x.strip() for x in parts[1].split("+")]
                    entry["modalities"] = {"input": input_mods, "output": output_mods}

            # 价格（$/token -> $/1M tokens）
            pricing = m.get("pricing", {})
            if pricing:
                price: dict[str, Any] = {"unit": USD_UNIT}
                for price_key, entry_key in [
                    ("prompt", "input"),
                    ("completion", "output"),
                    ("input_cache_read", "cache_input"),
                    ("input_cache_write", "cache_write"),
                    ("image", "image_input"),
                    ("reasoning", "reasoning"),
                    ("web_search", "web_search"),
                ]:
                    val = pricing.get(price_key)
                    if val and val != "0":
                        try:
                            price[entry_key] = round(float(val) * 1_000_000, 6)
                        except (ValueError, TypeError):
                            pass
                if len(price) > 1:
                    entry["price"] = price

            # 能力
            supported_params = m.get("supported_parameters", [])
            if supported_params:
                capabilities: dict[str, Any] = {}
                if "tools" in supported_params or "tool_choice" in supported_params:
                    capabilities["tool_call"] = True
                if "reasoning" in supported_params or "include_reasoning" in supported_params:
                    capabilities["reasoning"] = True
                if "structured_outputs" in supported_params or "response_format" in supported_params:
                    capabilities["structured_output"] = True
                if "temperature" in supported_params:
                    capabilities["temperature"] = True
                if capabilities:
                    entry["capabilities"] = capabilities

            # 知识截止
            if "knowledge_cutoff" in m:
                entry["knowledge_cutoff"] = m["knowledge_cutoff"]

            # 创建时间
            if "created" in m and isinstance(m["created"], (int, float)):
                entry["created_at"] = datetime.fromtimestamp(
                    m["created"], tz=timezone.utc
                ).isoformat()

            # 描述
            if "description" in m and m["description"]:
                entry["description"] = m["description"][:500]

            result.setdefault(vendor_id, []).append(entry)

        return result
