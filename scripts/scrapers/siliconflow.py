#!/usr/bin/env python3
"""
硅基流动(SiliconFlow) 数据源

从 SiliconFlow API (https://api.siliconflow.cn/v1/models) 获取模型列表。
需要 API Key（环境变量 SILICONFLOW_API_KEY）。

SiliconFlow 是聚合平台，模型来自多个开源厂商。
模型 ID 格式:
  - 标准: org/model-name (如 deepseek-ai/DeepSeek-V4-Pro)
  - Pro版: Pro/org/model-name (如 Pro/deepseek-ai/DeepSeek-V3.2)
  - LoRA版: LoRA/org/model-name (跳过)
"""

import os
from datetime import datetime, timezone
from typing import Any

from .base import BaseSource, register_source

SF_API_BASE = "https://api.siliconflow.cn/v1"
SF_API_KEY_ENV = "SILICONFLOW_API_KEY"


@register_source
class SiliconFlowSource(BaseSource):
    source_id = "siliconflow"
    name = "硅基流动"
    is_aggregator = True

    # 硅基流动特有的 org -> vendor_id 映射
    # 补充 vendors.toml alias_map 中没有的映射
    EXTRA_ORG_MAP: dict[str, str] = {
        "qwen": "alibaba",
        "thudm": "z-ai",
        "zai-org": "z-ai",
        "minimaxai": "minimax",
        "moonshotai": "moonshot",
        "stepfun-ai": "stepfun",
        "tencent": "tencent",
        "baidu": "baidu",
        "deepseek-ai": "deepseek",
        "teleai": "tencent",
        "tongyi-mai": "alibaba",
        "wan-ai": "alibaba",
        "inclusionai": "alibaba",
        "funaudiollm": "alibaba",
        "nex-agi": "stepfun",
        "kwai-kolors": "xiaomi",
    }

    def fetch(self, alias_map: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
        """从硅基流动 API 获取模型列表"""
        api_key = os.environ.get(SF_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"SiliconFlow API key not found (set {SF_API_KEY_ENV} env var)"
            )

        try:
            data = self.http_get_json(
                f"{SF_API_BASE}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
        except Exception as e:
            raise RuntimeError(f"SiliconFlow fetch failed: {e}") from e

        raw_models = data.get("data", [])
        print(f"  {len(raw_models)} models from API", flush=True)

        # 构建完整的 org 映射
        org_map = dict(alias_map)
        for org_key, vid in self.EXTRA_ORG_MAP.items():
            if org_key not in org_map:
                org_map[org_key] = vid

        result: dict[str, list[dict[str, Any]]] = {}
        now = datetime.now(timezone.utc).isoformat()
        skipped = 0

        for m in raw_models:
            model_id = m.get("id", "")
            if not model_id:
                continue

            # 解析模型 ID
            parts = model_id.split("/")
            prefix = None
            if parts[0] == "Pro" and len(parts) == 3:
                prefix = "Pro"
                org = parts[1]
                model_name = parts[2]
            elif parts[0] == "LoRA":
                continue
            elif len(parts) == 2:
                org = parts[0]
                model_name = parts[1]
            else:
                continue

            # 匹配厂商
            vendor_id = org_map.get(org.lower())
            if not vendor_id:
                skipped += 1
                continue

            base_id = self.normalize_id(model_name)

            entry: dict[str, Any] = {
                "id": base_id,
                "vendor_id": vendor_id,
                "name": model_name,
                "open_weights": True,
            }

            # 别名
            model_aliases: list[str] = []
            seen_aliases: set[str] = {base_id}

            if prefix == "Pro":
                pro_alias = f"pro-{base_id}"
                if pro_alias not in seen_aliases:
                    model_aliases.append(pro_alias)
                    seen_aliases.add(pro_alias)

            sf_alias = self.normalize_id(model_id)
            if sf_alias not in seen_aliases:
                model_aliases.append(sf_alias)
                seen_aliases.add(sf_alias)

            if model_aliases:
                entry["model_aliases"] = model_aliases

            # 模态推断
            entry["modalities"] = self._infer_modalities(model_name)

            # 能力推断
            caps = self._infer_capabilities(model_name)
            if caps:
                entry["capabilities"] = caps

            result.setdefault(vendor_id, []).append(entry)

        if skipped:
            print(f"  Skipped {skipped} models (no vendor mapping)", flush=True)

        return result

    @staticmethod
    def _infer_modalities(model_name: str) -> dict[str, list[str]]:
        """从模型名推断模态"""
        nl = model_name.lower()
        if any(kw in nl for kw in ["image", "flux", "sdxl", "kolors", "stable-diffusion"]):
            return {"input": ["text"], "output": ["image"]}
        if any(kw in nl for kw in ["video", "i2v", "t2v", "wan2"]):
            return {"input": ["text", "image"], "output": ["video"]}
        if any(kw in nl for kw in ["tts", "speech", "asr", "cosyvoice", "sensevoice", "moss-ttsd"]):
            return {"input": ["text"], "output": ["audio"]}
        if any(kw in nl for kw in ["vl", "vision", "4.5v"]):
            return {"input": ["text", "image"], "output": ["text"]}
        return {"input": ["text"], "output": ["text"]}

    @staticmethod
    def _infer_capabilities(model_name: str) -> dict[str, Any] | None:
        """从模型名推断能力"""
        nl = model_name.lower()
        exclude = ["embed", "rerank", "bge-", "image", "flux", "sdxl", "kolors",
                    "stable-diffusion", "video", "i2v", "t2v", "wan2",
                    "tts", "speech", "asr", "cosyvoice", "sensevoice"]
        if any(kw in nl for kw in exclude):
            return None

        caps: dict[str, Any] = {"tool_call": True, "temperature": True}
        if any(kw in nl for kw in ["r1", "reasoning", "z1", "thinking"]):
            caps["reasoning"] = True
        return caps
