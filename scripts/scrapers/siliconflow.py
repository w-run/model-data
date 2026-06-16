#!/usr/bin/env python3
"""
硅基流动(SiliconFlow) API 模型列表及价格爬虫

模型列表 API：
  https://api.siliconflow.cn/v1/models
  兼容 OpenAI 格式，需要 API Key 认证（环境变量 SILICONFLOW_API_KEY）

  返回数据格式简洁：
  {
    "id": "deepseek-ai/DeepSeek-V4-Pro",
    "object": "model",
    "created": 0,
    "owned_by": ""
  }

  模型 ID 格式:
    - 标准: org/model-name (如 deepseek-ai/DeepSeek-V4-Pro)
    - Pro版: Pro/org/model-name (如 Pro/deepseek-ai/DeepSeek-V3.2)
    - LoRA版: LoRA/org/model-name (如 LoRA/Qwen/Qwen2.5-7B-Instruct)

  org 前缀用于匹配厂商（通过 alias_map）

价格页面：
  https://docs.siliconflow.cn/cn/userguide/pricing
  SPA 页面，数据通过 JS 渲染。使用 API 替代爬取。
"""

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .base import BaseScraper, register_scraper

# 硅基流动 API 地址
SF_API_BASE = "https://api.siliconflow.cn/v1"

# 环境变量名
SF_API_KEY_ENV = "SILICONFLOW_API_KEY"


def _get_api_key() -> str | None:
    """从环境变量获取 API Key"""
    return os.environ.get(SF_API_KEY_ENV)


def _parse_model_id(model_id: str) -> tuple[str, str, str | None]:
    """
    解析硅基流动模型 ID

    返回: (org, model_name, prefix)
      - org: 厂商组织名 (如 "deepseek-ai")
      - model_name: 模型名 (如 "DeepSeek-V4-Pro")
      - prefix: 前缀 (如 "Pro", "LoRA") 或 None

    示例:
      "deepseek-ai/DeepSeek-V4-Pro" -> ("deepseek-ai", "DeepSeek-V4-Pro", None)
      "Pro/deepseek-ai/DeepSeek-V3.2" -> ("deepseek-ai", "DeepSeek-V3.2", "Pro")
      "LoRA/Qwen/Qwen2.5-7B-Instruct" -> ("Qwen", "Qwen2.5-7B-Instruct", "LoRA")
    """
    parts = model_id.split("/")
    if len(parts) == 2:
        return parts[0], parts[1], None
    elif len(parts) == 3 and parts[0] in ("Pro", "LoRA"):
        return parts[1], parts[2], parts[0]
    elif len(parts) >= 2:
        # 未知格式，取最后两段
        return parts[-2], parts[-1], "/".join(parts[:-2])
    else:
        return "", model_id, None


def _infer_modalities(model_name: str, org: str) -> dict[str, list[str]] | None:
    """根据模型名和组织推断模态"""
    name_lower = model_name.lower()

    # 图像生成模型
    if any(kw in name_lower for kw in ["image", "flux", "sdxl", "kolors", "stable-diffusion"]):
        return {"input": ["text"], "output": ["image"]}

    # 视频生成模型
    if any(kw in name_lower for kw in ["video", "i2v", "t2v", "wan2"]):
        return {"input": ["text", "image"], "output": ["video"]}

    # 语音模型
    if any(kw in name_lower for kw in ["tts", "speech", "asr", "cosyvoice", "sensevoice"]):
        return {"input": ["text"], "output": ["audio"]}

    # 视觉语言模型
    if any(kw in name_lower for kw in ["vl", "vision", "4.5v"]):
        return {"input": ["text", "image"], "output": ["text"]}

    # Embedding/Reranker
    if any(kw in name_lower for kw in ["embed", "rerank", "bge-"]):
        return {"input": ["text"], "output": ["text"]}

    # 默认文本模型
    return {"input": ["text"], "output": ["text"]}


def _infer_capabilities(model_name: str) -> dict[str, Any] | None:
    """根据模型名推断能力"""
    name_lower = model_name.lower()
    caps: dict[str, Any] = {}

    # Embedding/Reranker 不是对话模型
    if any(kw in name_lower for kw in ["embed", "rerank", "bge-"]):
        return None

    # 图像/视频生成不是对话模型
    if any(kw in name_lower for kw in ["image", "flux", "sdxl", "kolors", "stable-diffusion",
                                         "video", "i2v", "t2v", "wan2", "tts", "speech",
                                         "asr", "cosyvoice", "sensevoice"]):
        return None

    # 推理模型
    if any(kw in name_lower for kw in ["r1", "reasoning", "z1", "thinking"]):
        caps["reasoning"] = True

    # 大语言模型通用能力
    caps["tool_call"] = True
    caps["temperature"] = True

    return caps if caps else None


def _is_chat_model(model_name: str) -> bool:
    """判断是否为对话/文本生成模型（排除 embedding/reranker/图像/视频等）"""
    name_lower = model_name.lower()
    exclude_keywords = [
        "embed", "rerank", "bge-", "bge",
        "image", "flux", "sdxl", "kolors", "stable-diffusion",
        "video", "i2v", "t2v", "wan2",
        "tts", "speech", "asr", "cosyvoice", "sensevoice", "moss-ttsd",
    ]
    return not any(kw in name_lower for kw in exclude_keywords)


@register_scraper
class SiliconFlowScraper(BaseScraper):
    vendor_id = "siliconflow"
    name = "硅基流动"
    doc_url = "https://docs.siliconflow.cn"
    price_url = "https://docs.siliconflow.cn/cn/userguide/pricing"

    def _api_get(self, path: str) -> Any:
        """发送 API 请求"""
        api_key = _get_api_key()
        if not api_key:
            raise RuntimeError(
                f"SiliconFlow API key not found. Set environment variable {SF_API_KEY_ENV}"
            )

        url = f"{SF_API_BASE}{path}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(f"SiliconFlow API HTTP {e.code}: {body[:500]}") from e

    def scrape_models(self) -> list[dict[str, Any]]:
        """
        从硅基流动 API 获取模型列表

        硅基流动是中转商/聚合商，其模型来自各厂商。
        我们需要将 org 前缀映射到 vendors.toml 中对应的厂商。
        """
        data = self._api_get("/models")
        raw_models = data.get("data", [])
        if not raw_models:
            raise RuntimeError("No models returned from SiliconFlow API")

        all_models: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for m in raw_models:
            model_id = m.get("id", "")
            if not model_id:
                continue

            org, model_name, prefix = _parse_model_id(model_id)

            # 跳过 LoRA 模型（微调服务，不是独立模型）
            if prefix == "LoRA":
                continue

            # 生成标准化 model_id（不含 org 前缀，小写短横线分隔）
            # 保留原始名称中的大写和点号，normalize_id 统一处理
            base_id = model_name.replace("/", "-").replace("_", "-")
            base_id = re.sub(r"-+", "-", base_id).strip("-")
            normalized_id = base_id.lower()

            # 去重
            if normalized_id in seen_ids:
                continue
            seen_ids.add(normalized_id)

            entry: dict[str, Any] = {
                "id": normalized_id,
                "name": model_name,
                "open_weights": True,  # 硅基流动只提供开源模型
            }

            # Pro 版本别名
            model_aliases: list[str] = []
            if prefix == "Pro":
                pro_alias = f"pro-{normalized_id}"
                if pro_alias != normalized_id:
                    model_aliases.append(pro_alias)

            # 硅基流动原始 ID 作为别名（去掉 org 前缀）
            sf_alias = model_id.lower().replace("/", "-")
            if sf_alias != normalized_id and sf_alias not in model_aliases:
                model_aliases.append(sf_alias)

            if model_aliases:
                entry["model_aliases"] = model_aliases

            # 模态推断
            modalities = _infer_modalities(model_name, org)
            if modalities:
                entry["modalities"] = modalities

            # 能力推断
            capabilities = _infer_capabilities(model_name)
            if capabilities:
                entry["capabilities"] = capabilities

            # 保存 org 信息（用于后续匹配厂商）
            entry["_sf_org"] = org.lower()
            if prefix:
                entry["_sf_prefix"] = prefix.lower()

            all_models.append(entry)

        if not all_models:
            raise RuntimeError("No usable models found from SiliconFlow API")

        return all_models

    def scrape_prices(self) -> dict[str, dict[str, Any]]:
        """
        从硅基流动价格页面获取模型价格

        由于定价页面是 SPA，且 API 不返回价格信息，
        这里返回空字典。价格信息将通过其他渠道获取（如 OpenRouter）。
        """
        return {}
