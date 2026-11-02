#!/usr/bin/env python3
"""
models.dev 数据源

https://models.dev/api.json

结构：{provider_id: {"models": {model_id: {...}}}}
每个 provider 条目可能带 canonical_model_id，指向该模型的标准 slug。

本模块只负责「观测」：把每个 provider 对某个 slug 的 context/output 记一条，
交给 limits.vote_limits() 做共识投票。不在这一层做任何取舍。
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

URL = "https://models.dev/api.json"
UA = "w-run-model-data/3.0 (+https://github.com/w-run/model-data)"

# 真正研发模型的厂商 provider key（第一方）。
# 与 data/vendors.toml 的键保持一致；由 build.py 在运行时传入覆盖。
DEFAULT_FIRST_PARTY = {
    "openai", "anthropic", "google", "meta", "deepseek", "alibaba", "mistral",
    "cohere", "xai", "nvidia", "minimax", "moonshotai", "zhipuai", "xiaomi",
    "tencent", "perplexity", "stepfun", "bytedance", "baidu", "sarvam",
    # models.dev 上这几个是官方托管平台的 key 变体
    "alibaba-cn", "google-vertex", "azure", "azure-cognitive-services",
    "amazon-bedrock", "google-vertex-anthropic", "aws-bedrock",
}


def fetch(timeout: int = 60) -> dict[str, Any]:
    req = urllib.request.Request(URL, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def observations(data: dict[str, Any], first_party: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """
    展开为 {canonical_slug: [observation, ...]}

    observation = {
      provider, first_party, context, output,
      cost, release_date, knowledge, modalities, reasoning, open_weights, name,
      doc_id   # 该 provider 下的原始 model id，用于回溯
    }

    canonical slug 的确定顺序：
      1. 条目的 canonical_model_id（去掉 provider 前缀后）
      2. 否则用 model id 本身（去掉 org/ 前缀）
    """
    fp = first_party if first_party is not None else DEFAULT_FIRST_PARTY
    out: dict[str, list[dict[str, Any]]] = {}

    for provider, pdata in (data or {}).items():
        for doc_id, model in ((pdata or {}).get("models") or {}).items():
            if not isinstance(model, dict):
                continue

            canonical = model.get("canonical_model_id") or model.get("id") or doc_id
            slug = str(canonical).split("/")[-1]
            if not slug:
                continue

            limit = model.get("limit") or {}
            out.setdefault(slug, []).append({
                "provider": provider,
                "first_party": provider in fp,
                "context": limit.get("context"),
                "output": limit.get("output"),
                "cost": model.get("cost"),
                "release_date": model.get("release_date"),
                "knowledge": model.get("knowledge"),
                "modalities": model.get("modalities"),
                "reasoning": model.get("reasoning"),
                "open_weights": model.get("open_weights"),
                "name": model.get("name"),
                "doc_id": doc_id,
            })

    return out


def load_first_party_from_vendors(vendors_toml_path: str) -> set[str]:
    """从 data/vendors.toml 读出厂商键，作为第一方 provider 白名单。"""
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11
        import tomli as tomllib  # type: ignore

    with open(vendors_toml_path, "rb") as f:
        data = tomllib.load(f)
    return set(data.keys()) | DEFAULT_FIRST_PARTY
