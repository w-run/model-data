#!/usr/bin/env python3
"""
Hugging Face Inference Providers 目录

https://router.huggingface.co/v1/models

结构特点
--------
每个模型带 `providers` 数组，列出**该模型在各家推理提供商处**的实测参数：

    deepseek-ai/DeepSeek-V4.1-Flash-1017
      providers:
        novita    ctx=1048576  $0.3/$1.2  ttft=1214ms  tps=75.7   author=false
        baseten   ctx=1048576  $0.3/$1.2  ttft=207ms   tps=128.6  author=false

三个别的源都没有的东西
----------------------
1. `is_model_author`  明确标注「这家就是模型作者」，比从目录推断可靠
2. `first_token_latency_ms` / `throughput`  性能指标
3. `status`  live / staging，可判断路由是否还在服务

一个坑
------
`is_model_author=true` 的条目**全部缺 context_length**（实测 0/33）——
作者方的条目是「有这个模型」的声明，不是容量声明。
所以不能拿它当容量证据，只能用于判定归属和标记作者直供。

容量观测主要来自非作者 provider（277 条里 169 条有 ctx）。
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from ..trust.engine import Observation

URL = "https://router.huggingface.co/v1/models"
UA = "w-run-model-data/4.0 (+https://github.com/w-run/model-data)"
CACHE = Path(os.environ.get("MODEL_DATA_CACHE", Path.home() / ".cache" / "model-data"))

# 这些 org 在 HF 上自营推理端点（作者直供）
AUTHOR_RUN_ORGS = {"deepseek-ai", "zai-org", "qwen", "moonshotai", "minimaxai",
                   "coherelabs", "google", "meta-llama", "xiaomimimo"}


def fetch(offline: bool = False, timeout: int = 60) -> dict:
    cached = CACHE / "hf_router.json"
    if offline and cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception:
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        raise


def _slug(model_id: str) -> str:
    """deepseek-ai/DeepSeek-V4.1-Flash-1017 -> DeepSeek-V4.1-Flash-1017"""
    return str(model_id).split("/")[-1]


def observations(data: dict[str, Any]) -> dict[str, list[Observation]]:
    """
    展开为 {slug: [Observation, ...]}。

    每个 provider 条目算一条独立观测 —— 它们在 HF 的语义里确实是
    各平台自己上报的（字段齐全度参差不齐也印证了这点），
    所以聚类投票能正常工作。
    """
    out: dict[str, list[Observation]] = {}

    for m in data.get("data", []) or []:
        mid = str(m.get("id") or "")
        if not mid:
            continue
        slug = _slug(mid)
        owner = str(m.get("owned_by") or "").strip()

        # 归属厂商：HF 的 owned_by 就是发布方（deepseek-ai / zai-org / Qwen）
        if owner:
            out.setdefault(slug, []).append(Observation(
                source=f"hf:{owner}",
                tier="first_party" if owner.lower() in AUTHOR_RUN_ORGS else "aggregator",
                field_name="vendor",
                value=owner.lower(),
                provenance="doc_url",
                url=URL,
                observed_at=None,
                note="hf owned_by",
                valid=True,
            ))

        arch = m.get("architecture") or {}
        mods = arch.get("input_modalities")
        if isinstance(mods, list) and mods:
            out.setdefault(slug, []).append(Observation(
                source="hf",
                tier="aggregator",
                field_name="input_modalities",
                value=tuple(mods),
                provenance="doc_url",
                url=URL,
                observed_at=None,
                valid=True,
            ))

        for p in m.get("providers") or []:
            prov = str(p.get("provider") or "?")
            is_author = bool(p.get("is_model_author"))
            tier = "first_party" if is_author else "aggregator"

            ctx = p.get("context_length")
            if isinstance(ctx, int) and ctx > 0:
                out.setdefault(slug, []).append(Observation(
                    source=f"hf:{prov}",
                    tier=tier,
                    field_name="context",
                    value=ctx,
                    provenance="api_live",
                    url=URL,
                    observed_at=None,
                    note="model author" if is_author else None,
                    valid=True,
                ))

            pricing = p.get("pricing") or {}
            for key, field_name in (("input", "price_input"), ("output", "price_output")):
                val = pricing.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    out.setdefault(slug, []).append(Observation(
                        source=f"hf:{prov}",
                        tier=tier,
                        field_name=field_name,
                        value=round(float(val), 6),
                        provenance="api_live",
                        url=URL,
                        observed_at=None,
                        valid=True,
                    ))

            # 性能指标：不属于容量字段，但别的源都没有，单独保留
            ttft = p.get("first_token_latency_ms")
            if isinstance(ttft, (int, float)) and ttft > 0:
                out.setdefault(slug, []).append(Observation(
                    source=f"hf:{prov}", tier=tier, field_name="latency_ms",
                    value=round(float(ttft), 1), provenance="api_live",
                    url=URL, observed_at=None, valid=True,
                ))
            tps = p.get("throughput")
            if isinstance(tps, (int, float)) and tps > 0:
                out.setdefault(slug, []).append(Observation(
                    source=f"hf:{prov}", tier=tier, field_name="throughput_tps",
                    value=round(float(tps), 1), provenance="api_live",
                    url=URL, observed_at=None, valid=True,
                ))

    return out
