#!/usr/bin/env python3
"""
mimi.wrun.cc 数据源

它是什么
--------
用户自建的模型聚合网关（Agent2API 反代中转）。性质上分两半：

1. **自有模型**（`mimi-*` / `hy3` 等）
   只有这里有，是唯一来源 → first_party，权重最高

2. **转售的第三方模型**（`deepseek-v4.1-flash` / `glm-5.3` / `kimi-k3` …）
   可能被中转层**降配**：
       deepseek-v4.1-flash  第一方 1048576/393216
       mimi 上实际只能 1000000/128000     ← 输出砍到 1/3

   所以它的值不能拿去「投票决定模型能力」（会把能力拉低），
   但它是**用户真实调用时会遇到的限制**，必须作为「保守上限」参与决策。

重要：mimi 的声明值**可能报低**
-------------------------------
中转层的模型条目常滞后于厂商，或按自身资源配置保守声明。实测：

    catpaw 的 deepseek-v4.1-flash   声明 600000/384000
    官方真值                        1048576/393216     ← 实际可用

所以 mimi 的值**不能当硬上限**（按它取小会把 1M 压成 600k），
只能作为一条**普通观测**参与投票：够权威就采信，是孤证就不采信。

它在两个地方仍不可替代：
  1. `mimi-*` 自有模型 —— 唯一来源，tier=first_party
  2. 同一 id 多个上游条目 —— 条目间取**最小**，这是真正的"保守值"
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from ..trust.engine import Observation
from .collect import placeholder_reason

DEFAULT_URL = "https://mimi.wrun.cc/v1/models"
UA = "w-run-model-data/4.0 (+https://github.com/w-run/model-data)"

# 自有模型的 id 前缀（这些是 mimi 独有，不在别处出现）
OWN_PREFIXES = ("mimi-",)


def fetch(url: str | None = None, api_key: str | None = None, timeout: int = 30) -> dict[str, Any]:
    url = url or os.environ.get("MIMI_MODELS_URL", DEFAULT_URL)
    key = api_key or os.environ.get("MIMI_API_KEY")
    headers = {"User-Agent": UA, "Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_own_model(model_id: str) -> bool:
    return str(model_id).lower().startswith(OWN_PREFIXES)


def observations(data: dict[str, Any]) -> tuple[dict[str, list[Observation]], dict[str, dict[str, Any]]]:
    """
    返回:
      {slug: [Observation, ...]}        所有观测
      {slug: {"context": int, "output": int}}   仅 serving 约束（用于取上限）
    """
    obs: dict[str, list[Observation]] = {}
    # 同一 id 可能对应多个上游条目（不同供应商接入同一模型）。按保守值收口：
    # 只有这些条目**全部**能支持的上限才算数，避免消费方按最大值传参被拒。
    declared: dict[str, dict[str, Any]] = {}

    for m in data.get("data", []) or []:
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue

        own = is_own_model(mid)
        tier = "first_party" if own else "aggregator"

        ctx = m.get("max_input_tokens")
        outp = m.get("max_output_tokens")

        # mimi 不给 —— 缺失就是缺失，不猜
        bad = placeholder_reason(ctx, outp)

        if isinstance(ctx, int) and ctx > 0:
            obs.setdefault(mid, []).append(Observation(
                source=f"mimi:{m.get('owned_by') or '?'}",
                tier=tier,
                field_name="context",
                value=ctx,
                provenance="api_live",
                url="https://mimi.wrun.cc/v1/models",
                observed_at=None,
                note="own model" if own else "resold, may be downscaled",
                valid=True,
            ))

        if isinstance(outp, int) and outp > 0:
            obs.setdefault(mid, []).append(Observation(
                source=f"mimi:{m.get('owned_by') or '?'}",
                tier=tier,
                field_name="max_output",
                value=outp,
                provenance="api_live",
                url="https://mimi.wrun.cc/v1/models",
                observed_at=None,
                note="own model" if own else (
                    f"resold, may be downscaled; REJECTED: {bad}" if bad else "resold, may be downscaled"),
                valid=bad is None,
            ))

        # 同一 id 可能对应多个上游条目。
        # 自有模型（mimi-* 等）没有外部权威可参照，条目间取**最小**保守值；
        # 转售模型以后由权威源裁决，这里只原样记录，不做收口。
        slot = declared.setdefault(mid, {"entries": 0, "own": own, "values": []})
        if isinstance(ctx, int) and ctx > 0:
            slot["values"].append(("context", ctx))
        if isinstance(outp, int) and outp > 0 and bad is None:
            slot["values"].append(("output", outp))
        slot["entries"] += 1

    # 收口：仅自有模型取保守值（min）
    for mid, slot in declared.items():
        picked: dict[str, Any] = {"entries": slot["entries"], "own": slot["own"]}
        for field_name in ("context", "output"):
            vals = [v for f, v in slot["values"] if f == field_name]
            if vals:
                picked[field_name] = min(vals) if slot["own"] else max(vals)
        declared[mid] = picked
    return obs, declared


def extra_fields(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    mimi 独有的元信息（能力标志、计费倍数），别的源没有。
    返回 {slug: {...}}，由装配层并入模型条目。
    """
    out: dict[str, dict[str, Any]] = {}
    for m in data.get("data", []) or []:
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        mods_in = m.get("input_modalities")
        if not isinstance(mods_in, list) or not mods_in:
            mods_in = ["text"]
            if m.get("supports_images"):
                mods_in.append("image")
            if m.get("supports_video"):
                mods_in.append("video")

        credits = m.get("credits")
        mult = None
        if isinstance(credits, str):
            try:
                mult = float(credits.replace("x", "").strip())
            except ValueError:
                mult = None

        out[mid] = {
            "owned_by": m.get("owned_by"),
            "kind": m.get("kind") or "chat",
            "input_modalities": mods_in,
            "supports_tool_call": bool(m.get("supports_tool_call")),
            "supports_reasoning": bool(m.get("supports_reasoning")),
            "credits_multiplier": mult,
        }
    return out
