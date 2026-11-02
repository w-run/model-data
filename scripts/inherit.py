#!/usr/bin/env python3
"""
变体继承

OpenRouter 上同一个模型有多种变体后缀，窗口与基础模型一致：

    claude-haiku-4.5        ctx=200000
    claude-haiku-4.5:batch  ctx=200000    ← 批处理只是计费/排队方式不同

这些变体在 models.dev 上没有独立条目，共识投票拿不到票，会被降级成
inferred 并回落到 OpenRouter 自报值（而该值可能就是占位填充）。

本模块在共识之后补一轮：变体自身没有可信数据时，继承基础模型的结论。
"""

from __future__ import annotations

from typing import Any

# 只继承这些变体：它们的上下文窗口与基础模型**完全一致**
# （走不同路由/计费/排队，但模型本身同一份权重）
INHERIT_SUFFIXES = (":batch", ":free", ":extended", ":nitro", ":thinking", ":floor", ":online")

# 不继承：这些是**不同模型**，窗口可能真的不同
NO_INHERIT_SUFFIXES = (":vision", ":reasoning", "-vision", "-exp", "-preview")


def base_id(model_id: str) -> str | None:
    """去掉变体后缀，返回基础模型 id。不是变体则返回 None。"""
    for suf in NO_INHERIT_SUFFIXES:
        if model_id.endswith(suf):
            return None
    for suf in INHERIT_SUFFIXES:
        if model_id.endswith(suf):
            b = model_id[: -len(suf)]
            return b or None
    return None


def apply_inheritance(models: dict[str, dict[str, Any]]) -> int:
    """
    变体继承基础模型的 limit（仅当变体自己 confidence=inferred）。

    必须在共识投票完成、且基础模型已定稿之后调用。
    返回继承数量。
    """
    inherited = 0

    # 先算好每个基础模型的结论
    bases: dict[str, dict[str, Any]] = {}
    for mid, model in models.items():
        limit = model.get("limit") or {}
        if limit.get("context") and limit.get("confidence") in ("first_party", "consensus"):
            bases[mid] = limit

    for mid, model in models.items():
        limit = model.get("limit") or {}
        if limit.get("confidence") != "inferred":
            continue

        b = base_id(mid)
        if not b:
            continue
        base_limit = bases.get(b)
        if not base_limit:
            continue

        # 继承能力上限，但保留变体自己的来源痕迹
        limit["context"] = base_limit["context"]
        limit["max_output"] = base_limit.get("max_output")
        limit["confidence"] = base_limit["confidence"]
        limit["votes"] = base_limit.get("votes", 0)
        limit["sources"] = ["inherited:" + b]
        limit.pop("openrouter_context", None)
        model["limit"] = limit

        # recommended 必须跟着重算，否则预算还按旧值
        from .limits import DEFAULT_OUTPUT_CAP, derive_recommended

        rec = derive_recommended(limit, output_cap=DEFAULT_OUTPUT_CAP)
        rec["output_cap"] = DEFAULT_OUTPUT_CAP
        model["recommended"] = rec

        inherited += 1

    return inherited
