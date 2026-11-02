#!/usr/bin/env python3
"""
完整计费模型

除了「输入 / 输出」两个基础价，真实计费还有三类必须建模：

1. 缓存计费（cache read / cache write）
   命中已有的 prompt 缓存 vs 写入新缓存，价格差异可达 10 倍。
       Anthropic: input 3.00 / cache_read 0.30 / cache_write 3.75
   不建模会导致「长对话成本」估算偏低数倍。

2. 阶梯计费（tiered by context）
   同样的模型，上下文越长单价越贵：
       xai/grok-4.3  ≤200K: 1.25/2.5   >200K: 2.5/5.0
   用基础价估长上下文请求会低估一半。

3. 时段计费（time-based multiplier）
   国内厂商常见，按时段打折或加价：
       DeepSeek  高峰期（01:00-04:00, 06:00-10:00 UTC 工作日）双倍
       models.dev 里记为 [[cost.multipliers]] factor = 2

数值口径
--------
所有价格统一为「每百万 token」，与既有产出的 USD/Mtok、CNY/Mtok 一致。
"""

from __future__ import annotations

from typing import Any

# ── 1. 缓存 ─────────────────────────────────────────────────────────────────

CACHE_FIELDS = {
    "cache_read": "命中缓存的输入",
    "cache_write": "写入缓存",
}


def build_cache_block(rows: list) -> dict[str, float] | None:
    """
    从观测里取缓存价。取不到就返回 None —— 不猜。

    cache_write 常为 0（表示不额外收费），与「缺失」不同，
    所以只接受明确出现的数值。
    """
    out: dict[str, float] = {}
    for field, key in (("price_cache_read", "read"), ("price_cache_write", "write")):
        cands = [o for o in rows
                 if o.field_name == field and o.valid
                 and isinstance(o.value, (int, float)) and o.value >= 0]
        if not cands:
            continue
        cands.sort(key=lambda o: -o.weight())
        best = cands[0]
        out[key] = round(float(best.value), 6)
        out[f"{key}_currency"] = best.currency
    return out or None


# ── 2. 阶梯 ─────────────────────────────────────────────────────────────────

def build_tiers(rows: list) -> list[dict[str, Any]] | None:
    """
    阶梯价。观测里的 `tier` 字段携带触发条件：

        {"input": 4, "output": 18, "tier": {"type": "context", "size": 200000}}

    语义：**超过** size 之后按这组价格计。
    返回按 size 升序排列，便于消费方二分查找。
    """
    tiers: list[dict[str, Any]] = []
    for o in rows:
        if o.field_name != "price_tier" or not o.valid:
            continue
        val = o.value
        if not isinstance(val, dict):
            continue
        cond = val.get("tier") or {}
        size = cond.get("size")
        if not isinstance(size, int) or size <= 0:
            continue
        entry = {
            "over": size,
            "type": cond.get("type") or "context",
            "input": round(float(val.get("input") or 0), 6),
            "output": round(float(val.get("output") or 0), 6),
            "currency": o.currency,
        }
        for key in ("cache_read", "cache_write"):
            if isinstance(val.get(key), (int, float)):
                entry[key] = round(float(val[key]), 6)
        tiers.append(entry)

    if not tiers:
        return None
    # 去重：同一 over 值只留一条
    seen: dict[int, dict] = {}
    for t in sorted(tiers, key=lambda x: x["over"]):
        seen.setdefault(t["over"], t)
    return list(seen.values())


# ── 3. 时段 ─────────────────────────────────────────────────────────────────

def build_multipliers(rows: list) -> list[dict[str, Any]] | None:
    """
    时段倍率。国内厂商的错峰定价：

        DeepSeek 高峰 01:00-04:00 / 06:00-10:00 UTC 工作日，factor = 2

    观测里存的是原始 when 条件与 factor，原样透出 —— 条件表达式很复杂，
    在信息库里做求值没有意义，交给消费方按自己的时区解释。
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for o in rows:
        if o.field_name != "price_multiplier" or not o.valid:
            continue
        val = o.value
        if not isinstance(val, dict):
            continue
        factor = val.get("factor")
        if not isinstance(factor, (int, float)) or factor <= 0:
            continue
        entry = {
            "factor": round(float(factor), 4),
            "when": val.get("when"),
            "currency": o.currency,
        }
        # 多个 provider 转述同一份厂商政策时会出现完全相同的条目，去重
        key = repr(sorted(entry.items(), key=lambda kv: kv[0]))
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out or None


def build_billing(rows: list) -> dict[str, Any] | None:
    """
    汇总成一个 billing 块，只在有额外计费维度时出现。

    结构与既有 `price` 并存，不替换它：
        price:   基础价（保持既有格式）
        billing: 缓存 / 阶梯 / 时段
    """
    block: dict[str, Any] = {}

    cache = build_cache_block(rows)
    if cache:
        block["cache"] = cache

    tiers = build_tiers(rows)
    if tiers:
        block["tiers"] = tiers

    mults = build_multipliers(rows)
    if mults:
        block["multipliers"] = mults

    return block or None
