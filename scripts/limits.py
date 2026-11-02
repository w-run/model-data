#!/usr/bin/env python3
"""
上下文 / 输出上限的解析层

问题背景
--------
OpenRouter（以及大量聚合平台）在 provider 没有上报 max_completion_tokens 时，
会用 floor(context * 0.9) 填充一个占位值。以 deepseek-v4.1-flash 为例：

    context = 1048576  ->  max_completion_tokens = 943718

这看起来像个合法数字，但它是 1048576 的 90%，不是模型真实能力。
消费方若直接拿它当 max_tokens 用，输入侧就只剩 10%，必然触发自动压缩/截断。

本模块的三件事
--------------
1. is_placeholder()   识别这类填充值
2. vote_limits()      跨 provider 共识投票，得出可信的 (context, max_output)
3. derive_recommended() 由能力上限推导「使用预算」，供客户端直接消费
"""

from __future__ import annotations

import math
from typing import Any, Iterable

# 判定 floor(0.9*ctx) 占位值的容差（token）
PLACEHOLDER_TOLERANCE = 1

# 容量等价：同一容量常有 x1000 / x1024 两种写法（384K = 384000 或 393216）。
# 见 _cluster_key()：按 K 倍数归一化，而不是按百分比容差 ——
# 百分比容差无法区分「384K 的两种写法」（差 2.4%）和「512K vs 500K」（差 4.8%）。

# 低于这个票数不足以称为共识
MIN_CONSENSUS_VOTES = 2

# 推荐输出的默认封顶（128K）。真实上限更高时也不建议一次吃满：
# 输出预算越大，留给输入的窗口越小，长上下文任务反而更容易被压缩。
DEFAULT_OUTPUT_CAP = 131072


def is_placeholder(context: int | None, output: int | None) -> bool:
    """判断 output 是否是平台填充的占位值而非真实能力。"""
    if not (context and output):
        return False
    if output >= context:
        # 输出 >= 上下文：语义上不可能（共享窗口时），视为填充
        return True
    return abs(output - math.floor(context * 0.9)) <= PLACEHOLDER_TOLERANCE


def _cluster_keys(v: int) -> set[int]:
    """
    一个容量值的所有等价「K 倍数」，**不含基数**。

    同一容量常有 x1000 / x1024 两种写法：
        393216 = 384 * 1024  →  384
        384000 = 384 * 1000  →  384        ← 等价，应并簇
    而不同容量天然得到不同倍数：
        1048576 → 1024
        1000000 → 1000                      ← 不等价，分开
        524288  → 512
        500000  → 500                       ← 不等价，分开

    坑：384000 % 1024 == 0（= 375 * 1024），所以 384000 会同时产出
    {375, 384}。若把基数也放进键，就会与 393216 的 {384(w1024)} 错过；
    不放进键才能正确合并。代价是 375000 这类真·375K 会误并到 384K，
    实际数据里没出现过，可接受。
    """
    keys: set[int] = set()
    for base in (1024, 1000):
        if v % base == 0:
            keys.add(v // base)
    return keys or {v}


def _cluster(values: Iterable[int]) -> list[list[int]]:
    """按共享 K 基数做并查集聚类。返回按最小值升序的簇列表。"""
    uniq = sorted({v for v in values if v})
    parent = {v: v for v in uniq}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen: dict[int, int] = {}
    for v in uniq:
        for k in _cluster_keys(v):
            if k in seen:
                union(seen[k], v)
            else:
                seen[k] = v

    groups: dict[int, list[int]] = {}
    for v in uniq:
        groups.setdefault(find(v), []).append(v)
    return sorted(groups.values(), key=lambda cl: min(cl))


def _mode(values: list[int]) -> int:
    """簇内代表值：取众数（票数最多的写法），平票时取最小值（保守）。"""
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    return min(v for v, c in counts.items() if c == best)


def _rank(cand: dict[str, Any], all_cands: list[dict[str, Any]]) -> tuple[int, int, int]:
    """
    候选簇的排序键（越大越优）。

    规则：
      1. 有第一方观测 → 加权 2x 票数（官方口径本质上比聚合商可信，
         但不给无限权重：第一方孤票 vs 20 家聚合共识，共识仍应赢）
      2. 其余按原始票数
    """
    votes = cand["votes"]
    weight = votes * 2 if cand.get("first_party") else votes
    return (weight, votes, 1 if cand.get("first_party") else 0)


def vote_limits(
    observations: list[dict[str, Any]],
    *,
    first_party_vendor: str | None = None,
) -> dict[str, Any]:
    """
    跨源共识投票。

    observations: [{"provider": str, "context": int|None, "output": int|None,
                    "first_party": bool}]

    返回:
      {"context": int|None, "max_output": int|None,
       "confidence": "first_party"|"consensus"|"inferred",
       "votes": int, "sources": [provider...]}
    """
    usable = [o for o in observations if o.get("context")]
    if not usable:
        return {"context": None, "max_output": None, "confidence": "inferred", "votes": 0, "sources": []}

    # ── 第一方观测：优先，但只在与大票数不矛盾时采信 ──
    fp = [o for o in usable if o.get("first_party")]

    # ── context 聚类投票 ──
    ctx_clusters = _cluster(o["context"] for o in usable)
    ctx_scored = []
    for cl in ctx_clusters:
        providers = [o for o in usable if o["context"] in cl]
        has_fp = any(o.get("first_party") for o in providers)
        ctx_scored.append({
            "values": cl,
            "votes": len(providers),
            "first_party": has_fp,
            "providers": [o["provider"] for o in providers],
        })
    ctx_scored.sort(key=lambda c: _rank(c, ctx_scored), reverse=True)
    best_ctx = ctx_scored[0]

    # 簇内代表值：有第一方取第一方众数，否则取簇内众数。
    # 不能取 min —— 簇内偶有离群低值（如 1048000），会把真实值带偏。
    fp_ctx = [o["context"] for o in fp if o["context"] in best_ctx["values"]]
    if fp_ctx:
        context = _mode(fp_ctx)
        confidence = "first_party"
    else:
        context = _mode(best_ctx["values"])
        confidence = "consensus" if best_ctx["votes"] >= MIN_CONSENSUS_VOTES else "inferred"

    # ── output：只在该 context 簇的观测里投，剔除占位值 ──
    in_ctx = [o for o in usable if o["context"] in best_ctx["values"] and o.get("output")]
    real_out = [o for o in in_ctx if not is_placeholder(o["context"], o["output"])]
    dropped = len(in_ctx) - len(real_out)

    max_output: int | None = None
    if real_out:
        out_clusters = _cluster(o["output"] for o in real_out)
        out_scored = []
        for cl in out_clusters:
            providers = [o for o in real_out if o["output"] in cl]
            out_scored.append({
                "values": cl,
                "votes": len(providers),
                "first_party": any(o.get("first_party") for o in providers),
            })
        out_scored.sort(key=lambda c: _rank(c, out_scored), reverse=True)
        best_out = out_scored[0]

        fp_out = [o["output"] for o in fp if o.get("output") in best_out["values"]
                  and not is_placeholder(o["context"], o["output"])]
        if fp_out:
            max_output = _mode(fp_out)
        else:
            max_output = _mode(best_out["values"])
            if confidence == "first_party":
                # ctx 是第一方值但 out 不是 → 整体降级
                confidence = "consensus"
    elif in_ctx:
        # 该 context 簇下所有输出值都是占位 → 真实上限未知。
        # 不能保留任何占位值：宁愿留空，也不给出一个会把输入挤死的假上限。
        max_output = None
        confidence = "inferred"

    if dropped and confidence == "first_party":
        confidence = "consensus"

    return {
        "context": context,
        "max_output": max_output,
        "confidence": confidence,
        "votes": best_ctx["votes"],
        "sources": sorted(set(best_ctx["providers"])),
        "dropped_placeholders": dropped,
    }


def derive_recommended(
    limit: dict[str, Any],
    *,
    output_cap: int = DEFAULT_OUTPUT_CAP,
    semantics: str = "shared",
) -> dict[str, Any]:
    """
    由能力上限推导使用预算。

    semantics:
      shared      context 是输入+输出共享的总窗口（OpenAI / DeepSeek / 多数厂商）
      input_only  context 只是输入上限，输出另算（少数厂商）

    约束（由 validate.py 强制校验）：
      shared:      input_budget + output_budget <= context
                   output_budget <= max_output
      input_only:  input_budget <= context
                   output_budget <= max_output
    """
    context = limit.get("context")
    max_output = limit.get("max_output")

    if not context:
        return {"input_budget": None, "output_budget": None, "source": "derived"}

    if semantics == "input_only":
        input_budget = context
        output_budget = min(max_output or output_cap, output_cap)
        return {
            "input_budget": input_budget,
            "output_budget": output_budget,
            "source": "derived",
        }

    # shared：输出预算取 min(真实上限, 封顶, 半个窗口)
    # 半个窗口的限制保证任何模型至少留 50% 给输入
    out = output_cap
    if max_output:
        out = min(out, max_output)
    out = min(out, max(context // 2, 1024))

    return {
        "input_budget": context - out,
        "output_budget": out,
        "source": "derived",
    }
