#!/usr/bin/env python3
"""
可信度引擎

设计原则
--------
不是「谁说了什么就信谁」，而是「这个值有多少**独立的、可追溯的**证据支持」。

三个必须区分的概念
------------------
1. tier        源的先天可信度（第一方官方 > 云转售 > 聚合商 > 社区）
2. independence 同一结论被多少**独立**来源印证
3. agreement   支持最佳值的权重占全部权重的比例

只在最后把它们合成一个 0..1 的 confidence，并且**同时保留完整的证据链**——
消费方需要知道「为什么是 384000」，而不只是一个分数。

聚合器陷阱
----------
40 家 provider 都报 384000，不等于 40 票：它们可能全部转述同一个上游。
所以同 tier 同值只按 sqrt(数量) 折算为「有效票数」，避免刷分。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any, Iterable


# ── 源分层 ──────────────────────────────────────────────────────────────────

TIER_WEIGHT: dict[str, float] = {
    "manual":     1.00,   # 人工标称，最终裁定
    "first_party": 0.95,  # models.dev 第一方 provider（人工核对、带出处）
    "first_party_api": 0.90,  # 厂商 API GET /models
    "official_cloud": 0.80,   # Azure / Bedrock / Vertex / 阿里云 转售
    "aggregator": 0.55,   # models.dev 聚合 provider
    "community":  0.50,   # LiteLLM 等社区维护
    "openrouter": 0.30,   # 占位值污染重，仅补覆盖
    "heuristic":  0.20,   # 推导值
}

# 出处加成：能指到具体文档 URL 的，比空口报数可信
PROVENANCE_BONUS = {
    "doc_url": 0.10,      # 注释里有官方文档链接
    "api_live": 0.08,     # 实时调用 API 得到
    "pricing_url": 0.05,  # 指向计费页面
    # 经汇率换算得到的值：不是原价，不给加成
    "converted": -0.05,
    "none": 0.0,
}

# 新鲜度：越久的观测越不可信（容量会随版本变化）
STALE_DAYS = 180
MAX_STALE_PENALTY = 0.30

# 「权威层级」：原文来源，不是转述。
# 判定必须看 tier，不能看算完的 weight —— 无观测日期的条目会被新鲜度折损
# （0.95+0.10 再乘 0.85 = 0.8925），恰好掉到阈值以下，权威就被误判成普通观测。
AUTHORITATIVE_TIERS = ("manual", "first_party", "first_party_api")

# 置信度分级阈值
LEVELS = (
    (0.90, "verified"),
    (0.70, "high"),
    (0.50, "medium"),
    (0.00, "low"),
)


@dataclass
class Observation:
    """一条观测：某个源对某个字段的报告。"""

    source: str           # 源标识，如 "models.dev:deepseek"
    tier: str             # TIER_WEIGHT 的键
    field_name: str       # "context" / "max_output" / ...
    value: Any
    provenance: str = "none"   # PROVENANCE_BONUS 的键
    url: str | None = None     # 具体出处
    observed_at: str | None = None   # ISO 日期
    note: str | None = None
    # 该观测是否可独立采信（占位值 / 明显错误 → False）
    valid: bool = True
    # 计价币种：价格类字段必填，其他字段忽略
    currency: str = "USD"

    @property
    def tier_weight(self) -> float:
        return TIER_WEIGHT.get(self.tier, 0.3)

    def weight(self, *, today: date | None = None) -> float:
        """单条观测的有效权重，0..1。"""
        if not self.valid:
            return 0.0
        w = self.tier_weight
        w += PROVENANCE_BONUS.get(self.provenance, 0.0)
        w *= self._freshness(today)
        return round(min(w, 1.0), 4)

    def _freshness(self, today: date | None) -> float:
        if not self.observed_at:
            # 无日期的观测：轻微折损，但不至于归零
            return 0.85
        try:
            d = datetime.fromisoformat(str(self.observed_at)[:10]).date()
        except ValueError:
            return 0.85
        today = today or date.today()
        age = (today - d).days
        if age <= 30:
            return 1.0
        if age >= STALE_DAYS:
            return 1.0 - MAX_STALE_PENALTY
        # 线性衰减
        return 1.0 - MAX_STALE_PENALTY * (age - 30) / (STALE_DAYS - 30)


@dataclass
class Evidence:
    """一个字段的聚合结论 + 完整证据链。"""

    field_name: str
    value: Any
    confidence: float
    level: str
    agreement: float
    n_independent: int
    total_weight: float
    supporting: list[dict] = field(default_factory=list)
    dissenting: list[dict] = field(default_factory=list)
    contested: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["supporting"] = self.supporting
        d["dissenting"] = self.dissenting
        return d


# ── 数值等价 ────────────────────────────────────────────────────────────────

def capacity_keys(v: int) -> set[int]:
    """
    容量的等价「K 倍数」集合。

    393216 (=384×1024) 与 384000 (=384×1000) 是同一个 384K → 都给出 384。
    1048576 (1024×1024) 与 1000000 (1000×1000) 是不同容量 → 1024 / 1000。

    坑：384000 % 1024 == 0（= 375×1024），因此会同时产出 {375, 384}。
    不把基数放进键里，才能与 393216 的 {384} 正确合并。
    """
    if not isinstance(v, int) or v <= 0:
        return {v} if v is not None else set()
    keys = {v // b for b in (1024, 1000) if v % b == 0}
    return keys or {v}


def cluster_by_value(values: Iterable[int]) -> list[list[int]]:
    """按等价容量并查集聚类。返回按最小值升序的簇。"""
    uniq = sorted({v for v in values if isinstance(v, int) and v > 0})
    if not uniq:
        return []
    parent = {v: v for v in uniq}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen: dict[int, int] = {}
    for v in uniq:
        for k in capacity_keys(v):
            if k in seen:
                ra, rb = find(seen[k]), find(v)
                if ra != rb:
                    parent[rb] = ra
            else:
                seen[k] = v

    groups: dict[int, list[int]] = {}
    for v in uniq:
        groups.setdefault(find(v), []).append(v)
    return sorted(groups.values(), key=lambda c: min(c))


# ── 证据聚合 ────────────────────────────────────────────────────────────────

def aggregate(observations: list[Observation], *, field_name: str,
              today: date | None = None) -> Evidence | None:
    """
    把一个字段的所有观测聚合成一个结论。

    - 数值型：按容量聚类投票
    - 其他型：按值的字面量投票
    """
    obs = [o for o in observations
           if o.field_name == field_name and o.valid and o.value is not None]
    # 数值字段剔除 0 / 负数：上游偶有 value=0 的脏数据，
    # 放任它进入聚类会让簇列表为空，进而索引越界。
    if obs and all(isinstance(o.value, int) for o in obs):
        obs = [o for o in obs if o.value > 0]
    # 该字段可能全部观测都被判无效（例如只有占位值）→ 无证据，不是错误
    if not obs:
        return None

    weights = {id(o): o.weight(today=today) for o in obs}
    total = sum(weights.values())
    if total <= 0:
        return None

    is_numeric = all(isinstance(o.value, int) for o in obs)

    if is_numeric:
        clusters = cluster_by_value(o.value for o in obs)
        scored = []
        for cl in clusters:
            members = [o for o in obs if o.value in cl]
            # 独立票：同 tier 同值只按 sqrt 折算，防聚合器刷分
            by_tier: dict[str, list[Observation]] = {}
            for o in members:
                by_tier.setdefault(o.tier, []).append(o)
            indep = sum(math.sqrt(len(v)) for v in by_tier.values())
            scored.append({
                "values": cl,
                "members": members,
                "weight": sum(weights[id(o)] for o in members),
                "indep": indep,
            })
        scored.sort(key=lambda c: (c["weight"], c["indep"]), reverse=True)
        best = scored[0]
        value = _representative(best["members"], weights)
    else:
        by_val: dict[Any, list[Observation]] = {}
        for o in obs:
            by_val.setdefault(_hashable(o.value), []).append(o)
        scored = []
        for val, members in by_val.items():
            by_tier = {}
            for o in members:
                by_tier.setdefault(o.tier, []).append(o)
            scored.append({
                "values": [val],
                "members": members,
                "weight": sum(weights[id(o)] for o in members),
                "indep": sum(math.sqrt(len(v)) for v in by_tier.values()),
            })
        scored.sort(key=lambda c: (c["weight"], c["indep"]), reverse=True)
        best = scored[0]
        value = best["members"][0].value

    support_w = best["weight"]
    agreement = support_w / total

    # 有效独立来源数（含所有簇，用于衡量整体证据面）
    n_indep = round(sum(c["indep"] for c in scored))

    # 基础分 = 支持方的平均单条权重
    base = support_w / max(len(best["members"]), 1)

    # ── 权威证据优先，不被票数稀释 ──
    # 聚合商之间大量互抄同一个上游，票数是「假独立」的。
    # 若最佳簇里有强权威（tier_weight >= 0.90，即第一方/人工），
    # 它以自身权重为准，只按独立度小幅加成，不与聚合商做加权平均。
    AUTHORITATIVE = 0.90
    auth_members = [o for o in best["members"] if weights[id(o)] >= AUTHORITATIVE]
    if auth_members:
        strongest = max(weights[id(o)] for o in auth_members)
        # 独立印证加成：还有别家也一致时再加一点，但上限 +12%
        corroboration = min(0.12, 0.02 * max(n_indep - len(auth_members), 0))
        confidence = strongest + corroboration
        # 若聚合端有强异议（簇权重不低），仍要扣分
        if any(c["weight"] >= support_w * 0.6 for c in scored if c is not best):
            confidence *= 0.90
    else:
        # 无权威证据：基础权重 × 一致度 × 独立度加成
        indep_bonus = 1.0 + 0.05 * math.log2(max(n_indep, 1))
        confidence = base * agreement * min(indep_bonus, 1.35)

    confidence = round(min(max(confidence, 0.0), 1.0), 4)

    level = next(lv for th, lv in LEVELS if confidence >= th)

    supporting, dissenting = [], []
    for c in scored:
        target = supporting if c is best else dissenting
        for o in c["members"]:
            target.append({
                "source": o.source,
                "tier": o.tier,
                "value": o.value,
                "weight": weights[id(o)],
                "provenance": o.provenance,
                "url": o.url,
                "observed_at": o.observed_at,
                "note": o.note,
            })
    supporting.sort(key=lambda x: -x["weight"])
    dissenting.sort(key=lambda x: -x["weight"])

    return Evidence(
        field_name=field_name,
        value=value,
        confidence=confidence,
        level=level,
        agreement=round(agreement, 4),
        n_independent=n_indep,
        total_weight=round(total, 4),
        supporting=supporting[:8],
        dissenting=dissenting[:8],
        contested=any(c["weight"] >= support_w * 0.5 for c in scored if c is not best),
    )


def aggregate_pair(observations: list[Observation], *, today: date | None = None) -> tuple:
    """
    同时投票 context 和 max_output，**保持同源性**。

    为什么不能用 aggregate() 各投各的
    ----------------------------------
    context 和 max_output 是两个字段，但它们必须来自**同一份声明**才有意义。
    分开投票会拼出「A 家的窗口 + B 家的输出」这种物理上不存在的组合：

        unslopnemo-12b
          models.dev/aimlapi   ctx=32768   out=32768(无效)
          openrouter           ctx=1024000 out=819200
        → ctx 取 aimlapi 的 32768，out 取 openrouter 的 819200
        → 819200 >= 32768，荒谬

    正确做法：把 (context, max_output) 当成一个整体候选，按容量聚类，
    每个簇内再定 context 与 output。
    """
    usable = [o for o in observations if o.valid and o.value is not None]
    if not usable:
        return None, None

    # 按来源归组：同一 source 的 ctx/out 视为一份声明
    by_source: dict[str, dict[str, Observation]] = {}
    for o in usable:
        if o.field_name in ("context", "max_output"):
            if isinstance(o.value, int) and o.value <= 0:
                continue
            by_source.setdefault(o.source, {})[o.field_name] = o

    if not by_source:
        return None, None

    # 每份声明一个候选：以 context 为聚类主轴
    decls = []
    for src, fields in by_source.items():
        ctx_o = fields.get("context")
        out_o = fields.get("max_output")
        ctx = ctx_o.value if ctx_o else None
        out = out_o.value if out_o else None
        if ctx is None and out is None:
            continue
        decls.append({
            "source": src,
            "context": ctx,
            "output": out,
            "ctx_obs": ctx_o,
            "out_obs": out_o,
            "weight": max((o.weight(today=today) for o in fields.values()), default=0.0),
            "tier": (ctx_o or out_o).tier,
        })

    # 聚类：有 context 的按容量簇；没有 context 的单独归池
    with_ctx = [d for d in decls if d["context"]]
    without_ctx = [d for d in decls if not d["context"]]

    clusters = cluster_by_value(d["context"] for d in with_ctx)
    scored = []
    for cl in clusters:
        members = [d for d in with_ctx if d["context"] in cl]
        by_tier: dict[str, int] = {}
        for d in members:
            by_tier[d["tier"]] = by_tier.get(d["tier"], 0) + 1
        indep = sum(math.sqrt(n) for n in by_tier.values())
        scored.append({
            "members": members,
            "weight": sum(d["weight"] for d in members),
            "indep": indep,
            "context": _weighted_mode(
                [(d["context"], d["weight"]) for d in members]),
        })
    # ── 权威优先 ──
    # 第一方条目是原文，聚合商是转述。转述再多也不能推翻原文 ——
    # 20 家聚合商一致报 1000000，常见的真实原因是它们都从官方抄来后做了
    # 人类友好化取整（1048576 → 1000000），而非各自独立观测。
    # 所以只要存在第一方簇，就取它；仅当第一方内部自相矛盾时才在组内投票。
    auth_scored = [c for c in scored
                   if any(d["tier"] in AUTHORITATIVE_TIERS for d in c["members"])]
    pool = auth_scored or scored
    pool.sort(key=lambda c: (c["weight"], c["indep"]), reverse=True)
    # 排序仍按全量，便于统计异议；但选择范围限定在 pool
    scored.sort(key=lambda c: (c["weight"], c["indep"]), reverse=True)
    best = pool[0]
    # output 只在**同一簇内**投票，天然保证 out 与 ctx 同源
    out_members = [d for d in best["members"] if d["output"]]
    max_output = None
    if out_members:
        out_clusters = cluster_by_value(d["output"] for d in out_members)
        out_scored = []
        for cl in out_clusters:
            ms = [d for d in out_members if d["output"] in cl]
            out_scored.append({
                "members": ms,
                "weight": sum(d["weight"] for d in ms),
                "value": _weighted_mode([(d["output"], d["weight"]) for d in ms]),
            })
        out_scored.sort(key=lambda c: c["weight"], reverse=True)
        max_output = out_scored[0]["value"]

    ctx_ev = _build_evidence(scored, "context", today=today, best=best)
    if ctx_ev:
        ctx_ev.value = best["context"]

    out_scored_all = []
    for c in scored:
        ms = [d for d in c["members"] if d["output"]]
        if not ms:
            continue
        out_clusters = cluster_by_value(d["output"] for d in ms)
        best_o = None
        for cl in out_clusters:
            grp = [d for d in ms if d["output"] in cl]
            cand = {
                "members": grp,
                "weight": sum(d["weight"] for d in grp),
                "indep": c["indep"],
                "value": _weighted_mode([(d["output"], d["weight"]) for d in grp]),
            }
            if best_o is None or cand["weight"] > best_o["weight"]:
                best_o = cand
        if best_o:
            out_scored_all.append(best_o)

    out_best = max(out_scored_all, key=lambda c: (c["weight"], c["indep"])) if out_scored_all else None
    out_ev = _build_evidence(out_scored_all, "max_output", today=today, best=out_best)
    if out_ev:
        out_ev.value = max_output

    return ctx_ev, out_ev


def _weighted_mode(pairs: list[tuple[int, float]]) -> int:
    tally: dict[int, float] = {}
    for v, w in pairs:
        tally[v] = tally.get(v, 0.0) + w
    best = max(tally.values())
    return min(v for v, w in tally.items() if abs(w - best) < 1e-9)


def _build_evidence(scored: list, field_name: str, *, today=None, best=None):
    """
    把配对投票的结果包装成 Evidence（用于 confidence 与证据链）。

    best 必须由调用方给出：选择规则（权威优先）与排序规则（票数优先）
    不同，让本函数自己按票数重选，会出现「值是权威的、置信度却是聚合商的」。
    """
    total = sum(c["weight"] for c in scored)
    if not scored or total <= 0:
        return None
    if best is None:
        best = max(scored, key=lambda c: (c["weight"], c["indep"]))
    support_w = best["weight"]
    agreement = support_w / total
    n_indep = round(sum(c["indep"] for c in scored))
    members = best["members"]
    base = support_w / max(len(members), 1)

    auth = [d for d in members if d["tier"] in AUTHORITATIVE_TIERS]
    if auth:
        # 置信度锚定层级，而非折损后的权重：
        # 权威条目即使观测日期久远（新鲜度 0.85），它的「原文可信」属性不变。
        # 新鲜度只用于在多个权威来源之间排序，不用于削弱整体置信度。
        strongest = max(TIER_WEIGHT.get(d["tier"], 0.5) for d in auth)
        confidence = strongest + min(0.12, 0.02 * max(n_indep - len(auth), 0))
        # 异议扣分，但要区分「真异议」和「取整差异」：
        # 聚合商普遍把 1048576 抄成 1000000，这不是矛盾，是同一个值的人类写法。
        support_val = best.get("context") or best.get("value")
        real_dissent = False
        for c in scored:
            if c is best:
                continue
            if c["weight"] < support_w * 0.6:
                continue
            other_val = c.get("context") or c.get("value")
            if support_val and isinstance(other_val, int) \
               and capacity_keys(support_val) & capacity_keys(other_val):
                continue        # 等价容量，不算异议
            real_dissent = True
            break
        if real_dissent:
            confidence *= 0.90
    else:
        confidence = base * agreement * min(1.0 + 0.05 * math.log2(max(n_indep, 1)), 1.35)
    confidence = round(min(max(confidence, 0.0), 1.0), 4)
    level = next(lv for th, lv in LEVELS if confidence >= th)

    def pack(d: dict) -> dict:
        o = d.get("ctx_obs") if field_name == "context" else d.get("out_obs")
        return {
            "source": d["source"],
            "tier": d["tier"],
            "value": d["context"] if field_name == "context" else d["output"],
            "weight": round(d["weight"], 4),
            "provenance": o.provenance if o else "none",
            "url": o.url if o else None,
            "observed_at": o.observed_at if o else None,
            "note": o.note if o else None,
        }

    supporting = sorted((pack(d) for d in members), key=lambda x: -x["weight"])[:8]
    dissenting = sorted(
        (pack(d) for c in scored if c is not best for d in c["members"]),
        key=lambda x: -x["weight"])[:8]

    return Evidence(
        field_name=field_name,
        value=best.get("value") or best.get("context"),
        confidence=confidence,
        level=level,
        agreement=round(agreement, 4),
        n_independent=n_indep,
        total_weight=round(total, 4),
        supporting=supporting,
        dissenting=dissenting,
        contested=any(c["weight"] >= support_w * 0.5 for c in scored if c is not best),
    )


def _representative(members: list[Observation], weights: dict[int, float]) -> int:
    """簇内代表值：取权重加权众数，平票取小（保守）。"""
    tally: dict[int, float] = {}
    for o in members:
        tally[o.value] = tally.get(o.value, 0.0) + weights[id(o)]
    best = max(tally.values())
    return min(v for v, w in tally.items() if abs(w - best) < 1e-9)


def _hashable(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return v
