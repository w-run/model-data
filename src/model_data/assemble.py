#!/usr/bin/env python3
"""
装配层：证据 → 产出

格式约束（硬性）
----------------
系统已经对接了既有格式，**不得改动**：

    {
      "version": "2.0.0",
      "generated_at": "...",
      "count": 430,
      "models": [
        {
          "id": str, "vendor": str, "name": str, "aliases": [str],
          "context": int|null, "max_output": int|null,
          "modalities": {"input": [...], "output": [...]},
          "capabilities": {...},
          "price": {"unit", "input", "output", ...}|null,
          "kind": str, "description": str|null,
          "sources": [str]
        }
      ]
    }

可信度信息**只做增量**：新增 `confidence` 字段，不改动任何已有字段的语义与类型。
消费方不认识它也不会出错。

保守取值
--------
context / max_output 取「能力」与「serving（经 mimi 实际可用）」的较小值。
模型客观支持 1048576，但经 mimi 只给 1000000，就写 1000000 —— 消费方按它传参不会踩坑。
"""

from __future__ import annotations

import functools
import re

from datetime import datetime, timezone
from typing import Any

from . import (attribution, billing, currency, family, filtering,
               identity, idnorm, notation, official_rules, vendors)
from .trust.engine import Evidence, aggregate, aggregate_pair

# 产出顶层版本号：保持 2.0.0，因为结构与既有对接一致（仅新增可选字段）
OUTPUT_VERSION = "2.0.0"

USD_UNIT = "USD/Mtok"
USD_MTOK = "USD/Mtok"

# 过滤结果（供 CLI 汇总输出）
assemble_stats: dict[str, Any] = {}


def canonical_output_id(model_id: str) -> str:
    """
    产出的模型 id（族键与条目 id 必须共用这一个函数，否则会出现
    claude-opus-4-7 与 claude-opus-4.7 并存这样的重复）。

    Claude 按官方格式：claude-{name}-{major}[-{minor}]
    其余按通用规则：小写、版本小数点、剥离渠道前缀。
    """
    # 先剥离区域/渠道/厂商前缀（apac-anthropic-claude-opus-4-7 → claude-opus-4-7），
    # 否则 parse_claude 认不出，会掉进通用的「横线转小数点」规则，
    # 产生 claude-opus-4.7 这种与官方格式不符的 id。
    stripped = official_rules.strip_cloud_prefix(str(model_id or "").lower())
    k = identity.norm_key(stripped)
    claude = official_rules.canonical_claude_id(k)
    if claude:
        return claude
    return idnorm.canonical_id(model_id)


# 别名属同一模型的校验：去掉日期快照与渠道/区域标记后必须一致。
# 用于拦截 canonical_slug 的过度关联。
_VARIANT_TAIL = __import__("re").compile(
    r"(:[a-z0-9_-]+|@[a-z0-9]+|-spot|-flex|-priority|-free)$", 2)


@functools.lru_cache(maxsize=8192)
def _norm_family(x: str) -> str:
    """
    家族键：剥渠道前缀 → 归一写法 → 去尾部日期（含短日期与 -latest 兜底）。

    带缓存 —— 否则每个家族都要重算，实测把 03 阶段从 30 秒拖到 97 秒。
    """
    return family.family_of_with_short_date(family.normalize_spelling(x))


def _compatible(ma: dict, mb: dict) -> tuple[bool, str]:
    """
    两个模型条目是否真的是同一个模型。

    名字相近不足以判定 —— 必须有证据支持：
      - 容量冲突（两边都有值且不同）→ 不同模型
      - 发布日期冲突                  → 不同模型
      - 容量完全一致                  → 强证据，可直接合并
    例如 claude-sonnet-4-20250514 是 200K/64K，而 claude-sonnet-4 是 1M，
    它们是两条不同记录，不能互相作为别名。
    """
    ac, bc = ma.get("context"), mb.get("context")
    ao, bo = ma.get("max_output"), mb.get("max_output")
    if ac and bc and ac != bc:
        return False, "context 不同"
    if ao and bo and ao != bo:
        return False, "max_output 不同"
    ad, bd = ma.get("release_date"), mb.get("release_date")
    if ad and bd and ad != bd:
        return False, "发布日期不同"
    return True, ""


def _same_family(a: str, b: str, byid: dict | None = None) -> bool:
    """两个写法是否确定指向同一模型：名称同族 + 参数不冲突。"""
    if _norm_family(a) != _norm_family(b):
        return False
    if byid and a in byid and b in byid:
        ok, _ = _compatible(byid[a], byid[b])
        return ok
    return True


def pick(evidence: Evidence | None, declared: dict[str, Any] | None, field: str) -> tuple[Any, dict]:
    """
    取一个字段的最终值。

    三类模型，三种规则
    ------------------
    1. mimi 自有模型（mimi-* 等，declared["own"]=True）
       别处根本没有，也没有外部权威可参照。它就是唯一来源，
       同 id 多条时已在 sources/mimi.py 取过 min（保守值）。

    2. 有权威/共识证据的转售模型
       以证据为准。mimi 的声明值只作参证 —— 中转层常滞后或保守声明，
       按它取小会把官方 1M 压成 600k。

    3. 没有任何证据覆盖的 mimi 特有模型
       没有可参照的权威，只能保守：直接采信 mimi 声明值（已取 min）。
    """
    key = "output" if field == "max_output" else "context"
    cap = evidence.value if evidence else None
    decl = (declared or {}).get(key)
    is_own = bool((declared or {}).get("own"))

    # ① 自有模型 / 无任何外部证据 → 保守采信 mimi
    if is_own or (cap is None and decl is not None):
        if decl is None:
            if cap is None:
                return None, {"level": "unknown", "confidence": 0.0, "source": "none"}
            return cap, {"level": evidence.level, "confidence": evidence.confidence,
                         "source": "consensus"}
        return decl, {
            "level": "own" if is_own else "declared",
            "confidence": 0.70 if is_own else 0.45,
            "source": "mimi",
            "note": "自有模型，mimi 为唯一来源" if is_own else "仅 mimi 声明，未获其他源印证",
        }

    if cap is None:
        return None, {"level": "unknown", "confidence": 0.0, "source": "none"}

    # ② 有证据 → 以证据为准，mimi 声明仅留痕
    out = {"level": evidence.level, "confidence": evidence.confidence, "source": "consensus"}
    if decl is not None and decl != cap:
        out["mimi_declared"] = decl
    return cap, out


def infer_vendor(model_id: str, rows: list, extra: dict | None) -> str:
    """（model_id 已由调用方规范化为统一口径）"""
    """
    模型归属厂商。

    首选 models.dev 的 org 信息（发布方，不是转售方），
    其次看第一方 provider 是否与 slug 前缀一致，最后才留 unknown。
    """
    # 先用 id 前缀推断（它能识别「谁发布了这个模型」），再用 org 观测交叉校验
    known = set(vendors.BUILTIN_ALIASES)
    guessed = attribution.infer_vendor(model_id, known)
    guessed = vendors.canonical_vendor(guessed) if guessed else None

    # 1) models.dev 的 org 观测（指向发布方）
    vend = aggregate(rows, field_name="vendor")
    observed = vendors.canonical_vendor(str(vend.value)) if (vend and vend.value) else None

    # 冲突时以前缀推断为准：org 可能是转售渠道的目录
    # （providers/alibaba/models/deepseek/... 的 provider 是 alibaba-cn）
    if guessed and observed and guessed != observed:
        return guessed
    if observed:
        return observed
    if guessed:
        return guessed

    # 2) mimi 自有模型
    if extra and extra.get("owned_by") and str(model_id).lower().startswith("mimi-"):
        return "w-run"

    # 3) slug 前缀与第一方 provider 一致
    slug_head = model_id.split("-")[0].split(".")[0].lower()
    for o in rows:
        if o.tier == "first_party" and ":" in o.source:
            prov = o.source.split(":", 1)[1]
            if prov.split("-")[0] == slug_head:
                return vendors.canonical_vendor(prov)

    # 4) 从 id 推断（剥区域/渠道前缀后按厂商名前缀匹配）
    known = set(vendors.BUILTIN_ALIASES)
    guessed = attribution.infer_vendor(model_id, known)
    if guessed:
        # 推断出的可能是别名（moonshot / qwen），同样要归一到 canonical
        return vendors.canonical_vendor(guessed)

    return "unknown"


def _best_text(rows: list, field: str) -> str | None:
    """文本字段（name/description）：按权重重取值最长的那个，够用了。"""
    ev = aggregate(rows, field_name=field)
    if ev and ev.value:
        return str(ev.value)
    # 回落到权重最高的非空值
    cands = [o for o in rows if o.field_name == field and o.valid and o.value]
    if not cands:
        return None
    cands.sort(key=lambda o: -o.weight())
    return str(cands[0].value)


def _price(rows: list) -> tuple[dict | None, dict | None]:
    """
    价格：**优先人民币**。

    返回 (主字段, 美元参考值)。

    主字段策略：
      1. 有人民币渠道的原始报价 → unit=CNY/Mtok，直接用原值（无汇率误差）
      2. 只有美元报价           → unit=USD/Mtok（保持既有格式）

    两者都有时，美元值单独放进 price_usd 供参考，不参与主字段。

    为什么不能统一换算成美元：人民币报价是厂商的干净整数（8 元/M），
    换算成美元会变成 1.19516 这种无法还原的尾巴；再乘回去也回不到 8。
    """
    # 按币种分组聚合
    by_cur: dict[str, dict[str, Any]] = {}
    for cur in ("CNY", "USD"):
        bucket: dict[str, Any] = {}
        for field, key in (("price_input", "input"),
                           ("price_output", "output"),
                           ("price_cache_read", "cache_read")):
            cands = [o for o in rows if o.field_name == field and o.valid
                     and isinstance(o.value, (int, float)) and o.value > 0
                     and o.currency == cur]
            if not cands:
                continue
            # 同一币种内按可信度取：权重最高者
            cands.sort(key=lambda o: -o.weight())
            bucket[key] = round(float(cands[0].value), 6)
        if bucket:
            by_cur[cur] = bucket

    if not by_cur:
        return None, None

    if "CNY" in by_cur:
        main = {"unit": "CNY/Mtok", **by_cur["CNY"]}
        usd_ref = None
        if "USD" in by_cur:
            usd_ref = {"unit": "USD/Mtok", **by_cur["USD"]}
        return main, usd_ref

    return {"unit": "USD/Mtok", **by_cur["USD"]}, None


@functools.lru_cache(maxsize=16384)
def _decide_cached(model_id: str, vendor: str, name: str, kind: str,
                   allow_key: tuple[str, ...]) -> tuple[bool, str | None]:
    return filtering.decide(
        model_id=model_id, vendor=vendor, name=name, kind=kind,
        allowlist=set(allow_key))


def build_model(
    model_id: str,
    *,
    ctx_ev: Evidence | None,
    out_ev: Evidence | None,
    declared: dict[str, Any] | None,
    meta: dict[str, Any] | None,
    extra: dict[str, Any] | None,
    rows: list | None = None,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    context, ctx_conf = pick(ctx_ev, declared, "context")
    max_output, out_conf = pick(out_ev, declared, "max_output")

    conf = {
        "context": ctx_conf,
        "max_output": out_conf,
    }
    # 总体分：取两者较低者，缺失字段不算
    scores = [c["confidence"] for c in conf.values() if c["confidence"] > 0]
    conf["overall"] = round(min(scores), 4) if scores else 0.0
    conf["level"] = (
        "verified" if conf["overall"] >= 0.90 else
        "high" if conf["overall"] >= 0.70 else
        "medium" if conf["overall"] >= 0.50 else
        "low"
    ) if scores else "unknown"

    # 证据摘要（不含全量，保持文件不至于过大）
    if ctx_ev:
        conf["context"]["agreement"] = ctx_ev.agreement
        conf["context"]["n_sources"] = ctx_ev.n_independent
    if out_ev:
        conf["max_output"]["agreement"] = out_ev.agreement
        conf["max_output"]["n_sources"] = out_ev.n_independent

    e = extra or {}
    rows = rows or []
    name = _best_text(rows, "name")
    description = _best_text(rows, "description")
    release_date = _best_text(rows, "release_date")
    price, price_usd = _price(rows)
    billing_block = billing.build_billing(rows)
    sources = sorted({o.source.split(":")[0] for o in rows}) or []
    modalities_in = e.get("input_modalities") or ["text"]
    caps: dict[str, Any] = {}
    if e.get("supports_tool_call"):
        caps["tool_call"] = True
    if e.get("supports_reasoning"):
        caps["reasoning"] = True
    if "image" in modalities_in:
        caps["vision"] = True

    # 输出 id 统一口径。
    # Claude 例外：官方格式是 claude-{name}-{major}[-{minor}]，
    # 主版本省略 minor（claude-opus-4，没有 -0 / .0），
    # 不能套用通用的「横线转小数点」规则。
    out_id = canonical_output_id(model_id)
    return {
        "id": out_id,
        "vendor": (meta or {}).get("vendor") or infer_vendor(model_id, rows, e),
        "name": name or model_id,
        "aliases": aliases or (meta or {}).get("aliases") or [out_id],
        "context": context,
        "max_output": max_output,
        "modalities": {"input": modalities_in, "output": ["text"]},
        "capabilities": caps,
        "price": price,
        **({"price_usd": price_usd} if price_usd else {}),
        **({"billing": billing_block} if billing_block else {}),
        "kind": e.get("kind") or (meta or {}).get("kind") or "chat",
        "description": description,
        "release_date": release_date,
        "sources": sources or [],
        # ── 以下为增量字段，既有消费方可忽略 ──
        "confidence": conf,
    }


def assemble(
    merged_obs: dict[str, list],
    declared: dict[str, dict[str, Any]],
    alias_index=None,
    family_index=None,
    allowed_vendors: set[str] | None = None,
    *,
    mimi_models: set[str] | None = None,
    meta: dict[str, dict[str, Any]] | None = None,
    extra: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    对每个模型 id 计算证据、应用 serving 约束、输出条目。

    declared: mimi 上报的声明值（同 id 已取最小），仅作参证不作硬上限
    mimi_models: mimi 上报的 id 集合（含自有模型，这些必须保留，即便别处没有）
    """
    allowed_vendors = {str(x).lower() for x in (allowed_vendors or set())}
    mimi_models = mimi_models or set()
    meta = meta or {}
    extra = extra or {}


    # 所有已知 id：证据里出现的 ∪ mimi 上报的
    all_ids = set(merged_obs) | set(mimi_models)

    # 按族聚合。用外部权威 canonical（OpenRouter canonical_slug /
    # models.dev canonical_model_id）决定归属，**不做字符串猜测**。
    # 版本号不同 = 不同模型（v4 与 v4.1、sonnet-4-5 与 sonnet-5 各自独立）。
    # 族键统一走 norm_key，避免 deepseek-v4.1-flash / deepseek_v4.1-flash
    # 因大小写、下划线、小数点写法不同而被拆成两族（会产生重复条目）
    # ── 族分组 ──────────────────────────────────────────────────────
    # 规则来自主流源，不自己造：OpenRouter 的 canonical_slug 已经表达了
    # 「无日期 id 是滚动别名 / 日期快照是独立身份」。
    #   家族 = canonical 去掉尾部日期
    #   deepseek-v4-flash 与 deepseek-v4-flash-0731 同族，各自保留 canonical
    families: dict[str, list[str]] = {}
    for mid in sorted(all_ids):
        # 渠道变体（:batch / @eu）归到基名所在的族
        base_mid = mid
        if vm := re.match(r"^(.*?)(:[a-z0-9_-]+|@[a-z0-9]+)$", mid):
            base_mid = vm.group(1)

        # 族键 = 家族名。
        #
        # 只有当「带日期快照」与「基础名」是**明确的快照关系**时才合并
        # （由 canonical / 短日期后缀给出，见 family 模块）。
        # 不按容量指纹拆分 —— 同一模型在不同源的容量常不一致
        # （官方 200000/4000 vs 中转 189096/8000），那是数据分歧，
        # 应由可信度投票解决，不该拆成两个模型。
        #
        # 反例（确实要分开）：qwen3-235b-a22b 与 qwen3-235b-a22b-2507。
        # 它们不是快照关系（-2507 有自己的 canonical 且 canonical 不指向基础名），
        # 名字像但规格不同 → 不同模型。
        # 族键优先级：
        #   1. 有 canonical 且 canonical 不同于自身 → 用 canonical 的家族
        #      （canonical 是权威源的判断，比名字推断可靠）
        #   2. 否则用自身写法的家族（含短日期/-latest 兜底）
        #
        # 不能取「较短的那个」：qwen3-235b-a22b-2507 的 canonical 是
        # qwen3-235b-a22b-07-25，取短的会被并进 qwen3-235b-a22b，
        # 而两者容量不同（256K vs 128K），是不同模型。
        own = _norm_family(base_mid)
        if family_index:
            resolved = family_index.resolve(base_mid)
            canon_fam = _norm_family(resolved)
            # canonical 指向了带日期的具体快照 → 认它是独立模型
            if family.snapshot_date(resolved) and canon_fam != _norm_family(base_mid):
                fam_key = canon_fam
            else:
                fam_key = own
        else:
            fam_key = own
        families.setdefault(fam_key, []).append(mid)

    models: list[dict[str, Any]] = []
    for root, members in families.items():
        # merged_obs 的键是 loose_key 规范化形式，members 里存的是原始写法，
        # 必须过一遍 loose_key 才能取到数据（否则族合并后会误判为无证据）
        from .sources.collect import loose_key as _lk
        rows = []
        for k in members:
            rows.extend(merged_obs.get(k, []))
            lk = _lk(k)
            if lk != k:
                rows.extend(merged_obs.get(lk, []))

        ctx_ev, out_ev = aggregate_pair(rows)

        # 一族只输出**一条**：同族内的其它写法（快照、别名、变体）
        # 全部并入 aliases。否则会产出多条互为别名的重复卡片 ——
        # 搜索任一种写法时会同时命中它们，看起来就是「一堆重复」。
        if not members:
            continue

        # 主体选择：优先能被权威源识别的写法，其余作为别名。
        #
        # 曾经的坑：mimi 自有模型（morph-dsv4flash）因 declared 有值而被选为
        # primary，结果整个族以它命名，deepseek-v4-flash 反而成了别名。
        # 正确的是「谁在权威源里有身份就用谁」。
        def pick_score(mid: str) -> tuple:
            has_canon = 1 if (family_index and
                              family_index.resolve(mid) != family.normalize_spelling(mid)) else 0
            is_base = 1 if not family.snapshot_date(
                family.normalize_spelling(mid)) else 0
            has_meta = 1 if meta.get(mid) else 0
            # 长度取负：越短越像基础名
            clean = 1 if not re.match(r"^[a-z]+-", mid) or mid.startswith(
                ("gpt-", "glm-", "kimi-", "qwen", "claude-", "gemini-", "grok-",
                 "deepseek-", "mistral-", "llama-", "doubao-", "hunyuan-",
                 "minimax-", "step-", "mimo-", "nova-", "ernie-")) else 0
            return (clean, is_base, has_canon, has_meta, -len(mid), mid)

        primary = max(members, key=pick_score)
        decl = declared.get(primary) or next(
            (declared[m] for m in members if declared.get(m)), None)

        if not rows and not decl:
            continue

        # 同族全部写法（含各快照）作为别名 —— 单独提供某个快照时也能命中
        alias_pool = set(members)
        if alias_index:
            for m in members:
                alias_pool |= alias_index.aliases_of(m)

        # 别名只保留同族写法。canonical_slug 偶尔过度关联
        # （如 deepseek-r1 与 deepseek-r1-distill-qwen-32b），用家族键拦掉，
        # 避免搜索带出不相干结果。
        # 别名需同族。族键在分组时已经带上了容量维度，
        # 所以容量不同的快照本就不同族。
        # 同一族键（已含容量维度）内的写法才是合法别名
        primary_fam = _norm_family(primary)
        alias_pool = {a for a in alias_pool if _norm_family(a) == primary_fam}
        alias_pool.discard(primary)
        alias_list = sorted({idnorm.canonical_id(a) for a in alias_pool if a})

        models.append(build_model(
            primary,
            ctx_ev=ctx_ev,
            out_ev=out_ev,
            declared=decl,
            meta=meta.get(primary),
            extra=extra.get(primary),
            rows=rows,
            aliases=alias_list,
        ))

    # 去重：同一 id 可能因源写法不同产生两条，保留信息更全的一条
    dedup: dict[str, dict[str, Any]] = {}
    for m in models:
        mid = m["id"]
        if mid not in dedup:
            dedup[mid] = m
            continue
        prev = dedup[mid]
        # 打分：有价格、有置信度、别名多、来源多者优先
        def weight(x):
            return ((x.get("price") is not None) * 4
                    + ((x.get("confidence") or {}).get("overall") or 0)
                    + len(x.get("aliases") or []) * 0.1
                    + len(x.get("sources") or []) * 0.5)
        if weight(m) > weight(prev):
            dedup[mid] = m

    models = sorted(dedup.values(), key=lambda m: m["id"])

    # 收录过滤：排除社区微调/无审查、个人发布方、非对话模型
    kept: list[dict[str, Any]] = []
    dropped: list[tuple[str, str]] = []
    for m in models:
        ok, reason = _decide_cached(
            m["id"], m.get("vendor") or "", m.get("name") or "",
            m.get("kind") or "", tuple(sorted(allowed_vendors)),
        )
        if ok:
            kept.append(m)
        else:
            dropped.append((m["id"], reason or ""))
    if dropped:
        assemble_stats["dropped"] = dropped
    return kept


def wrap(models: list[dict[str, Any]]) -> dict[str, Any]:
    """
    套上既有顶层结构。

    容量统一用十进制整 K（向下取整）：1048576 -> 1024000，393216 -> 384000。
    理由见 notation.py：声明值本就是理论上限，取整不损失实际可用空间，
    且只向下取整，消费方按它算永远不会超限。
    """
    models = [notation.apply_to_model(m) for m in models]
    return {
        "version": OUTPUT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(models),
        "models": models,
    }
