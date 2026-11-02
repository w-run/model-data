#!/usr/bin/env python3
"""
证据采集：把各数据源翻译成统一的 Observation 流

这一层只做「翻译」，不做「判断」。任何取舍都交给 trust.engine。

源清单
------
  models.dev 仓库   TOML + 出处注释      → first_party / official_cloud / aggregator
  LiteLLM           JSON + source URL    → community
  OpenRouter        API                  → openrouter（占位值直接标 invalid）
  厂商 API          GET /models           → first_party_api
"""

from __future__ import annotations

import json
import math
import urllib.request
from typing import Any, Iterable

from ..currency import channel_currency, is_cny_channel as channel_is_cn
from ..trust.engine import Observation, capacity_keys
from . import models_dev_repo as mdr

UA = "w-run-model-data/4.0 (+https://github.com/w-run/model-data)"


# ── 有效性判定 ──────────────────────────────────────────────────────────────

def placeholder_reason(context: int | None, output: int | None) -> str | None:
    """
    判断一个 (context, output) 是否明显是填充/错误值。

    这是全库最重要的检查：OpenRouter 和 models.dev 的聚合 provider 都会用
    floor(ctx*0.9) 填 max_completion_tokens，看起来合法但会误导消费方。
    """
    if not (context and output):
        return None
    if output >= context:
        return "output>=context"
    if abs(output - math.floor(context * 0.9)) <= 1:
        return "floor(0.9*context)"
    return None


# ── models.dev 仓库 → 观测 ─────────────────────────────────────────────────

def observations_from_models_dev(docs: dict[str, list[mdr.Doc]]) -> dict[str, list[Observation]]:
    """
    返回 {canonical_slug: [Observation, ...]}

    provenance 判定：
      注释里有官方文档的 URL → doc_url（+0.10）
      有 pricing 链接          → pricing_url（+0.05）
      都没有                   → none
    """
    index = mdr.build_index(docs)
    out: dict[str, list[Observation]] = {}

    for provider, items in docs.items():
        tier = mdr.tier_of(provider)
        for doc in items:
            slug = mdr.canonical_key(doc)
            resolved = mdr.resolve_limit(doc, index)

            # 出处质量
            urls = doc.sources
            if any("api-docs" in u or "/docs" in u or "models" in u for u in urls):
                prov = "doc_url"
            elif any("pricing" in u or "price" in u for u in urls):
                prov = "pricing_url"
            else:
                prov = "none"
            main_url = urls[0] if urls else None

            base_note = None
            if resolved and resolved.get("inherited_from"):
                base_note = f"inherited from {resolved['inherited_from']} ({resolved['hops']} hops)"

            if resolved:
                ctx, outp = resolved["context"], resolved["output"]
                bad = placeholder_reason(ctx, outp)
                out.setdefault(slug, []).append(Observation(
                    source=f"models.dev:{provider}",
                    tier=tier,
                    field_name="context",
                    value=ctx,
                    provenance=prov,
                    url=main_url,
                    observed_at=doc.accessed_at,
                    note=base_note,
                    valid=True,
                ))
                if outp is not None:
                    out.setdefault(slug, []).append(Observation(
                        source=f"models.dev:{provider}",
                        tier=tier,
                        field_name="max_output",
                        value=outp,
                        provenance=prov,
                        url=main_url,
                        observed_at=doc.accessed_at,
                        note=(base_note + "; " if base_note else "") + (
                            f"REJECTED: {bad}" if bad else ""
                        ) or None,
                        valid=bad is None,
                    ))

            # 价格：models.dev 用 {"text": {"input": {"price": x}, "output": {"price": y}}}
            # 单位是 USD per Mtok（与既有产出一致）
            cost = doc.data.get("cost") or {}
            text_cost = cost.get("text") or {}
            for key, field_name in (("input", "price_input"), ("output", "price_output")):
                block = text_cost.get(key) or {}
                if isinstance(block, dict) and isinstance(block.get("price"), (int, float)):
                    out.setdefault(slug, []).append(Observation(
                        source=f"models.dev:{provider}",
                        tier=tier,
                        field_name=field_name,
                        value=float(block["price"]),
                        # models.dev 对国内 provider 存的是**人民币按汇率换算后的美元**。
                        # 标成 converted（负加成），让原生人民币源优先，
                        # 但不丢弃 —— 有些模型只有它才有价。
                        provenance=("converted" if channel_is_cn(provider) else prov),
                        url=main_url,
                        observed_at=doc.accessed_at,
                        valid=True,
                        currency="USD",
                    ))
                    cache = block.get("cache")
                    if isinstance(cache, dict):
                        for ck, cf in (("read", "price_cache_read"), ("write", "price_cache_write")):
                            if isinstance(cache.get(ck), (int, float)):
                                out.setdefault(slug, []).append(Observation(
                                    source=f"models.dev:{provider}",
                                    tier=tier,
                                    field_name=cf,
                                    value=float(cache[ck]),
                                    provenance=prov,
                                    url=main_url,
                                    observed_at=doc.accessed_at,
                                    valid=True,
                                    currency="USD",
                                ))

            # 阶梯计费：cost.text.tiers = [{input, output, cache_read, tier:{type,size}}]
            for t in (text_cost.get("tiers") or []):
                if not isinstance(t, dict):
                    continue
                out.setdefault(slug, []).append(Observation(
                    source=f"models.dev:{provider}",
                    tier=tier,
                    field_name="price_tier",
                    value=t,
                    provenance=prov,
                    url=main_url,
                    observed_at=doc.accessed_at,
                    valid=True,
                    currency="USD",
                ))

            # 时段倍率：[[cost.multipliers]] factor = 2
            for mult in (cost.get("multipliers") or []):
                if not isinstance(mult, dict):
                    continue
                out.setdefault(slug, []).append(Observation(
                    source=f"models.dev:{provider}",
                    tier=tier,
                    field_name="price_multiplier",
                    value=mult,
                    provenance=prov,
                    url=main_url,
                    observed_at=doc.accessed_at,
                    valid=True,
                    currency="USD",
                ))

            # 归属厂商：只在有明确 org 目录或第一方 provider 时记录
            org = mdr.org_of(doc)
            if org:
                out.setdefault(slug, []).append(Observation(
                    source=f"models.dev:{provider}",
                    tier=tier,
                    field_name="vendor",
                    value=org,
                    provenance=prov,
                    url=main_url,
                    observed_at=doc.accessed_at,
                    valid=True,
                ))
            elif tier == "first_party":
                out.setdefault(slug, []).append(Observation(
                    source=f"models.dev:{provider}",
                    tier=tier,
                    field_name="vendor",
                    value=provider,
                    provenance=prov,
                    url=main_url,
                    observed_at=doc.accessed_at,
                    valid=True,
                ))

            # 发布日期：仓库里每条都有
            rel = doc.data.get("release_date")
            if isinstance(rel, str) and rel.strip():
                out.setdefault(slug, []).append(Observation(
                    source=f"models.dev:{provider}",
                    tier=tier,
                    field_name="release_date",
                    value=rel.strip()[:10],
                    provenance=prov,
                    url=main_url,
                    observed_at=doc.accessed_at,
                    valid=True,
                ))

            # 描述与显示名（只从条目自身读，不继承）
            if isinstance(doc.data.get("description"), str) and doc.data["description"].strip():
                out.setdefault(slug, []).append(Observation(
                    source=f"models.dev:{provider}",
                    tier=tier,
                    field_name="description",
                    value=doc.data["description"].strip(),
                    provenance=prov,
                    url=main_url,
                    observed_at=doc.accessed_at,
                    valid=True,
                ))
            if isinstance(doc.data.get("name"), str) and doc.data["name"].strip():
                out.setdefault(slug, []).append(Observation(
                    source=f"models.dev:{provider}",
                    tier=tier,
                    field_name="name",
                    value=doc.data["name"].strip(),
                    provenance=prov,
                    url=main_url,
                    observed_at=doc.accessed_at,
                    valid=True,
                ))

    return out


# ── LiteLLM → 观测 ─────────────────────────────────────────────────────────

def observations_from_litellm(data: dict[str, Any]) -> dict[str, list[Observation]]:
    """
    LiteLLM 的 key 形如 "provider/model"，条目内 litellm_provider 标明渠道。
    source 字段通常是计费 API 或文档 URL，可用于 provenance。
    """
    out: dict[str, list[Observation]] = {}
    for key, rec in data.items():
        if not isinstance(rec, dict):
            continue
        model = key.split("/", 1)[1] if "/" in key else key
        slug = model.split("/")[-1]
        if not slug:
            continue

        src = rec.get("source")
        prov = "pricing_url" if isinstance(src, str) and "pric" in src.lower() else (
            "doc_url" if isinstance(src, str) and src.startswith("http") else "none")

        ctx = rec.get("max_input_tokens")
        outp = rec.get("max_output_tokens")
        bad = placeholder_reason(ctx, outp)

        if isinstance(ctx, int) and ctx > 0:
            out.setdefault(slug, []).append(Observation(
                source=f"litellm:{rec.get('litellm_provider', '?')}",
                tier="community",
                field_name="context",
                value=ctx,
                provenance=prov,
                url=src if isinstance(src, str) else None,
                observed_at=None,
                valid=True,
            ))
        if isinstance(outp, int) and outp > 0:
            out.setdefault(slug, []).append(Observation(
                source=f"litellm:{rec.get('litellm_provider', '?')}",
                tier="community",
                field_name="max_output",
                value=outp,
                provenance=prov,
                url=src if isinstance(src, str) else None,
                observed_at=None,
                note=f"REJECTED: {bad}" if bad else None,
                valid=bad is None,
            ))
    return out


# ── OpenRouter → 观测 ──────────────────────────────────────────────────────

def observations_from_openrouter(data: dict[str, Any]) -> dict[str, list[Observation]]:
    """
    最低可信度的一源。保留是为了覆盖率，但占位值一律 invalid。
    """
    out: dict[str, list[Observation]] = {}
    for m in data.get("data", []) or []:
        mid = str(m.get("id") or "")
        if not mid or mid.startswith("~") or "/" not in mid:
            continue
        slug = mid.split("/", 1)[1]

        ctx = m.get("context_length")
        top = m.get("top_provider") or {}
        outp = top.get("max_completion_tokens")
        bad = placeholder_reason(ctx, outp)

        if isinstance(ctx, int) and ctx > 0:
            out.setdefault(slug, []).append(Observation(
                source="openrouter",
                tier="openrouter",
                field_name="context",
                value=ctx,
                provenance="none",
                observed_at=None,
                valid=True,
            ))
        if isinstance(outp, int) and outp > 0:
            out.setdefault(slug, []).append(Observation(
                source="openrouter",
                tier="openrouter",
                field_name="max_output",
                value=outp,
                provenance="none",
                observed_at=None,
                note=f"REJECTED: {bad}" if bad else None,
                valid=bad is None,
            ))

        # 价格：OpenRouter 是「每 token」字符串，换成每 Mtok
        pricing = m.get("pricing") or {}
        for src_key, field_name in (("prompt", "price_input"),
                                    ("completion", "price_output"),
                                    ("input_cache_read", "price_cache_read")):
            raw = pricing.get(src_key)
            try:
                per_mtok = round(float(raw) * 1_000_000, 6)
            except (TypeError, ValueError):
                continue
            if per_mtok <= 0:
                continue
            out.setdefault(slug, []).append(Observation(
                source="openrouter",
                tier="openrouter",
                field_name=field_name,
                value=per_mtok,
                provenance="none",
                observed_at=None,
                valid=True,
                currency="USD",
            ))

        created = m.get("created")
        if isinstance(created, (int, float)) and created > 0:
            from datetime import datetime, timezone
            try:
                d = datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%d")
                out.setdefault(slug, []).append(Observation(
                    source="openrouter",
                    tier="openrouter",
                    field_name="release_date",
                    value=d,
                    provenance="none",
                    observed_at=None,
                    valid=True,
                ))
            except (OSError, ValueError):
                pass

        desc = m.get("description")
        if isinstance(desc, str) and desc.strip():
            out.setdefault(slug, []).append(Observation(
                source="openrouter",
                tier="openrouter",
                field_name="description",
                value=desc.strip(),
                provenance="none",
                observed_at=None,
                valid=True,
            ))
        nm = m.get("name")
        if isinstance(nm, str) and nm.strip():
            out.setdefault(slug, []).append(Observation(
                source="openrouter",
                tier="openrouter",
                field_name="name",
                value=nm.strip(),
                provenance="none",
                observed_at=None,
                valid=True,
            ))
    return out


# ── 合并 ────────────────────────────────────────────────────────────────────

def merge_observations(*maps: dict[str, list[Observation]]) -> dict[str, list[Observation]]:
    """
    按宽松键合并多源观测。

    不同源对同一模型的写法不一致（4.5 vs 4-5、org/model vs model），
    必须在对齐层抹平，否则投票会被拆散。
    """
    merged: dict[str, list[Observation]] = {}
    for m in maps:
        for slug, obs in m.items():
            k = loose_key(slug)
            merged.setdefault(k, []).extend(obs)
    return merged


def loose_key(slug: str) -> str:
    """
    跨源对齐键。

    用 identity.norm_key：小写 + 小数点/短横线统一，
    **不折叠日期、不动版本号** —— 版本号不同是不同模型。
    """
    from ..identity import norm_key
    return norm_key(slug)


# ── models.dev API（仅用于补充仓库缺失的阶梯价）───────────────────────────

def observations_from_models_dev_api(data: dict[str, Any]) -> dict[str, list[Observation]]:
    """
    API 版才有 tiers（仓库 TOML 里没有，只出现在注释）。

    实测 821 个条目带 tiers，格式：
        cost.tiers = [{"input":4,"output":18,"cache_read":0.2,
                       "tier":{"type":"context","size":200000}}]

    只提取阶梯，容量/价格仍以仓库版本为准 —— 仓库有人工核对与出处。
    """
    out: dict[str, list[Observation]] = {}
    for provider, pdata in (data or {}).items():
        tier = mdr.tier_of(provider)
        for doc_id, m in ((pdata or {}).get("models") or {}).items():
            if not isinstance(m, dict):
                continue
            slug = str(m.get("canonical_model_id") or m.get("id") or doc_id).split("/")[-1]

            # 发布日期：API 覆盖 100%，用来补仓库缺失的部分
            rel = m.get("release_date")
            if isinstance(rel, str) and rel.strip():
                out.setdefault(slug, []).append(Observation(
                    source=f"models.dev-api:{provider}",
                    tier=tier,
                    field_name="release_date",
                    value=rel.strip()[:10],
                    provenance="api_live",
                    url="https://models.dev/api.json",
                    observed_at=None,
                    valid=True,
                ))

            cost = m.get("cost") or {}
            tiers = cost.get("tiers")
            if not isinstance(tiers, list) or not tiers:
                continue
            for t in tiers:
                if not isinstance(t, dict):
                    continue
                out.setdefault(slug, []).append(Observation(
                    source=f"models.dev-api:{provider}",
                    tier=tier,
                    field_name="price_tier",
                    value=t,
                    provenance="api_live",
                    url="https://models.dev/api.json",
                    observed_at=None,
                    valid=True,
                    currency="USD",
                ))
    return out
