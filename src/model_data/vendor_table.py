#!/usr/bin/env python3
"""
厂商表（vendors.json）

与 models.json 并列产出，独立维护。

数据来源
--------
  data/vendors.toml   人工维护：显示名、官网、API 文档（权威，可覆盖）
  models.json 汇总    自动统计：每个厂商有多少模型

两者合并：toml 里有的用人工信息，没有的用自动统计的最小集（只有 id）。

为什么单独一个文件
------------------
1. 厂商信息（官网、文档、别名）变动远慢于模型，不必每次重建
2. 消费方按厂商聚合模型时，不必从 models.json 里再扫一遍
3. 搜索页需要「按厂商检索 / 含别名」，单独一张表查询更直接
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .vendors import canonical_vendor, vendor_aliases, vendor_meta

try:
    import tomllib
except ModuleNotFoundError:  # py<3.11
    import tomli as tomllib  # type: ignore

VERSION = "1.0.0"


def load_toml(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "rb") as f:
            return tomllib.load(f)
    except Exception:  # noqa: BLE001
        return {}


def build(
    models: list[dict[str, Any]],
    vendors_toml_path: str | Path = "data/vendors.toml",
) -> dict[str, Any]:
    """
    汇总出 vendors.json。

    每个条目：
        id        归一后的厂商 id（deepseek / zhipuai …）
        name      显示名（中文厂商用中文）
        aliases   全部已知写法（含 toml 的 alias 与内置别名表）
        website / api_docs
        model_count
        currencies  该厂商模型使用的计价币种集合
    """
    manual = load_toml(vendors_toml_path)

    # 自动统计
    stats: dict[str, dict[str, Any]] = {}
    for m in models:
        vid = m.get("vendor") or "unknown"
        st = stats.setdefault(vid, {"model_count": 0, "currencies": set()})
        st["model_count"] += 1
        unit = (m.get("price") or {}).get("unit")
        if unit:
            st["currencies"].add(unit.split("/")[0])

    # 人工表：canonical id -> 配置（toml 是权威源，别名全在 alias 字段里）
    manual_by_canon: dict[str, dict[str, Any]] = {}
    for raw_id, cfg in manual.items():
        canon = canonical_vendor(raw_id)
        entry = dict(cfg)
        entry.setdefault("id", canon)
        entry["aliases"] = sorted(set(
            (cfg.get("alias") or []) + vendor_aliases(canon) + [raw_id, canon]))
        manual_by_canon[canon] = entry

    out: list[dict[str, Any]] = []
    for vid, st in stats.items():
        canon = canonical_vendor(vid)
        base = manual_by_canon.get(canon, {})
        item = {
            "id": canon,
            "name": base.get("name") or canon,
            "aliases": base.get("aliases") or sorted(set(vendor_aliases(canon))),
            "website": base.get("website"),
            "api_docs": base.get("api_docs"),
            "model_count": st["model_count"],
            "currencies": sorted(st["currencies"]),
            "region": base.get("region"),
            "cn": base.get("cn"),
            "own": base.get("own", False),
            "aggregator": base.get("aggregator", False),
            "source": "manual" if base else "auto",
        }
        out.append(item)

    # 人工表里有但没有任何模型的厂商也保留（便于搜索时发现）
    known_ids = {x["id"] for x in out}
    for canon, base in manual_by_canon.items():
        if canon in known_ids:
            continue
        out.append({
            "id": canon,
            "name": base.get("name") or canon,
            "aliases": base["aliases"],
            "website": base.get("website"),
            "api_docs": base.get("api_docs"),
            "model_count": 0,
            "currencies": [],
            "source": "manual",
        })

    out.sort(key=lambda x: (-x["model_count"], x["id"]))

    return {
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(out),
        "vendors": out,
    }
