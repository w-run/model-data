#!/usr/bin/env python3
"""
构建入口（多阶段）

    python -m model_data.cli                 # 全量（01→04）
    python -m model_data.cli --from 03       # 从 03 开始跑到底
    python -m model_data.cli --only 04       # 只重跑 04（改过滤规则时用，秒级）
    python -m model_data.cli --keep          # 复用已有中间产物

阶段产物在 .build/：
    01-collect.json    原始观测
    02-identity.json   canonical / 族索引
    03-assemble.json   模型条目（未取整、未过滤）
    04-finalize.json   最终产出

改一处只需重跑对应阶段，便于逐段验证。
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from datetime import datetime
from pathlib import Path

from . import assemble, vendor_table
from .family import build_index as build_family_index
from .pipeline import STAGES, Context, Stage, log, run_pipeline
from .sources import collect
from .sources import aliyun as aliyun_src
from .sources import bigmodel as bigmodel_src
from .sources import huggingface as hf_src
from .sources import litellm as litellm_src
from .sources import mimi as mimi_src
from .sources import models_dev_api as models_dev_api_src
from .sources import models_dev_repo, openrouter as openrouter_src
from .sources import siliconflow as siliconflow_src
from .vendors import BUILTIN_ALIASES, load_from_toml, vendor_aliases

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__import__("os").environ.get("MODEL_DATA_CACHE", Path.home() / ".cache" / "model-data"))


# ══ 阶段 01：采集 ══════════════════════════════════════════════════════════

def ensure_models_dev_repo(offline: bool) -> Path | None:
    """稀疏克隆 models.dev 源仓库（只取 providers/，约 39MB）。"""
    import subprocess

    target = CACHE / "models.dev"
    if target.exists():
        if offline:
            return target
        try:
            subprocess.run(["git", "-C", str(target), "pull", "--ff-only", "-q"],
                           check=True, capture_output=True, timeout=180)
            log("    models.dev repo: 已更新")
        except Exception as exc:  # noqa: BLE001
            log(f"    models.dev repo: 更新失败，用现有副本（{exc.__class__.__name__}）")
        return target

    if offline:
        return None

    CACHE.mkdir(parents=True, exist_ok=True)
    log("    models.dev repo: 首次克隆（稀疏，约 39MB）...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                        "--sparse", "https://github.com/sst/models.dev", str(target)],
                       check=True, capture_output=True, timeout=600)
        subprocess.run(["git", "-C", str(target), "sparse-checkout", "set", "providers"],
                       check=True, capture_output=True, timeout=300)
        return target
    except Exception as exc:  # noqa: BLE001
        log(f"    models.dev repo: 克隆失败 {exc}")
        return None


def stage_collect(ctx: Context) -> None:
    """
    采集各源，产出**原始观测**（未合并、未去重），同时保存构建索引所需的原始数据。

    产物结构：
      {
        "observations": {slug: [obs...]},   # 已按 loose_key 合并
        "openrouter_raw": {...},            # 供 02 构建 canonical 索引
        "modelsdev_api_raw": {...},
        "mimi": {"declared": ..., "models": [...], "extra": ...},
      }
    """
    sel = {s.strip() for s in ctx.only_sources.split(",") if s.strip()}
    def want(n: str) -> bool:
        return (not sel) or (n in sel)

    obs_maps: list[dict] = []
    or_data: dict = {}
    md_api_data: dict = {}

    if want("modelsdev"):
        repo = ensure_models_dev_repo(ctx.offline)
        if repo:
            docs = models_dev_repo.scan(str(repo))
            n = sum(len(v) for v in docs.values())
            log(f"    models.dev: {len(docs)} providers, {n} docs")
            obs_maps.append(collect.observations_from_models_dev(docs))
            if not ctx.offline:
                try:
                    md_api_data = models_dev_api_src.fetch()
                    tier_obs = collections_count(
                        collect.observations_from_models_dev_api(md_api_data))
                    log(f"    models.dev API: {tier_obs} 条观测")
                    obs_maps.append(collect.observations_from_models_dev_api(md_api_data))
                except Exception as exc:  # noqa: BLE001
                    log(f"    models.dev API: FAIL {exc}")

    if want("litellm"):
        try:
            data = litellm_src.fetch(offline=ctx.offline)
            log(f"    litellm: {len(data)} entries")
            obs_maps.append(collect.observations_from_litellm(data))
        except Exception as exc:  # noqa: BLE001
            log(f"    litellm: FAIL {exc}")

    if want("aliyun"):
        try:
            o = aliyun_src.observations(offline=ctx.offline)
            log(f"    aliyun: {len(o)} 模型, {collections_count(o)} 条观测（人民币）")
            obs_maps.append(o)
        except Exception as exc:  # noqa: BLE001
            log(f"    aliyun: FAIL {exc}")

    if want("bigmodel"):
        try:
            o = bigmodel_src.observations(offline=ctx.offline)
            log(f"    bigmodel: {len(o)} 模型, {collections_count(o)} 条观测（官方规格）")
            obs_maps.append(o)
        except Exception as exc:  # noqa: BLE001
            log(f"    bigmodel: FAIL {exc}")

    if want("siliconflow"):
        try:
            o = siliconflow_src.observations(offline=ctx.offline)
            log(f"    siliconflow: {len(o)} 模型, {collections_count(o)} 条观测（人民币）")
            obs_maps.append(o)
        except Exception as exc:  # noqa: BLE001
            log(f"    siliconflow: FAIL {exc}")

    if want("huggingface"):
        try:
            data = hf_src.fetch(offline=ctx.offline)
            log(f"    huggingface: {len(data.get('data', []))} models")
            obs_maps.append(hf_src.observations(data))
        except Exception as exc:  # noqa: BLE001
            log(f"    huggingface: FAIL {exc}")

    if want("openrouter"):
        try:
            or_data = openrouter_src.fetch(offline=ctx.offline)
            log(f"    openrouter: {len(or_data.get('data', []))} models")
            obs_maps.append(collect.observations_from_openrouter(or_data))
        except Exception as exc:  # noqa: BLE001
            log(f"    openrouter: FAIL {exc}")

    declared: dict = {}
    mimi_models: list[str] = []
    mimi_extra: dict = {}
    if want("mimi"):
        try:
            data = mimi_src.fetch()
            mimi_obs, declared = mimi_src.observations(data)
            mimi_extra = mimi_src.extra_fields(data)
            mimi_models = [str(m.get("id")) for m in data.get("data", []) if m.get("id")]
            own = [x for x in mimi_models if mimi_src.is_own_model(x)]
            log(f"    mimi: {len(mimi_models)} models（自有 {len(own)}）")
            obs_maps.append(mimi_obs)
        except Exception as exc:  # noqa: BLE001
            log(f"    mimi: FAIL {exc}")

    if not obs_maps:
        raise RuntimeError("没有可用数据源")

    merged = collect.merge_observations(*obs_maps)
    log(f"    对齐后 {len(merged)} 个模型键")

    ctx.save("01", {
        "observations": {
            k: [
                {"field": o.field_name, "value": o.value, "source": o.source,
                 "tier": o.tier, "valid": o.valid, "note": o.note,
                 "currency": o.currency, "provenance": o.provenance,
                 "url": o.url, "observed_at": o.observed_at}
                for o in v
            ]
            for k, v in merged.items()
        },
        "openrouter_raw": or_data,
        "modelsdev_api_raw": md_api_data,
        "mimi": {"declared": declared, "models": mimi_models, "extra": mimi_extra},
    })


def collections_count(o: dict) -> int:
    return sum(len(v) for v in o.values())


# ══ 阶段 02：身份对齐 ══════════════════════════════════════════════════════

def stage_identity(ctx: Context) -> None:
    """
    构建 canonical / 族索引。**规则来自主流源**（OpenRouter canonical_slug），
    不自己造。

    产物：
      {"canonical": {写法: canonical}, "families": {族: [canonical...]},
       "vendors_allowed": [...], "vendors_toml_ids": [...]}
    """
    d = ctx.load("01")
    idx = build_family_index(
        ("openrouter", d.get("openrouter_raw") or {}),
        ("models.dev", d.get("modelsdev_api_raw") or {}),
    )
    log(f"    canonical 关系: {len(idx.canonical)}")
    log(f"    家族数: {len(idx.family_members)}")

    load_from_toml(str(REPO_ROOT / "data" / "vendors.toml"))
    try:
        with open(REPO_ROOT / "data" / "vendors.toml", "rb") as f:
            vids = list(tomllib.load(f).keys())
    except Exception:  # noqa: BLE001
        vids = []

    allowed: set[str] = set()
    for vid in vids:
        allowed.add(vid.lower())
        allowed.update(a.lower() for a in vendor_aliases(vid))
    allowed.update(k.lower() for k in BUILTIN_ALIASES)
    log(f"    白名单厂商写法: {len(allowed)}")

    ctx.save("02", {
        "canonical": idx.canonical,
        "families": {k: sorted(v) for k, v in idx.family_members.items()},
        "spellings": {k: sorted(v) for k, v in idx.spellings.items()},
        "vendors_allowed": sorted(allowed),
        "vendors_toml_ids": vids,
    })


# ══ 阶段 03：装配 ══════════════════════════════════════════════════════════

def stage_assemble(ctx: Context, build_dir: Path) -> None:
    """
    按族装配模型条目（未取整、未过滤），便于单独检查装配结果。
    """
    d1 = ctx.load("01")
    d2 = ctx.load("02")

    from .family import CanonicalIndex
    idx = CanonicalIndex()
    idx.canonical = {k: v for k, v in d2["canonical"].items()}
    idx.family_members = {k: set(v) for k, v in d2["families"].items()}
    idx.spellings = {k: set(v) for k, v in d2["spellings"].items()}

    # 观测还原成 Observation 对象
    from .trust.engine import Observation
    merged: dict[str, list[Observation]] = {}
    for slug, rows in d1["observations"].items():
        merged[slug] = [
            Observation(
                source=r["source"], tier=r["tier"], field_name=r["field"],
                value=r["value"], valid=r["valid"], note=r.get("note"),
                currency=r.get("currency", "USD"),
                provenance=r.get("provenance", "none"),
                url=r.get("url"), observed_at=r.get("observed_at"),
            )
            for r in rows
        ]

    m = d1.get("mimi") or {}
    models = assemble.assemble(
        merged, m.get("declared") or {},
        alias_index=idx,
        family_index=idx,
        allowed_vendors=set(d2["vendors_allowed"]),
        mimi_models=set(m.get("models") or []),
        extra=m.get("extra") or {},
    )
    log(f"    装配出 {len(models)} 条")

    dropped = assemble.assemble_stats.get("dropped") or []
    if dropped:
        from collections import Counter
        reasons = Counter(r.split("（")[0] for _, r in dropped)
        log("    过滤: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))

    ctx.save("03", {"models": models, "dropped": dropped})


# ══ 阶段 04：定稿 ══════════════════════════════════════════════════════════

def stage_finalize(ctx: Context) -> None:
    """取整、校验、写出最终产出。"""
    d = ctx.load("03")
    models = d["models"]

    out = assemble.wrap(models)
    problems = validate(out["models"])

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    (ctx.output_dir / "models.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    vtab = vendor_table.build(models, REPO_ROOT / "data" / "vendors.toml")
    (ctx.output_dir / "vendors.json").write_text(
        json.dumps(vtab, ensure_ascii=False, indent=2), encoding="utf-8")

    levels: dict[str, int] = {}
    for m in models:
        levels[m["confidence"]["level"]] = levels.get(m["confidence"]["level"], 0) + 1

    log("")
    log(f"    models: {len(models)}")
    log(f"    confidence: {', '.join(f'{k}={v}' for k, v in sorted(levels.items()))}")
    log(f"    vendors: {vtab['count']}")
    if problems:
        log(f"    校验: {len(problems)} 个问题")
        for p in problems[:15]:
            log(f"      {p}")
    else:
        log("    校验: 通过")
    log(f"    output: {ctx.output_dir}")

    # 校验结果留给 main 判断 --strict
    ctx.data["problems"] = problems


def validate(models: list[dict]) -> list[str]:
    """产出校验：只看会不会坑到消费方。"""
    problems: list[str] = []
    for m in models:
        mid = m.get("id")
        ctx, out = m.get("context"), m.get("max_output")
        if ctx is not None and out is not None:
            if out >= ctx:
                problems.append(f"{mid}: max_output({out}) >= context({ctx})")
            elif abs(out - (ctx * 9) // 10) <= 1:
                problems.append(f"{mid}: max_output({out}) 疑似 floor(0.9*context) 占位值")
        if ctx is not None and ctx <= 0:
            problems.append(f"{mid}: context={ctx} 非法")
        if out is not None and out <= 0:
            problems.append(f"{mid}: max_output={out} 非法")
    return problems


# ══ main ══════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="构建模型信息库（多阶段）")
    ap.add_argument("-o", "--output-dir", type=Path, default=REPO_ROOT)
    ap.add_argument("--build-dir", type=Path, default=None,
                    help="中间产物目录，默认 <output>/.build")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--only", default="", help="只跑指定阶段（01,02,03,04）")
    ap.add_argument("--sources", default="",
                    help="逗号分隔数据源：modelsdev,aliyun,bigmodel,siliconflow,huggingface,litellm,openrouter,mimi")
    ap.add_argument("--from", dest="start", default="", help="从某阶段开始（如 03）")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)

    build_dir = args.build_dir or (args.output_dir / ".build")
    build_dir.mkdir(parents=True, exist_ok=True)

    ctx = Context(build_dir=build_dir, output_dir=args.output_dir,
                  offline=args.offline, only_sources=args.sources)

    log("=" * 64)
    log("  model-data build（多阶段）")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  产物目录: {build_dir}")
    log("=" * 64)

    stages = [
        Stage("01", "collect", stage_collect),
        Stage("02", "identity", stage_identity, needs=("01",)),
        Stage("03", "assemble", lambda c: stage_assemble(c, build_dir), needs=("02",)),
        Stage("04", "finalize", stage_finalize, needs=("03",)),
    ]

    run_pipeline(stages, ctx, start=args.start, stage_only=args.only)
    log("")
    log("  全部完成")

    problems = ctx.data.get("problems") or []
    if args.strict and problems:
        log(f"  --strict：{len(problems)} 个校验问题，返回非零")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
