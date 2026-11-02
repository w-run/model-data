#!/usr/bin/env python3
"""
构建入口

    python -m model_data.cli              # 全量构建
    python -m model_data.cli --offline    # 不联网（用缓存）
    python -m model_data.cli -o /tmp/out  # 输出到别处

流程
----
  采集（models.dev 仓库 / LiteLLM / OpenRouter / mimi）
    → 身份对齐（同族合并）
    → 可信度聚合（分字段）
    → mimi 声明值参证（不当硬上限）
    → 产出（既有格式 + confidence 增量）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import assemble, vendor_table
from .vendors import load_from_toml
from .identity import build_alias_index
from .sources import collect
from .sources import aliyun as aliyun_src
from .sources import siliconflow as siliconflow_src
from .sources import huggingface as hf_src
from .sources import litellm as litellm_src
from .sources import mimi as mimi_src
from .sources import models_dev_api as models_dev_api_src
from .sources import models_dev_repo, openrouter as openrouter_src

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("MODEL_DATA_CACHE", Path.home() / ".cache" / "model-data"))


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_models_dev_repo(offline: bool) -> Path | None:
    """稀疏克隆 models.dev 源仓库（只取 providers/，约 39MB）。"""
    target = CACHE / "models.dev"
    if target.exists():
        if offline:
            return target
        try:
            subprocess.run(
                ["git", "-C", str(target), "pull", "--ff-only", "-q"],
                check=True, capture_output=True, timeout=180,
            )
            log("  models.dev repo: 已更新")
        except Exception as exc:  # noqa: BLE001
            log(f"  models.dev repo: 更新失败，用现有副本（{exc.__class__.__name__}）")
        return target

    if offline:
        return None

    CACHE.mkdir(parents=True, exist_ok=True)
    log("  models.dev repo: 首次克隆（稀疏，约 39MB）...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "https://github.com/sst/models.dev", str(target)],
            check=True, capture_output=True, timeout=600,
        )
        subprocess.run(
            ["git", "-C", str(target), "sparse-checkout", "set", "providers"],
            check=True, capture_output=True, timeout=300,
        )
        log("  models.dev repo: OK")
        return target
    except Exception as exc:  # noqa: BLE001
        log(f"  models.dev repo: 克隆失败 {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="构建模型信息库")
    ap.add_argument("-o", "--output-dir", type=Path, default=REPO_ROOT)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--only", default="", help="逗号分隔：modelsdev,aliyun,siliconflow,huggingface,litellm,openrouter,mimi")
    ap.add_argument("--strict", action="store_true", help="有 error 即退出非零")
    ap.add_argument("--dump-observations", type=Path, default=None,
                    help="把原始观测落盘，便于排查")
    args = ap.parse_args(argv)

    sel = {s.strip() for s in args.only.split(",") if s.strip()}
    def want(n: str) -> bool:
        return (not sel) or (n in sel)

    # 厂商表是归一与别名的权威源，必须在任何采集之前加载
    load_from_toml(str(REPO_ROOT / "data" / "vendors.toml"))

    log("=" * 64)
    log("  model-data build")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 64)

    # ── 1. 采集 ────────────────────────────────────────────────────
    obs_maps = []
    or_data: dict = {}
    md_api_data: dict = {}

    if want("modelsdev"):
        repo = ensure_models_dev_repo(args.offline)
        if repo:
            docs = models_dev_repo.scan(str(repo))
            n = sum(len(v) for v in docs.values())
            log(f"  models.dev: {len(docs)} providers, {n} docs")
            obs_maps.append(collect.observations_from_models_dev(docs))
            # 国内 provider 的价格反推回人民币，优先于美元换算值
            # 阶梯价只在 API 版有（仓库 TOML 缺），单独拉一次补上
            if not args.offline:
                try:
                    api_data = models_dev_api_src.fetch()
                    md_api_data = api_data
                    tier_obs = collect.observations_from_models_dev_api(api_data)
                    n = sum(len(v) for v in tier_obs.values())
                    log(f"  models.dev API: {len(tier_obs)} 模型, {n} 条阶梯价")
                    obs_maps.append(tier_obs)
                except Exception as exc:  # noqa: BLE001
                    log(f"  models.dev API: FAIL {exc}")


    if want("litellm"):
        try:
            data = litellm_src.fetch(offline=args.offline)
            log(f"  litellm: {len(data)} entries")
            obs_maps.append(collect.observations_from_litellm(data))
        except Exception as exc:  # noqa: BLE001
            log(f"  litellm: FAIL {exc}")

    if want("aliyun"):
        try:
            ali_obs = aliyun_src.observations(offline=args.offline)
            n = sum(len(v) for v in ali_obs.values())
            log(f"  aliyun: {len(ali_obs)} 模型, {n} 条观测（人民币原价）")
            obs_maps.append(ali_obs)
        except Exception as exc:  # noqa: BLE001
            log(f"  aliyun: FAIL {exc}")

    if want("siliconflow"):
        try:
            sf_obs = siliconflow_src.observations(offline=args.offline)
            n = sum(len(v) for v in sf_obs.values())
            log(f"  siliconflow: {len(sf_obs)} 模型, {n} 条观测（人民币）")
            obs_maps.append(sf_obs)
        except Exception as exc:  # noqa: BLE001
            log(f"  siliconflow: FAIL {exc}")

    if want("huggingface"):
        try:
            data = hf_src.fetch(offline=args.offline)
            models_n = len(data.get("data", []))
            prov_n = sum(len(m.get("providers") or []) for m in data.get("data", []))
            log(f"  huggingface: {models_n} models, {prov_n} provider entries")
            obs_maps.append(hf_src.observations(data))
        except Exception as exc:  # noqa: BLE001
            log(f"  huggingface: FAIL {exc}")

    if want("openrouter"):
        try:
            data = openrouter_src.fetch(offline=args.offline)
            or_data = data
            log(f"  openrouter: {len(data.get('data', []))} models")
            obs_maps.append(collect.observations_from_openrouter(data))
        except Exception as exc:  # noqa: BLE001
            log(f"  openrouter: FAIL {exc}")

    declared: dict = {}
    mimi_models: set = set()
    mimi_extra: dict = {}
    if want("mimi"):
        try:
            data = mimi_src.fetch()
            mimi_obs, declared = mimi_src.observations(data)
            mimi_extra = mimi_src.extra_fields(data)
            mimi_models = {str(m.get("id")) for m in data.get("data", []) if m.get("id")}
            own = [x for x in mimi_models if mimi_src.is_own_model(x)]
            log(f"  mimi: {len(mimi_models)} models（自有 {len(own)}）")
            obs_maps.append(mimi_obs)
        except Exception as exc:  # noqa: BLE001
            log(f"  mimi: FAIL {exc}")

    if not obs_maps:
        log("没有可用数据源，中止")
        return 2

    merged = collect.merge_observations(*obs_maps)

    if args.dump_observations:
        payload = {
            k: [
                {"field": o.field_name, "value": o.value, "source": o.source,
                 "tier": o.tier, "valid": o.valid, "note": o.note}
                for o in v
            ]
            for k, v in merged.items()
        }
        args.dump_observations.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        log(f"  观测已落盘: {args.dump_observations}")

    # ── 2. 装配 ────────────────────────────────────────────────────
    # 用外部权威元数据构建别名索引（不做字符串猜测）
    alias_index = build_alias_index(
        ("openrouter", or_data),
        ("models.dev", md_api_data),
    )
    log(f"  别名索引: {len(alias_index.to_canonical)} 条 canonical 关系")
    models = assemble.assemble(
        merged, declared,
        alias_index=alias_index,
        mimi_models=mimi_models,
        extra=mimi_extra,
    )
    out = assemble.wrap(models)

    # ── 3. 校验 ────────────────────────────────────────────────────
    # 校验最终产出（取整后），而不是取整前的中间态
    problems = validate(out["models"])
    if problems:
        log("")
        log(f"  校验: {len(problems)} 个问题")
        for p in problems[:15]:
            log(f"    {p}")
    else:
        log("")
        log("  校验: 通过")

    # ── 4. 写出 ────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 厂商表独立产出（搜索页需要按厂商/别名检索）
    vtab = vendor_table.build(models, REPO_ROOT / "data" / "vendors.toml")
    (args.output_dir / "vendors.json").write_text(
        json.dumps(vtab, ensure_ascii=False, indent=2), encoding="utf-8")

    levels: dict[str, int] = {}
    for m in models:
        lv = m["confidence"]["level"]
        levels[lv] = levels.get(lv, 0) + 1

    log("")
    log(f"  models: {len(models)}")
    log(f"  confidence: {', '.join(f'{k}={v}' for k, v in sorted(levels.items()))}")
    log(f"  vendors: {vtab['count']}")
    log(f"  output: {args.output_dir}  (models.json / vendors.json)")
    log("=" * 64)

    if args.strict and problems:
        return 1
    return 0


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


if __name__ == "__main__":
    sys.exit(main())
