#!/usr/bin/env python3
"""
model-data 构建脚本（重写版）

设计目标
--------
1. **不筛选**：旧版按 data/vendors.toml 白名单过滤，OpenRouter 上 452 个模型
   会被砍到 300 上下，大量真实模型直接丢失。新版收录所有模型，厂商只是元信息。
2. **权威优先、缺失继承**：按数据源优先级合并，高优先级只补自己的字段，
   低优先级补齐空字段。任何字段都不写死在脚本里。
3. **自有模型也能收录**：mimi-* 系列在公开目录里不存在，直接从我们的
   聚合网关 https://mimi.wrun.cc/v1/models 抓，含上下文/输出/模态/能力。
4. **消费方友好**：额外产出 index.json（归一化别名 -> 模型 id），
   让 sub2api / mimi-router 之类按自己的写法快速命中。

数据源优先级（高 → 低）
    1. manual       data/manual/overrides.json（人工标称，最高优先级）
    2. openrouter   https://openrouter.ai/api/v1/models（公开权威、覆盖最广）
    3. mimi         https://mimi.wrun.cc/v1/models（自有 mimi-* 系列）

产出（默认写仓库根目录，gh-pages 直接托管）
    models.json   模型列表
    vendors.json  厂商列表
    index.json    别名索引

用法
    python -m scripts.build                # 全量
    python -m scripts.build -o out         # 输出到指定目录
    python -m scripts.build --only openrouter,mimi
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "2.0.0"
BASE_DIR = Path(__file__).resolve().parent.parent
MANUAL_PATH = BASE_DIR / "data" / "manual" / "overrides.json"
UA = "w-run-model-data/2.0 (+https://github.com/w-run/model-data)"
USD_UNIT = "USD/Mtok"


# ── HTTP ─────────────────────────────────────────────────────────────────────


def http_get_json(url: str, timeout: int = 60, headers: dict[str, str] | None = None) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── 归一化 ───────────────────────────────────────────────────────────────────


def normalize_id(raw: str) -> str:
    """模型/厂商 ID 归一化：小写、空格与下划线转短横线、去掉 OpenRouter 变体后缀。"""
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    for suf in (":free", ":extended", ":nitro", ":online", ":thinking", ":floor"):
        s = s[: -len(suf)] if s.endswith(suf) else s
    s = s.replace("_", "-").replace(" ", "-")
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def alias_keys(model_id: str, full_id: str | None = None) -> set[str]:
    """生成用于 index.json 的匹配键：短名、全名、去 -solo、版本号点横两种写法。"""
    keys: set[str] = set()
    for candidate in (model_id, full_id):
        if not candidate:
            continue
        base = normalize_id(candidate)
        if not base:
            continue
        keys.add(base)
        if "/" in base:
            keys.add(base.split("/", 1)[1])
    # 版本号写法互通：v4-1 / v4.1
    for k in list(keys):
        keys.add(re.sub(r"-(\d)-1-", r"-\1.1-", k))
    keys.discard("")
    return keys


# ── 数据源 ───────────────────────────────────────────────────────────────────


def fetch_openrouter() -> list[dict[str, Any]]:
    data = http_get_json("https://openrouter.ai/api/v1/models", timeout=60)
    out: list[dict[str, Any]] = []
    for m in data.get("data", []) or []:
        full_id = str(m.get("id") or "")
        if not full_id or "/" not in full_id:
            continue
        org, name_part = full_id.split("/", 1)
        if org.startswith("~"):
            continue

        raw_name = str(m.get("name") or name_part)
        display = raw_name.split(": ", 1)[1] if ": " in raw_name else raw_name
        display = re.sub(r"\s*\((free|extended)\)\s*$", "", display, flags=re.I)

        arch = m.get("architecture") or {}
        modalities_in = arch.get("input_modalities") or []
        modalities_out = arch.get("output_modalities") or ["text"]
        if not modalities_in and arch.get("modality") and "->" in arch["modality"]:
            left = arch["modality"].split("->")[0]
            modalities_in = [x.strip() for x in left.split("+") if x.strip()]

        params = m.get("supported_parameters") or []
        caps: dict[str, Any] = {}
        if "tools" in params or "tool_choice" in params:
            caps["tool_call"] = True
        if "reasoning" in params or "include_reasoning" in params:
            caps["reasoning"] = True
        if "structured_outputs" in params or "response_format" in params:
            caps["structured_output"] = True
        if "temperature" in params:
            caps["temperature"] = True
        if "image" in modalities_in:
            caps["vision"] = True

        price: dict[str, Any] = {"unit": USD_UNIT}
        for src_key, dst_key in (
            ("prompt", "input"),
            ("completion", "output"),
            ("input_cache_read", "cache_read"),
            ("input_cache_write", "cache_write"),
            ("image", "image_input"),
            ("web_search", "web_search"),
            ("reasoning", "reasoning"),
        ):
            val = (m.get("pricing") or {}).get(src_key)
            if val in (None, "", "0"):
                continue
            try:
                price[dst_key] = round(float(val) * 1_000_000, 6)
            except (TypeError, ValueError):
                continue

        top = m.get("top_provider") or {}
        out.append(
            {
                "id": normalize_id(name_part),
                "vendor": normalize_id(org),
                "name": display,
                "aliases": sorted(alias_keys(name_part, full_id)),
                "context": m.get("context_length") or None,
                "max_output": top.get("max_completion_tokens") or None,
                "modalities": {"input": modalities_in or ["text"], "output": modalities_out},
                "capabilities": caps,
                "price": price if len(price) > 1 else None,
                "kind": str(m.get("kind") or "chat"),
                "description": (m.get("description") or None),
                "sources": ["openrouter"],
            }
        )
    print(f"  openrouter: {len(out)} models", flush=True)
    return out


def fetch_mimi() -> list[dict[str, Any]]:
    import os

    url = os.environ.get("MIMI_MODELS_URL", "https://mimi.wrun.cc/v1/models")
    headers = {}
    if os.environ.get("MIMI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["MIMI_API_KEY"]
    data = http_get_json(url, timeout=30, headers=headers)
    out: list[dict[str, Any]] = []
    for m in data.get("data", []) or []:
        raw_id = str(m.get("id") or "").strip()
        if not raw_id:
            continue
        modalities_in = m.get("input_modalities")
        if not isinstance(modalities_in, list) or not modalities_in:
            modalities_in = ["text"]
            if m.get("supports_images"):
                modalities_in.append("image")
            if m.get("supports_video"):
                modalities_in.append("video")
        caps: dict[str, Any] = {"temperature": True}
        if m.get("supports_tool_call"):
            caps["tool_call"] = True
        if m.get("supports_reasoning"):
            caps["reasoning"] = True
        if "image" in modalities_in:
            caps["vision"] = True
        out.append(
            {
                "id": normalize_id(raw_id),
                "vendor": "w-run",
                "name": str(m.get("name") or raw_id),
                "aliases": sorted(alias_keys(raw_id)),
                "context": m.get("max_input_tokens") or None,
                "max_output": m.get("max_output_tokens") or None,
                "modalities": {"input": modalities_in, "output": ["text"]},
                "capabilities": caps,
                "price": {"unit": USD_UNIT, "input": 0.0, "output": 0.0},
                "kind": str(m.get("kind") or "chat"),
                "description": None,
                "sources": ["mimi"],
            }
        )
    print(f"  mimi: {len(out)} models", flush=True)
    return out


def load_manual() -> dict[str, dict[str, Any]]:
    if not MANUAL_PATH.exists():
        return {}
    try:
        raw = json.loads(MANUAL_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  manual: 读取失败 {exc}", file=sys.stderr, flush=True)
        return {}
    models = raw.get("models") or {}
    print(f"  manual: {len(models)} overrides", flush=True)
    return {normalize_id(k): v for k, v in models.items() if v}


# ── 合并 ───────────────────────────────────────────────────────────────────


def merge(base: dict[str, dict[str, Any]], incoming: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按优先级合并：已存在的字段不覆盖，只补空；aliases 取并集；sources 累加。"""
    for model in incoming:
        mid = model.get("id")
        if not mid:
            continue
        cur = base.get(mid)
        if cur is None:
            base[mid] = dict(model)
            continue
        for key, value in model.items():
            if key in ("id", "aliases", "sources"):
                continue
            if value in (None, "", {}, []) and key not in ("price",):
                continue
            if cur.get(key) in (None, "", {}, []):
                cur[key] = value
        cur["aliases"] = sorted(set(cur.get("aliases") or []) | set(model.get("aliases") or []))
        cur["sources"] = sorted(set(cur.get("sources") or []) | set(model.get("sources") or []))
    return base


def apply_manual(models: dict[str, dict[str, Any]], overrides: dict[str, dict[str, Any]]) -> None:
    """人工标称：直接覆盖（含置 None/False 的显式关闭）。"""
    for mid, patch in overrides.items():
        target = models.get(mid)
        if target is None:
            # 允许手工新增官方没有的模型
            target = {
                "id": mid,
                "vendor": patch.get("vendor") or "w-run",
                "name": patch.get("name") or mid,
                "aliases": sorted(alias_keys(mid)),
                "context": None,
                "max_output": None,
                "modalities": None,
                "capabilities": {},
                "price": None,
                "kind": "chat",
                "description": None,
                "sources": [],
            }
            models[mid] = target
        for key, value in patch.items():
            if key in ("aliases", "sources"):
                continue
            if key == "capabilities" and isinstance(value, dict):
                merged_caps = dict(target.get("capabilities") or {})
                merged_caps.update(value)
                target["capabilities"] = merged_caps
                continue
            target[key] = value
        target["sources"] = sorted(set(target.get("sources") or []) | {"manual"})


# ── 输出 ───────────────────────────────────────────────────────────────────


def build_index(models: dict[str, dict[str, Any]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for mid, model in models.items():
        keys = set(alias_keys(mid))
        for a in model.get("aliases") or []:
            keys |= alias_keys(a)
        for k in keys:
            if k and k not in index:
                index[k] = mid
    return index


def write_outputs(output_dir: Path, models: dict[str, dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    vendors: dict[str, dict[str, Any]] = {}
    for model in models.values():
        vid = model.get("vendor") or "unknown"
        entry = vendors.setdefault(vid, {"id": vid, "name": vid, "model_count": 0})
        entry["model_count"] += 1

    ordered_models = [models[k] for k in sorted(models.keys())]
    (output_dir / "models.json").write_text(
        json.dumps(
            {"version": VERSION, "generated_at": now, "count": len(ordered_models), "models": ordered_models},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    vendor_list = [vendors[k] for k in sorted(vendors.keys())]
    (output_dir / "vendors.json").write_text(
        json.dumps(
            {"version": VERSION, "generated_at": now, "count": len(vendor_list), "vendors": vendor_list},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    index = build_index(models)
    (output_dir / "index.json").write_text(
        json.dumps({"version": VERSION, "generated_at": now, "index": index}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Build models.json / vendors.json / index.json")
    parser.add_argument("-o", "--output-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--only", default="", help="逗号分隔的数据源，如 openrouter,mimi,manual")
    args = parser.parse_args()

    selected = {s.strip() for s in args.only.split(",") if s.strip()}
    run = lambda name: (not selected) or (name in selected)  # noqa: E731

    print("=" * 60)
    print(f"  model-data build v{VERSION}")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    models: dict[str, dict[str, Any]] = {}
    # 优先级从低到高依次合入：后合的在"补空"策略下不会覆盖已有值，
    # 所以真正的优先级通过顺序体现——先合入的赢。
    if run("openrouter"):
        models = merge(models, fetch_openrouter())
    if run("mimi"):
        models = merge(models, fetch_mimi())
    if run("manual"):
        apply_manual(models, load_manual())

    write_outputs(args.output_dir, models)

    filled = sum(1 for m in models.values() if m.get("context"))
    print()
    print(f"  models: {len(models)}  (with context: {filled})")
    print(f"  output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
