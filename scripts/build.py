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

from .limits import DEFAULT_OUTPUT_CAP, derive_recommended, is_placeholder, vote_limits
from .sources import models_dev
from . import validate as validator
from .inherit import apply_inheritance

VERSION = "3.0.0"
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


def loose_key(slug: str) -> str:
    """
    宽松匹配键：抹平两家源对同一模型的写法差异。
        claude-sonnet-4.5  ==  claude-sonnet-4-5
        gpt-5.2            ==  gpt-5-2
        deepseek_v4.1      ==  deepseek-v4-1
    只用于跨源对齐，不写进产出（产出里的 id 仍按 OpenRouter 写法）。
    """
    s = str(slug or "").strip().lower()
    s = s.replace(".", "-").replace("_", "-")
    s = re.sub(r"-+", "-", s).strip("-")
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
                # 原始上报值。是否可信由 limits.py 的跨源共识判定，
                # 不能直接当成 max_output 用（可能是 floor(0.9*ctx) 占位）。
                "_or_context": m.get("context_length") or None,
                "_or_output": top.get("max_completion_tokens") or None,
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
                "_or_context": m.get("max_input_tokens") or None,
                "_or_output": m.get("max_output_tokens") or None,
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
            # 原始上报值：先到先得，不被低优先级源覆盖
            if key.startswith("_or_"):
                if cur.get(key) in (None, ""):
                    cur[key] = value
                continue
            if value in (None, "", {}, []) and key not in ("price",):
                continue
            if cur.get(key) in (None, "", {}, []):
                cur[key] = value
        cur["aliases"] = sorted(set(cur.get("aliases") or []) | set(model.get("aliases") or []))
        cur["sources"] = sorted(set(cur.get("sources") or []) | set(model.get("sources") or []))
    return base


def apply_manual(models: dict[str, dict[str, Any]], overrides: dict[str, dict[str, Any]]) -> None:
    """
    人工标称：优先级最高。

    limit / recommended 支持**深层合并**——只写要改的那几个字段，
    其余（如 confidence、votes）保留自动推导的结果。
    """
    for mid, patch in overrides.items():
        target = models.get(mid)
        if target is None:
            target = {
                "id": mid,
                "vendor": patch.get("vendor") or "w-run",
                "name": patch.get("name") or mid,
                "aliases": sorted(alias_keys(mid)),
                "limit": {},
                "recommended": {},
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
            if key in ("capabilities", "limit", "recommended") and isinstance(value, dict):
                merged = dict(target.get(key) or {})
                merged.update(value)
                target[key] = merged
                continue
            target[key] = value
        target["sources"] = sorted(set(target.get("sources") or []) | {"manual"})
        # 人工改了 limit 但没给 recommended → 重新推导，避免两者打架
        if "limit" in patch and "recommended" not in patch:
            rec = derive_recommended(target["limit"], output_cap=DEFAULT_OUTPUT_CAP)
            rec["output_cap"] = DEFAULT_OUTPUT_CAP
            target["recommended"] = rec


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
    parser.add_argument("--no-validate", action="store_true", help="跳过产出校验（不建议）")
    parser.add_argument("--strict", action="store_true", help="有 error 即以非零码退出")
    args = parser.parse_args()

    selected = {s.strip() for s in args.only.split(",") if s.strip()}
    run = lambda name: (not selected) or (name in selected)  # noqa: E731

    print("=" * 60)
    print(f"  model-data build v{VERSION}")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # ── 1. 收集观测（跨源共识所需的原始票）──────────────────────────
    observations: dict[str, list[dict[str, Any]]] = {}
    first_party: set[str] = models_dev.DEFAULT_FIRST_PARTY
    vendors_toml = BASE_DIR / "data" / "vendors.toml"
    if vendors_toml.exists():
        try:
            first_party = models_dev.load_first_party_from_vendors(str(vendors_toml))
            print(f"  vendors.toml: {len(first_party)} 个第一方厂商", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  vendors.toml 读取失败，用内置白名单: {exc}", file=sys.stderr, flush=True)

    if run("models_dev"):
        try:
            md_data = models_dev.fetch()
            observations = models_dev.observations(md_data, first_party)
            print(f"  models.dev: {len(md_data)} providers, {len(observations)} canonical slugs", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  models.dev: FAIL {exc}", file=sys.stderr, flush=True)

    # 宽松索引：两家源对同一模型常写成 4.5 / 4-5、gpt-5.2 / gpt-5-2。
    # 精确匹配不上的（实测 94/427）都靠这一层找回。
    loose_obs: dict[str, list[dict[str, Any]]] = {}
    for slug, rows in observations.items():
        loose_obs.setdefault(loose_key(slug), []).extend(rows)

    def lookup(mid: str) -> list[dict[str, Any]]:
        hit = observations.get(mid)
        if hit:
            return hit
        return loose_obs.get(loose_key(mid), [])

    # ── 2. 合并各源 ────────────────────────────────────────────────
    models: dict[str, dict[str, Any]] = {}
    if run("openrouter"):
        models = merge(models, fetch_openrouter())
    if run("mimi"):
        models = merge(models, fetch_mimi())

    # ── 3. 用共识结果修正 limit / recommended ──────────────────────
    now_iso = datetime.now(timezone.utc).isoformat()
    stats = {"placeholder_patched": 0, "inferred": 0, "consensus": 0, "first_party": 0}
    for mid, model in models.items():
        obs = lookup(mid)
        result = vote_limits(obs)

        or_ctx = model.pop("_or_context", None)
        or_out = model.pop("_or_output", None)
        or_placeholder = is_placeholder(or_ctx, or_out)

        ctx = result["context"] or or_ctx
        mx = result["max_output"]
        confidence = result["confidence"]

        # 共识缺失 → 回落到自报值，但要标 inferred
        if not result["context"]:
            confidence = "inferred"
        if not mx and or_out and not or_placeholder:
            mx = or_out
        if or_placeholder and result["max_output"]:
            stats["placeholder_patched"] += 1

        # 占位判定要在**最终 context** 下重做一次：
        # OpenRouter 按自己的 context 上报（262144），共识后的 context 可能
        # 不同（256000），此时 230400 恰好等于 floor(0.9*256000) —— 仍是占位。
        if mx and is_placeholder(ctx, mx):
            limit_placeholder_dropped = mx
            mx = None
            if confidence != "inferred":
                confidence = "inferred"
        else:
            limit_placeholder_dropped = None

        limit = {
            "context": ctx,
            "max_output": mx,
            "semantics": "shared",
            "confidence": confidence,
            "votes": result.get("votes", 0),
            "checked_at": now_iso[:10],
            "sources": (result.get("sources") or ["openrouter"])[:8],
        }
        if or_ctx and ctx and or_ctx != ctx:
            limit["openrouter_context"] = or_ctx
        if or_placeholder:
            limit["openrouter_placeholder_output"] = or_out
        if limit_placeholder_dropped:
            limit["dropped_placeholder_output"] = limit_placeholder_dropped

        rec = derive_recommended(limit, output_cap=DEFAULT_OUTPUT_CAP)
        rec["output_cap"] = DEFAULT_OUTPUT_CAP

        model["limit"] = limit
        model["recommended"] = rec
        stats[confidence] = stats.get(confidence, 0) + 1

    # ── 3b. 变体继承（:batch 等共用同一窗口）─────────────────────
    inherited = apply_inheritance(models)
    if inherited:
        print(f"  变体继承: {inherited} 个", flush=True)

    # ── 4. 人工标称（最高优先级，可覆盖上面一切）──────────────────
    if run("manual"):
        apply_manual(models, load_manual())

    # ── 5. 校验 ───────────────────────────────────────────────────
    violations = []
    if not args.no_validate:
        violations = validator.check_all(list(models.values()))
        ok, text = validator.report(violations)
        print()
        print(text)
        if not ok and args.strict:
            print("\n校验未通过，中止。", file=sys.stderr)
            sys.exit(1)

    write_outputs(args.output_dir, models)

    # 统计必须在所有修正（含继承、人工覆盖）之后重新数，不能沿用循环里的计数
    final = {"first_party": 0, "consensus": 0, "inferred": 0}
    for m in models.values():
        c = (m.get("limit") or {}).get("confidence")
        if c in final:
            final[c] += 1

    print()
    print(f"  models: {len(models)}")
    print(f"  confidence: first_party={final['first_party']} "
          f"consensus={final['consensus']} inferred={final['inferred']}")
    print(f"  placeholder 修正: {stats['placeholder_patched']}")
    print(f"  output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
