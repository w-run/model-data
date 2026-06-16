#!/usr/bin/env python3
"""
模型信息库构建脚本

数据源（均通过 scrapers/ 模块统一接口）：
  1. data/vendors.toml    — 厂商配置（name、alias、website 等）
  2. data/manual/*.json   — 本地维护的模型数据
  3. OpenRouter           — 聚合平台，免费获取模型详细信息
  4. SiliconFlow          — 聚合平台，需 API Key（环境变量 SILICONFLOW_API_KEY）
  5. 百度千帆              — 厂商专属，爬虫获取
  6. 火山引擎豆包           — 厂商专属，爬虫获取

产出：
  - vendors.json  — 厂商信息列表
  - models.json   — 所有模型列表（通过 vendor_id 关联厂商）

命名规范：
  vendor_id / model_id 均使用小写短横线分隔式命名
  中国厂商 name 使用中文名，国外厂商使用英文名

价格规范：
  price 字段统一为:
    input:        元(USD)/百万token  输入价格
    output:       元(USD)/百万token  输出价格
    cache_input:  元(USD)/百万token  命中缓存时的输入价格（null 表示不支持或不详）
    cache_output: 元(USD)/百万token  命中缓存时的输出价格（null 表示不支持或不详）
    web_search:   元(USD)/次         联网搜索价格
    completion:   元(USD)/百万token  补齐价格（null 表示不支持或不详）
    reasoning:    元(USD)/百万token  推理/思考价格（null 表示不支持或不详）
  百度/字节价格单位为元/百万token，OpenRouter价格为USD/百万token
  用 unit 字段标识: "CNY/Mtok" 或 "USD/Mtok"
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 确保 scripts/ 目录在 Python 路径中，以正确导入 scrapers 包
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Python 3.11+ 内置 tomllib
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


# ─── 配置 ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MANUAL_DIR = DATA_DIR / "manual"
VENDORS_TOML = DATA_DIR / "vendors.toml"
LOGOS_DIR = BASE_DIR / "logos"

VERSION = "1.0.0"
CNY_UNIT = "CNY/Mtok"
USD_UNIT = "USD/Mtok"

# 数据源 ID 列表（按优先级排序，高优先级在前）
# 合并时，排在前面的数据源优先级更高
DEFAULT_SOURCE_ORDER = [
    "openrouter",          # 聚合平台 — 信息最全（价格、上下文、模态、能力）
    "siliconflow",         # 聚合平台 — 补充国内开源模型
    "volcengine-doubao",   # 厂商专属 — 豆包系列详细规格
    "baidu",               # 厂商专属 — ERNIE 系列详细规格
]


# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def normalize_id(raw_id: str) -> str:
    """
    将任意格式的 ID 转为小写短横线分隔式命名。
    保留版本号中的 '.'（如 qwen3.5、deepseek-v3.1、gpt-4.1）。
    """
    s = raw_id.replace("/", "-").replace("_", "-").replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    return s.lower()


def strip_org_prefix(raw_id: str) -> str:
    """去掉 org/ 前缀，如 'qwen/qwen3-72b' -> 'qwen3-72b'"""
    if "/" in raw_id:
        return raw_id.split("/", 1)[1]
    return raw_id


def parse_token_count(s: str) -> int | None:
    """解析 token 数量字符串，如 '128k' -> 131072, '1M' -> 1048576"""
    if not s:
        return None
    s = s.strip().lower().replace(",", "").replace("，", "")
    multipliers = {"k": 1024, "m": 1048576, "w": 10000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                return int(float(s[:-1]) * mult)
            except ValueError:
                return None
    try:
        return int(s)
    except ValueError:
        return None


# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_vendors() -> dict[str, dict[str, Any]]:
    """加载厂商配置（vendors.toml）"""
    if not VENDORS_TOML.exists():
        print(f"Error: vendors.toml not found: {VENDORS_TOML}", file=sys.stderr)
        sys.exit(1)
    with open(VENDORS_TOML, "rb") as f:
        return tomllib.load(f)


def build_alias_map(vendors_config: dict[str, dict[str, Any]]) -> dict[str, str]:
    """构建 alias -> vendor_id 的映射"""
    alias_map: dict[str, str] = {}
    for vid, cfg in vendors_config.items():
        alias_map[vid.lower()] = vid
        for alias in cfg.get("alias", []):
            alias_map[alias.lower()] = vid
    return alias_map


def load_manual_data(vendor_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    """加载本地维护的模型数据（data/manual/<vendor_id>.json）"""
    result: dict[str, list[dict[str, Any]]] = {}
    if not MANUAL_DIR.exists():
        return result

    for f in sorted(MANUAL_DIR.glob("*.json")):
        vendor_id = f.stem
        if vendor_id not in vendor_ids:
            print(f"  Warning: manual data for unknown vendor '{vendor_id}', skipping", file=sys.stderr)
            continue
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            models = data.get("models", [])
            now = datetime.now(timezone.utc).isoformat()
            for m in models:
                m["vendor_id"] = vendor_id
                m["source"] = "manual"
                m["fetched_at"] = now
            result[vendor_id] = models
            print(f"    {vendor_id}: {len(models)} models from manual data")
        except Exception as e:
            print(f"    Error loading manual data for {vendor_id}: {e}", file=sys.stderr)

    return result


# ─── 数据合并 ──────────────────────────────────────────────────────────────────

def merge_models(*sources: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """
    合并多个渠道的模型数据（取长补短）

    合并策略：
    - 排在前面的数据源优先级更高
    - 对于同一模型 ID：
      - 优先使用高优先级渠道的数据
      - 低优先级渠道只补充高优先级没有的字段
      - alias 列表会合并所有来源（去重）
      - price 字段会合并（高优先级优先，低优先级补充null字段）
      - capabilities 字典会合并（高优先级优先，低优先级补充）
    """
    merged: dict[str, dict[str, dict[str, Any]]] = {}

    for source in sources:
        for vendor_id, models in source.items():
            if vendor_id not in merged:
                merged[vendor_id] = {}

            for model in models:
                model_id = model.get("id", "")
                if not model_id:
                    continue

                if model_id not in merged[vendor_id]:
                    merged[vendor_id][model_id] = model
                else:
                    existing = merged[vendor_id][model_id]
                    for key, value in model.items():
                        if key == "model_aliases":
                            # alias 列表合并去重
                            existing_aliases = set(existing.get("model_aliases", []))
                            for a in value:
                                if a not in existing_aliases:
                                    existing.setdefault("model_aliases", []).append(a)
                                    existing_aliases.add(a)
                        elif key == "price":
                            # price 字段合并
                            existing_price = existing.get("price", {})
                            if not existing_price:
                                existing["price"] = value
                            else:
                                for pk, pv in value.items():
                                    if pk == "unit":
                                        continue  # unit 不覆盖
                                    if pk not in existing_price or existing_price[pk] is None:
                                        existing_price[pk] = pv
                        elif key == "capabilities":
                            # capabilities 字典合并
                            existing_caps = existing.get("capabilities", {})
                            if not existing_caps:
                                existing["capabilities"] = value
                            else:
                                for ck, cv in value.items():
                                    if ck not in existing_caps or existing_caps[ck] is None:
                                        existing_caps[ck] = cv
                        elif key not in existing and value:
                            existing[key] = value

    result: dict[str, list[dict[str, Any]]] = {}
    for vendor_id, models_dict in merged.items():
        result[vendor_id] = sorted(models_dict.values(), key=lambda m: m.get("id", ""))

    return result


# ─── 输出构建 ──────────────────────────────────────────────────────────────────

def build_vendor_entry(vendor_id: str, cfg: dict[str, Any], model_count: int) -> dict[str, Any]:
    """构建单个厂商的输出条目"""
    entry: dict[str, Any] = {
        "id": vendor_id.lower(),
        "name": cfg.get("name", vendor_id),
    }

    # alias（仅当非空时输出，全部小写，排除与 vendor_id 重复的项）
    vid_lower = vendor_id.lower()
    alias = [a.lower() for a in cfg.get("alias", []) if a.lower() != vid_lower]
    if alias:
        entry["alias"] = alias

    # 可选字段
    for key in ["website", "api_docs"]:
        if key in cfg and cfg[key]:
            entry[key] = cfg[key]

    # logo
    logo_path = LOGOS_DIR / f"{vendor_id}.svg"
    if logo_path.exists():
        entry["logo"] = f"logos/{vendor_id}.svg"

    return entry


def build_model_entry(model: dict[str, Any], vendor_id: str, price_data: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """构建单个模型的输出条目"""
    # id: 小写短横线分隔，不含厂商前缀
    model_id = normalize_id(model["id"])

    # name: 去掉厂商前缀
    raw_name = model.get("name", model["id"])
    if ": " in raw_name:
        raw_name = raw_name.split(": ", 1)[1]

    m: dict[str, Any] = {
        "id": model_id,
        "vendor_id": vendor_id.lower(),
        "name": raw_name,
    }

    # 模型 alias
    model_aliases: list[str] = []
    seen_alias: set[str] = {model_id}
    for alias_raw in model.get("model_aliases", []):
        a = normalize_id(alias_raw) if "/" in str(alias_raw) else str(alias_raw).lower()
        if a and a not in seen_alias:
            model_aliases.append(a)
            seen_alias.add(a)
    if model_aliases:
        m["alias"] = model_aliases

    # 直接映射的可选字段
    direct_fields = [
        "family", "knowledge_cutoff", "license", "status",
        "version", "description", "description_zh",
    ]
    for field in direct_fields:
        if field in model and model[field] is not None:
            m[field] = model[field]

    # 能力
    if "capabilities" in model:
        m["capabilities"] = model["capabilities"]

    # 上下文限制
    limit: dict[str, Any] = {}
    if "context_length" in model:
        limit["context"] = model["context_length"]
    if "max_input" in model:
        limit["input"] = model["max_input"]
    elif "input_token_limit" in model:
        limit["context"] = model["input_token_limit"]
    if "max_output" in model:
        limit["output"] = model["max_output"]
    elif "max_output_tokens" in model:
        limit["output"] = model["max_output_tokens"]
    elif "output_token_limit" in model:
        limit["output"] = model["output_token_limit"]
    if "limit" in model and isinstance(model["limit"], dict):
        for k in ["context", "input", "output"]:
            if k in model["limit"] and k not in limit:
                limit[k] = model["limit"][k]
    if limit:
        m["limit"] = limit

    # 模态
    if "modalities" in model:
        m["modalities"] = model["modalities"]

    # 价格
    price: dict[str, Any] = {}

    if "price" in model:
        price = dict(model["price"])

    # 从厂商专属数据源的价格数据中补充
    scraper_price = None
    if price_data:
        candidate_keys = [model_id]
        for alias_raw in model.get("model_aliases", []):
            a = normalize_id(alias_raw) if "/" in str(alias_raw) else str(alias_raw).lower()
            if a and a not in candidate_keys:
                candidate_keys.append(a)

        preview_variants = []
        for key in list(candidate_keys):
            if key.endswith("-preview"):
                preview_variants.append(key[:-len("-preview")])
            else:
                preview_variants.append(f"{key}-preview")
        candidate_keys.extend(v for v in preview_variants if v not in candidate_keys)

        ctx_variants = []
        for key in list(candidate_keys):
            ctx_removed = re.sub(r"-(?:\d+)k$", "", key)
            if ctx_removed != key and ctx_removed not in candidate_keys:
                ctx_variants.append(ctx_removed)
        candidate_keys.extend(ctx_variants)

        for key in candidate_keys:
            if key in price_data:
                scraper_price = price_data[key]
                break

    if scraper_price:
        is_chinese_vendor = vendor_id in ("baidu", "bytedance", "alibaba", "moonshot", "z-ai", "xiaomi", "tencent", "stepfun")
        unit = CNY_UNIT if is_chinese_vendor else USD_UNIT

        if not price:
            price = {"unit": unit}

        for key in ["input", "output", "cache_input", "cache_output", "web_search", "completion", "reasoning"]:
            if key in scraper_price:
                if key not in price or price.get(key) is None:
                    price[key] = scraper_price[key]

        if "unit" not in price:
            price["unit"] = unit

    if price:
        is_cny = price.get("unit") == CNY_UNIT
        unit = CNY_UNIT if is_cny else USD_UNIT

        standardized_price: dict[str, Any] = {"unit": unit}
        for key in ["input", "output", "cache_input", "cache_output", "web_search", "completion", "reasoning"]:
            val = price.get(key)
            if val is not None:
                standardized_price[key] = val

        has_any_price = any(k != "unit" and v is not None for k, v in standardized_price.items())
        if has_any_price:
            m["price"] = standardized_price

    # 爬虫特有字段
    for field in ["max_thinking", "generation_specs", "supported_apis"]:
        if field in model:
            m[field] = model[field]

    # 开源权重
    if "open_weights" in model:
        caps = m.get("capabilities", {})
        caps["open_weights"] = model["open_weights"]
        m["capabilities"] = caps

    # 日期
    if "release_date" in model:
        m["release_date"] = model["release_date"]
    if "last_updated" in model:
        m["last_updated"] = model["last_updated"]

    return m


# ─── 主构建逻辑 ────────────────────────────────────────────────────────────────

def build(
    source_ids: list[str] | None = None,
    use_manual: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    构建 vendors.json 和 models.json 的数据

    source_ids: 要使用的数据源 ID 列表（按优先级排序，高优先级在前）。
                None 表示使用默认顺序 DEFAULT_SOURCE_ORDER。
    返回: (vendors_list, models_list)
    """
    from scrapers import SOURCES, run_all_sources, run_all_price_sources

    if source_ids is None:
        source_ids = DEFAULT_SOURCE_ORDER

    # 1. 加载厂商配置
    print("Loading vendors config...", flush=True)
    vendors_config = load_vendors()
    vendor_ids = set(vendors_config.keys())
    print(f"  Found {len(vendor_ids)} vendors")

    # 2. 构建 alias -> vendor_id 映射
    alias_map = build_alias_map(vendors_config)

    # 3. 从各数据源获取模型数据
    print("\nFetching from data sources...", flush=True)
    sources: list[dict[str, list[dict[str, Any]]]] = []

    for sid in source_ids:
        if sid not in SOURCES:
            print(f"  Warning: unknown source '{sid}', skipping", file=sys.stderr)
            continue

        source = SOURCES[sid]()
        print(f"  [{source.source_id}] {source.name}: fetching...", end=" ", flush=True)
        try:
            data = source.fetch(alias_map)
            model_count = sum(len(v) for v in data.values())
            sources.append(data)
            print(f"OK ({model_count} models)")
        except Exception as e:
            print(f"FAIL: {e}")

    # 4. 加载手动数据
    if use_manual:
        print("\nLoading manual data...", flush=True)
        manual_data = load_manual_data(vendor_ids)
        if manual_data:
            sources.append(manual_data)

    # 5. 合并数据
    print("\nMerging data from all sources...", flush=True)
    merged = merge_models(*sources)

    # 6. 获取价格数据（从厂商专属数据源）
    print("\nFetching prices from vendor sources...", flush=True)
    # 只运行非聚合平台的数据源获取价格
    vendor_source_ids = [
        sid for sid in source_ids
        if sid in SOURCES and not SOURCES[sid].is_aggregator
    ]
    price_data_all = run_all_price_sources(vendor_source_ids)

    # 7. 构建输出
    print("\nBuilding output...", flush=True)
    vendors: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    seen_model_ids: set[str] = set()

    for vendor_id in sorted(vendors_config.keys()):
        cfg = vendors_config[vendor_id]
        vendor_models = merged.get(vendor_id, [])
        model_count = len(vendor_models)

        vendor_entry = build_vendor_entry(vendor_id, cfg, model_count)
        vendors.append(vendor_entry)

        vendor_price_data = price_data_all.get(vendor_id, {})

        for model in vendor_models:
            model_entry = build_model_entry(model, vendor_id, vendor_price_data)
            mid = model_entry["id"]

            if mid in seen_model_ids:
                mid = f"{vendor_id}-{mid}"
                if mid in seen_model_ids:
                    continue
                model_entry["id"] = mid
            seen_model_ids.add(mid)

            models.append(model_entry)

    return vendors, models


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vendors.json and models.json")
    parser.add_argument("--no-openrouter", action="store_true", help="Skip OpenRouter fetch")
    parser.add_argument("--no-scraper", action="store_true", help="Skip vendor scrapers (baidu, volcengine-doubao)")
    parser.add_argument("--no-manual", action="store_true", help="Skip manual data")
    parser.add_argument("--no-siliconflow", action="store_true", help="Skip SiliconFlow API fetch")
    parser.add_argument("--openrouter-only", action="store_true", help="Only fetch from OpenRouter")
    parser.add_argument("--scraper-only", action="store_true", help="Only run vendor scrapers")
    parser.add_argument("--manual-only", action="store_true", help="Only load manual data")
    parser.add_argument("--siliconflow-only", action="store_true", help="Only fetch from SiliconFlow")
    parser.add_argument("--output-dir", "-o", type=Path, default=BASE_DIR, help="Output directory")

    args = parser.parse_args()

    # 确定使用的数据源
    if args.openrouter_only:
        source_ids = ["openrouter"]
    elif args.scraper_only:
        source_ids = ["volcengine-doubao", "baidu"]
    elif args.manual_only:
        source_ids = []
    elif args.siliconflow_only:
        source_ids = ["siliconflow"]
    else:
        source_ids = list(DEFAULT_SOURCE_ORDER)
        if args.no_openrouter and "openrouter" in source_ids:
            source_ids.remove("openrouter")
        if args.no_siliconflow and "siliconflow" in source_ids:
            source_ids.remove("siliconflow")
        if args.no_scraper:
            source_ids = [s for s in source_ids if s not in ("volcengine-doubao", "baidu")]

    use_manual = not args.no_manual and not any([
        args.openrouter_only, args.scraper_only, args.siliconflow_only,
    ])

    source_labels = []
    if source_ids:
        from scrapers import SOURCES
        source_labels = [SOURCES[sid].name if sid in SOURCES else sid for sid in source_ids]
    if use_manual:
        source_labels.append("Manual")

    print("=" * 60)
    print("  Model Info Builder")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Sources: {' > '.join(source_labels)}")
    print("=" * 60)
    print()

    vendors, models = build(source_ids=source_ids, use_manual=use_manual)

    now = datetime.now(timezone.utc).isoformat()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 写入 vendors.json
    vendors_output = {
        "version": VERSION,
        "generated_at": now,
        "total_vendors": len(vendors),
        "vendors": vendors,
    }
    with open(output_dir / "vendors.json", "w", encoding="utf-8") as f:
        json.dump(vendors_output, f, ensure_ascii=False, indent=2)

    # 写入 models.json
    models_output = {
        "version": VERSION,
        "generated_at": now,
        "total_models": len(models),
        "models": models,
    }
    with open(output_dir / "models.json", "w", encoding="utf-8") as f:
        json.dump(models_output, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"  Vendors: {len(vendors)}")
    print(f"  Models: {len(models)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
