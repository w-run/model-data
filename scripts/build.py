#!/usr/bin/env python3
"""
模型信息库构建脚本

数据源：
  1. data/vendors.toml    — 厂商配置（name、alias、website 等）
  2. data/manual/*.json   — 本地维护的模型数据
  3. scrapers/            — 网页爬虫（百度千帆、火山引擎豆包等）
  4. OpenRouter API       — 免费获取模型详细信息（价格、上下文、模态等）
  5. SiliconFlow API      — 硅基流动中转平台（需 API Key，环境变量 SILICONFLOW_API_KEY）

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
import os
import re
import sys
import time
import urllib.error
import urllib.request
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
    """去掉 org/ 前缀，如 'qwen/qwen3-72b' -> 'qwen3-72b'，'openai/gpt-4o' -> 'gpt-4o'"""
    if "/" in raw_id:
        return raw_id.split("/", 1)[1]
    return raw_id


def http_get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> Any:
    """发送 GET 请求并返回 JSON 响应"""
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} for {url}: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error for {url}: {e.reason}") from e


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


# ─── 硅基流动 API ──────────────────────────────────────────────────────────────

def fetch_siliconflow(alias_map: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    """
    从硅基流动(SiliconFlow) API 获取模型列表

    硅基流动是中转商/聚合商，其模型来自各开源厂商。
    模型 ID 格式: org/model-name (如 deepseek-ai/DeepSeek-V4-Pro)
    Pro 版本: Pro/org/model-name (如 Pro/deepseek-ai/DeepSeek-V3.2)

    API 需要认证，密钥通过环境变量 SILICONFLOW_API_KEY 获取。

    alias_map: alias -> vendor_id 的映射
    返回: {vendor_id: [model_dict, ...]}
    """
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("  SiliconFlow: API key not found (SILICONFLOW_API_KEY env var), skipping", file=sys.stderr)
        return {}

    print("  Fetching from SiliconFlow API...", flush=True)

    try:
        data = http_get_json(
            "https://api.siliconflow.cn/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
    except Exception as e:
        print(f"  SiliconFlow fetch FAILED: {e}", file=sys.stderr)
        return {}

    raw_models = data.get("data", [])
    print(f"  SiliconFlow returned {len(raw_models)} models total", flush=True)

    # 硅基流动的 org -> vendor_id 映射（需要额外扩展）
    # 不同于 OpenRouter，硅基流动的 org 名可能和 HuggingFace/OpenRouter 不同
    sf_org_map: dict[str, str] = dict(alias_map)  # 以 alias_map 为基础
    # 补充硅基流动特有的 org 名映射
    sf_extra_mappings = {
        "qwen": "alibaba",
        "thudm": "z-ai",
        "zai-org": "z-ai",
        # "baai" 暂不映射，BAAI(北京智源研究院)是独立机构，不属于任何已有厂商
        "minimaxai": "minimax",
        "moonshotai": "moonshot",
        "stepfun-ai": "stepfun",
        "tencent": "tencent",
        "baidu": "baidu",
        "deepseek-ai": "deepseek",
        "teleai": "tencent",
        "tongyi-mai": "alibaba",
        "wan-ai": "alibaba",
        "inclusionai": "alibaba",
        "funaudiollm": "alibaba",
        "nex-agi": "stepfun",
        "kwai-kolors": "xiaomi",  # 可商不属于小米，暂映射
    }
    for org_key, vid in sf_extra_mappings.items():
        if org_key not in sf_org_map:
            sf_org_map[org_key] = vid

    result: dict[str, list[dict[str, Any]]] = {}
    now = datetime.now(timezone.utc).isoformat()
    skipped = 0

    for m in raw_models:
        model_id = m.get("id", "")
        if not model_id:
            continue

        # 解析模型 ID
        parts = model_id.split("/")
        prefix = None
        if parts[0] == "Pro" and len(parts) == 3:
            prefix = "Pro"
            org = parts[1]
            model_name = parts[2]
        elif parts[0] == "LoRA":
            continue  # 跳过 LoRA 模型
        elif len(parts) == 2:
            org = parts[0]
            model_name = parts[1]
        else:
            continue

        # 匹配厂商
        vendor_id = sf_org_map.get(org.lower())
        if not vendor_id:
            skipped += 1
            continue

        # 标准化模型 ID
        base_id = normalize_id(model_name)

        entry: dict[str, Any] = {
            "id": base_id,
            "vendor_id": vendor_id,
            "source": "siliconflow",
            "fetched_at": now,
            "name": model_name,
            "open_weights": True,
        }

        # 别名
        model_aliases: list[str] = []
        seen_aliases: set[str] = {base_id}

        # Pro 版本别名
        if prefix == "Pro":
            pro_alias = f"pro-{base_id}"
            if pro_alias not in seen_aliases:
                model_aliases.append(pro_alias)
                seen_aliases.add(pro_alias)

        # 硅基流动原始 ID 作为别名
        sf_alias = normalize_id(model_id)
        if sf_alias not in seen_aliases:
            model_aliases.append(sf_alias)
            seen_aliases.add(sf_alias)

        if model_aliases:
            entry["model_aliases"] = model_aliases

        # 模态推断（从模型名推断）
        name_lower = model_name.lower()
        if any(kw in name_lower for kw in ["image", "flux", "sdxl", "kolors", "stable-diffusion"]):
            entry["modalities"] = {"input": ["text"], "output": ["image"]}
        elif any(kw in name_lower for kw in ["video", "i2v", "t2v", "wan2"]):
            entry["modalities"] = {"input": ["text", "image"], "output": ["video"]}
        elif any(kw in name_lower for kw in ["tts", "speech", "asr", "cosyvoice", "sensevoice", "moss-ttsd"]):
            entry["modalities"] = {"input": ["text"], "output": ["audio"]}
        elif any(kw in name_lower for kw in ["vl", "vision", "4.5v"]):
            entry["modalities"] = {"input": ["text", "image"], "output": ["text"]}
        elif any(kw in name_lower for kw in ["embed", "rerank", "bge-"]):
            entry["modalities"] = {"input": ["text"], "output": ["text"]}
        else:
            entry["modalities"] = {"input": ["text"], "output": ["text"]}

        # 能力推断
        if not any(kw in name_lower for kw in ["embed", "rerank", "bge-",
                                                 "image", "flux", "sdxl", "kolors",
                                                 "video", "i2v", "t2v", "wan2",
                                                 "tts", "speech", "asr", "cosyvoice", "sensevoice"]):
            caps: dict[str, Any] = {"tool_call": True, "temperature": True}
            if any(kw in name_lower for kw in ["r1", "reasoning", "z1", "thinking"]):
                caps["reasoning"] = True
            entry["capabilities"] = caps

        result.setdefault(vendor_id, []).append(entry)

    if skipped:
        print(f"    Skipped {skipped} models (no vendor mapping)", flush=True)

    # 统计
    for vid in result:
        print(f"    {vid}: {len(result[vid])} models from SiliconFlow")

    return result


def run_scrapers(vendor_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    """运行网页爬虫"""
    result: dict[str, list[dict[str, Any]]] = {}
    try:
        from scrapers import run_all_scrapers, SCRAPERS
        if not SCRAPERS:
            return result

        target_ids = [vid for vid in SCRAPERS.keys() if vid in vendor_ids]
        if not target_ids:
            return result

        print("  Scraping vendor doc pages...", flush=True)
        scraper_data = run_all_scrapers(target_ids)
        for vid, models in scraper_data.items():
            if models:
                for m in models:
                    m["vendor_id"] = vid
                    m["source"] = "scraper"
                result[vid] = models
    except Exception as e:
        print(f"  Scraper error: {e}", file=sys.stderr)

    return result


def run_price_scrapers(vendor_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """运行价格爬虫，返回 {vendor_id: {model_id: price_data}}"""
    result: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        from scrapers import run_all_price_scrapers, SCRAPERS
        if not SCRAPERS:
            return result

        target_ids = [vid for vid in SCRAPERS.keys() if vid in vendor_ids]
        if not target_ids:
            return result

        print("  Scraping vendor price pages...", flush=True)
        price_data = run_all_price_scrapers(target_ids)
        result.update(price_data)
    except Exception as e:
        print(f"  Price scraper error: {e}", file=sys.stderr)

    return result


def fetch_openrouter(alias_map: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    """
    从 OpenRouter API 抓取模型信息（免费，无需 API Key）

    alias_map: alias -> vendor_id 的映射
    返回: {vendor_id: [model_dict, ...]}
    """
    print("  Fetching from OpenRouter...", flush=True)

    try:
        data = http_get_json("https://openrouter.ai/api/v1/models", timeout=60)
    except Exception as e:
        print(f"  OpenRouter fetch FAILED: {e}", file=sys.stderr)
        return {}

    raw_models = data.get("data", [])
    print(f"  OpenRouter returned {len(raw_models)} models total", flush=True)

    result: dict[str, list[dict[str, Any]]] = {}
    now = datetime.now(timezone.utc).isoformat()

    for m in raw_models:
        model_id = m.get("id", "")
        if "/" not in model_id:
            continue

        org = model_id.split("/")[0]
        if org == "openrouter" or org.startswith("~"):
            continue

        # 大小写无关匹配：将 org 转小写后查 alias_map
        vendor_id = alias_map.get(org.lower())
        if not vendor_id:
            continue

        # 模型 ID（去掉 org/ 前缀）
        raw_model_id = model_id.split("/", 1)[1]
        # 检查 :free / :extended 等后缀
        has_suffix = ":" in raw_model_id
        base_model_id = raw_model_id.split(":")[0]

        # 模型 name（去掉 "Org: " 前缀和 " (free)" / " (extended)" 后缀）
        raw_name = m.get("name", "")
        if ": " in raw_name:
            raw_name = raw_name.split(": ", 1)[1]
        # 去掉 (free) / (extended) 等括号后缀
        raw_name = re.sub(r"\s*\((free|extended)\)\s*$", "", raw_name, flags=re.IGNORECASE)

        entry: dict[str, Any] = {
            "id": base_model_id,
            "vendor_id": vendor_id,
            "source": "openrouter",
            "fetched_at": now,
            "name": raw_name,
        }

        # 模型 alias（从 canonical_slug、hugging_face_id 和 :suffix 变体提取，去掉厂商前缀）
        model_aliases: list[str] = []
        seen_aliases: set[str] = {base_model_id.lower()}

        # 带后缀的完整 id 作为 alias（如 gpt-4o-mini:free -> alias "gpt-4o-mini:free"）
        if has_suffix:
            suffix_alias = raw_model_id.lower()
            if suffix_alias not in seen_aliases:
                model_aliases.append(suffix_alias)
                seen_aliases.add(suffix_alias)

        # canonical_slug
        cs = m.get("canonical_slug", "") or ""
        if cs:
            cs_stripped = strip_org_prefix(cs)
            cs_normalized = normalize_id(cs_stripped)
            if cs_normalized and cs_normalized not in seen_aliases:
                model_aliases.append(cs_normalized)
                seen_aliases.add(cs_normalized)

        # hugging_face_id
        hf = m.get("hugging_face_id", "") or ""
        if hf:
            hf_stripped = strip_org_prefix(hf)
            hf_normalized = normalize_id(hf_stripped)
            if hf_normalized and hf_normalized not in seen_aliases:
                model_aliases.append(hf_normalized)
                seen_aliases.add(hf_normalized)

        if model_aliases:
            entry["model_aliases"] = model_aliases

        # 上下文长度
        if "context_length" in m:
            entry["context_length"] = m["context_length"]

        # 最大输出 token 数（来自 top_provider）
        top_provider = m.get("top_provider", {})
        if top_provider:
            max_completion = top_provider.get("max_completion_tokens")
            if max_completion and isinstance(max_completion, (int, float)):
                entry["max_output_tokens"] = int(max_completion)

        # 模态
        arch = m.get("architecture", {})
        if arch:
            modality = arch.get("modality", "")
            if modality and "->" in modality:
                parts = modality.split("->")
                input_mods = [x.strip() for x in parts[0].split("+")]
                output_mods = [x.strip() for x in parts[1].split("+")]
                entry["modalities"] = {"input": input_mods, "output": output_mods}

        # 价格（$/token -> $/1M tokens，统一为 USD/Mtok）
        pricing = m.get("pricing", {})
        if pricing:
            price: dict[str, Any] = {"unit": USD_UNIT}
            for price_key, entry_key in [
                ("prompt", "input"),
                ("completion", "output"),
                ("input_cache_read", "cache_input"),
                ("input_cache_write", "cache_write"),
                ("image", "image_input"),
                ("reasoning", "reasoning"),
                ("web_search", "web_search"),
            ]:
                val = pricing.get(price_key)
                if val and val != "0":
                    try:
                        price[entry_key] = round(float(val) * 1_000_000, 6)
                    except (ValueError, TypeError):
                        pass
            if len(price) > 1:  # 仅有 unit 不算有效价格
                entry["price"] = price

        # 能力
        supported_params = m.get("supported_parameters", [])
        if supported_params:
            capabilities: dict[str, Any] = {}
            if "tools" in supported_params or "tool_choice" in supported_params:
                capabilities["tool_call"] = True
            if "reasoning" in supported_params or "include_reasoning" in supported_params:
                capabilities["reasoning"] = True
            if "structured_outputs" in supported_params or "response_format" in supported_params:
                capabilities["structured_output"] = True
            if "temperature" in supported_params:
                capabilities["temperature"] = True
            if capabilities:
                entry["capabilities"] = capabilities

        # 知识截止
        if "knowledge_cutoff" in m:
            entry["knowledge_cutoff"] = m["knowledge_cutoff"]

        # 创建时间
        if "created" in m and isinstance(m["created"], (int, float)):
            entry["created_at"] = datetime.fromtimestamp(m["created"], tz=timezone.utc).isoformat()

        # 描述
        if "description" in m and m["description"]:
            entry["description"] = m["description"][:500]

        result.setdefault(vendor_id, []).append(entry)

    # 统计
    for vid in result:
        print(f"    {vid}: {len(result[vid])} models from OpenRouter")

    return result


# ─── 数据合并 ──────────────────────────────────────────────────────────────────

def merge_models(*sources: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """
    合并多个渠道的模型数据

    合并策略（按优先级）：
    1. OpenRouter — 最详细（价格、上下文、模态、能力），优先
    2. Scraper — 官方文档爬虫，补充上下文/输出限制等
    3. Manual — 本地维护，兜底

    对于同一模型 ID：
    - 优先使用高优先级渠道的数据
    - 低优先级渠道只补充高优先级没有的字段
    - alias 列表会合并所有来源（去重）
    - price 字段会合并（高优先级优先，低优先级补充null字段）
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
                                # 高优先级优先，低优先级补充null字段
                                for pk, pv in value.items():
                                    if pk == "unit":
                                        continue  # unit 不覆盖
                                    if pk not in existing_price or existing_price[pk] is None:
                                        existing_price[pk] = pv
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

    # name: 去掉厂商前缀（如 "OpenAI: GPT-4o" -> "GPT-4o"）
    raw_name = model.get("name", model["id"])
    if ": " in raw_name:
        raw_name = raw_name.split(": ", 1)[1]

    m: dict[str, Any] = {
        "id": model_id,
        "vendor_id": vendor_id.lower(),
        "name": raw_name,
    }

    # 模型 alias（不含厂商前缀，全部小写，去重，不含 id 本身）
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

    # 上下文限制（统一格式）
    limit: dict[str, Any] = {}
    # 多种来源的上下文长度字段
    if "context_length" in model:
        limit["context"] = model["context_length"]
    if "max_input" in model:
        limit["input"] = model["max_input"]
    elif "input_token_limit" in model:
        limit["context"] = model["input_token_limit"]
    # 输出限制（多种来源字段）
    if "max_output" in model:
        limit["output"] = model["max_output"]
    elif "max_output_tokens" in model:
        limit["output"] = model["max_output_tokens"]
    elif "output_token_limit" in model:
        limit["output"] = model["output_token_limit"]
    # 已经是标准格式的 limit
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

    # 1. 从模型数据中的 price 字段提取（OpenRouter 来源）
    if "price" in model:
        price = dict(model["price"])

    # 2. 从爬虫价格数据中补充
    # 优先按 model_id 精确匹配，再尝试 alias 匹配，再尝试 -preview 变体
    scraper_price = None
    if price_data:
        # 候选匹配 key 列表（按优先级排序）
        candidate_keys = [model_id]
        # 加入 alias
        for alias_raw in model.get("model_aliases", []):
            a = normalize_id(alias_raw) if "/" in str(alias_raw) else str(alias_raw).lower()
            if a and a not in candidate_keys:
                candidate_keys.append(a)
        # 加入 -preview 变体
        preview_variants = []
        for key in list(candidate_keys):
            if key.endswith("-preview"):
                preview_variants.append(key[:-len("-preview")])
            else:
                preview_variants.append(f"{key}-preview")
        candidate_keys.extend(v for v in preview_variants if v not in candidate_keys)

        # 加入上下文长度后缀变体（价格表中可能省略 -32k/-128k 等后缀）
        ctx_variants = []
        for key in list(candidate_keys):
            # 去掉末尾的 -数字k 上下文长度后缀
            ctx_removed = re.sub(r"-(?:\d+)k$", "", key)
            if ctx_removed != key and ctx_removed not in candidate_keys:
                ctx_variants.append(ctx_removed)
        candidate_keys.extend(ctx_variants)

        for key in candidate_keys:
            if key in price_data:
                scraper_price = price_data[key]
                break

    if scraper_price:
        # 确定单位（中国厂商用 CNY，国外用 USD）
        is_chinese_vendor = vendor_id in ("baidu", "bytedance", "alibaba", "moonshot", "z-ai", "xiaomi", "tencent", "stepfun")
        unit = CNY_UNIT if is_chinese_vendor else USD_UNIT

        if not price:
            price = {"unit": unit}

        # 补充爬虫价格数据
        for key in ["input", "output", "cache_input", "cache_output", "web_search", "completion", "reasoning"]:
            if key in scraper_price:
                if key not in price or price.get(key) is None:
                    price[key] = scraper_price[key]

        # 确保单位正确
        if "unit" not in price:
            price["unit"] = unit

    # 标准化价格字段：确保所有字段都存在（null 表示不支持或不详）
    if price:
        # 判断单位
        is_cny = price.get("unit") == CNY_UNIT
        unit = CNY_UNIT if is_cny else USD_UNIT

        standardized_price: dict[str, Any] = {"unit": unit}
        for key in ["input", "output", "cache_input", "cache_output", "web_search", "completion", "reasoning"]:
            val = price.get(key)
            if val is not None:
                standardized_price[key] = val
            # 不设置为 null，只在有数据时才输出该字段

        # 只有有实际价格数据时才输出
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
    use_openrouter: bool = True,
    use_scraper: bool = True,
    use_manual: bool = True,
    use_siliconflow: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    构建 vendors.json 和 models.json 的数据

    返回: (vendors_list, models_list)
    """
    # 1. 加载厂商配置
    print("Loading vendors config...", flush=True)
    vendors_config = load_vendors()
    vendor_ids = set(vendors_config.keys())
    print(f"  Found {len(vendor_ids)} vendors")

    # 2. 构建 alias -> vendor_id 映射（用于 OpenRouter 匹配）
    # 同时将 vendor_id 本身也加入映射，这样 org==vendor_id 时也能匹配
    # 所有 key 统一转小写，确保大小写无关匹配
    alias_map: dict[str, str] = {}
    for vid, cfg in vendors_config.items():
        alias_map[vid.lower()] = vid  # vendor_id 自身也加入映射
        for alias in cfg.get("alias", []):
            alias_map[alias.lower()] = vid

    # 3. 从各渠道获取模型数据
    # 优先级：OpenRouter > SiliconFlow > Scraper > Manual
    sources: list[dict[str, list[dict[str, Any]]]] = []

    if use_openrouter:
        try:
            or_data = fetch_openrouter(alias_map)
            sources.append(or_data)
        except Exception as e:
            print(f"  OpenRouter error: {e}", file=sys.stderr)

    if use_siliconflow:
        try:
            sf_data = fetch_siliconflow(alias_map)
            if sf_data:
                sources.append(sf_data)
        except Exception as e:
            print(f"  SiliconFlow error: {e}", file=sys.stderr)

    if use_scraper:
        scraper_data = run_scrapers(vendor_ids)
        if scraper_data:
            sources.append(scraper_data)

    if use_manual:
        manual_data = load_manual_data(vendor_ids)
        if manual_data:
            sources.append(manual_data)

    # 4. 合并数据
    print("\nMerging data from all channels...", flush=True)
    merged = merge_models(*sources)

    # 5. 获取价格数据（从爬虫）
    price_data_all: dict[str, dict[str, dict[str, Any]]] = {}
    if use_scraper:
        price_data_all = run_price_scrapers(vendor_ids)

    # 6. 构建输出
    print("\nBuilding output...", flush=True)
    vendors: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    seen_model_ids: set[str] = set()

    for vendor_id in sorted(vendors_config.keys()):
        cfg = vendors_config[vendor_id]
        vendor_models = merged.get(vendor_id, [])
        model_count = len(vendor_models)

        # 构建厂商条目
        vendor_entry = build_vendor_entry(vendor_id, cfg, model_count)
        vendors.append(vendor_entry)

        # 获取该厂商的价格数据
        vendor_price_data = price_data_all.get(vendor_id, {})

        # 构建模型条目
        for model in vendor_models:
            model_entry = build_model_entry(model, vendor_id, vendor_price_data)
            mid = model_entry["id"]

            # 处理 model_id 重复
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
    parser.add_argument("--no-scraper", action="store_true", help="Skip doc page scrapers")
    parser.add_argument("--no-manual", action="store_true", help="Skip manual data")
    parser.add_argument("--no-siliconflow", action="store_true", help="Skip SiliconFlow API fetch")
    parser.add_argument("--openrouter-only", action="store_true", help="Only fetch from OpenRouter")
    parser.add_argument("--scraper-only", action="store_true", help="Only run scrapers")
    parser.add_argument("--manual-only", action="store_true", help="Only load manual data")
    parser.add_argument("--siliconflow-only", action="store_true", help="Only fetch from SiliconFlow")
    parser.add_argument("--output-dir", "-o", type=Path, default=BASE_DIR, help="Output directory")

    args = parser.parse_args()

    if args.openrouter_only:
        use_or, use_scraper, use_manual, use_sf = True, False, False, False
    elif args.scraper_only:
        use_or, use_scraper, use_manual, use_sf = False, True, False, False
    elif args.manual_only:
        use_or, use_scraper, use_manual, use_sf = False, False, True, False
    elif args.siliconflow_only:
        use_or, use_scraper, use_manual, use_sf = False, False, False, True
    else:
        use_or = not args.no_openrouter
        use_scraper = not args.no_scraper
        use_manual = not args.no_manual
        use_sf = not args.no_siliconflow

    print("=" * 60)
    print("  Model Info Builder")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Channels: OpenRouter={'ON' if use_or else 'OFF'}  SiliconFlow={'ON' if use_sf else 'OFF'}  Scraper={'ON' if use_scraper else 'OFF'}  Manual={'ON' if use_manual else 'OFF'}")
    print("=" * 60)
    print()

    vendors, models = build(
        use_openrouter=use_or,
        use_scraper=use_scraper,
        use_manual=use_manual,
        use_siliconflow=use_sf,
    )

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
