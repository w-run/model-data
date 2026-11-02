#!/usr/bin/env python3
"""
厂商归一

同一个厂商在各源写法不同，不归一会导致「同一家被当成两家」：

    DeepSeek   deepseek / deepseek-ai / deepseekai / deepseek_ai / 深度求索
    智谱        zhipuai / zhipu / z-ai / zai-org / bigmodel / 智谱AI / GLM
    月之暗面    moonshotai / moonshot / moonshot-ai / kimi / 月之暗面
    阿里        alibaba / alibaba-cn / qwen / dashscope / 通义千问
    字节        bytedance / volcengine / doubao / 豆包 / seed

规则
----
1. 小写 + 去分隔符后比对（`deepseek-ai` 与 `deepseek_ai` 等价）
2. 别名表是唯一权威，脚本里不散落硬编码
3. 无法判定时**原样保留**并标 unknown，不猜
"""

from __future__ import annotations

import re
from typing import Iterable

# canonical -> 别名列表。内置表仅作**回落**：
# 实际以 data/vendors.toml 为准，那才是单一权威源，可随时增补而不改代码。
BUILTIN_ALIASES: dict[str, list[str]] = {
    "deepseek": [
        "deepseek", "deepseek-ai", "deepseekai", "deepseek_ai", "deepseek ai",
        "深度求索", "depthseek", "ds",
    ],
    "zhipuai": [
        "zhipuai", "zhipu", "z-ai", "zai", "zai-org", "z.ai", "bigmodel",
        "智谱", "智谱ai", "智谱AI", "chatglm", "glm",
    ],
    "moonshotai": [
        "moonshotai", "moonshot", "moonshot-ai", "moonshot_ai", "kimi",
        "月之暗面", "moonshotcn",
    ],
    "alibaba": [
        "alibaba", "alibaba-cn", "aliyun", "qwen", "qwq", "dashscope",
        "tongyi", "通义", "通义千问", "阿里", "阿里云", "百炼",
    ],
    "anthropic": [
        "anthropic", "anthropic-ai", "claude", "anthropicai",
    ],
    "openai": [
        "openai", "open-ai", "oai", "chatgpt", "gpt", "openai-community",
    ],
    "google": [
        "google", "gemini", "google-vertex", "google-vertex-ai", "deepmind",
        "bard", "谷歌",
    ],
    "meta": [
        "meta", "meta-llama", "meta-llm", "meta-ai", "llama", "facebook",
    ],
    "mistralai": [
        "mistralai", "mistral", "mistral-ai", "mistral_ai", "codestral",
    ],
    "xai": [
        "xai", "x-ai", "grok", "x.ai",
    ],
    "minimax": [
        "minimax", "minimax-ai", "minimaxai", "minimaxi", "abab",
    ],
    "bytedance": [
        "bytedance", "bytedance-seed", "volcengine", "doubao", "doubao-ai",
        "seed", "字节", "字节跳动", "豆包", "火山引擎", "火山",
    ],
    "tencent": [
        "tencent", "tencent-ai", "tencentarc", "hunyuan", "tencent-cloud",
        "腾讯", "混元",
    ],
    "baidu": [
        "baidu", "baidu-ai", "wenxin", "qianfan", "qianfan-ai", "ernie",
        "百度", "文心", "千帆",
    ],
    "stepfun": [
        "stepfun", "stepfun-ai", "step", "阶跃", "阶跃星辰",
    ],
    "moonshot": ["moonshot"],
    "xiaomi": [
        "xiaomi", "mimo", "小米", "xiaomimimo",
    ],
    "nvidia": [
        "nvidia", "nvidia-nim", "nvidia-ai", "nemotron",
    ],
    "cohere": [
        "cohere", "cohereforai", "cohere-ai", "coherelabs", "aya",
    ],
    "perplexity": [
        "perplexity", "perplexity-ai", "pplx", "sonar",
    ],
    "sarvam": [
        "sarvam", "sarvamai", "sarvam-ai",
    ],
    "amazon": [
        "amazon", "aws", "bedrock", "amazon-bedrock", "aws-bedrock", "nova",
    ],
    "microsoft": [
        "microsoft", "azure", "azure-cognitive-services", "phi",
    ],
    "inclusionai": [
        "inclusionai", "inclusion-ai", "ling",
    ],
    "sensenova": [
        "sensenova", "sensetime", "商汤", "日日新",
    ],
    "01ai": [
        "01ai", "yi", "零一万物", "01-ai",
    ],
    "baichuan": ["baichuan", "百川"],
    "w-run": ["w-run", "wrun", "mimi", "workbuddy", "catpaw"],
}

# 反向索引：归一化后的别名 -> canonical
_INDEX: dict[str, str] = {}
_TOML_CACHE: dict[str, dict] = {}


def load_from_toml(path: str = "data/vendors.toml") -> None:
    """
    从 vendors.toml 加载别名表，**覆盖并扩充**内置表。

    这样新增厂商或别名只需改 toml，不改代码 —— 模型信息是持续更新的，
    硬编码在代码里的别名表必然滞后。
    """
    import os
    from pathlib import Path

    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11
        import tomli as tomllib  # type: ignore

    p = Path(path)
    if not p.exists():
        return
    try:
        with open(p, "rb") as f:
            data = tomllib.load(f)
    except Exception:  # noqa: BLE001
        return

    for vid, cfg in data.items():
        aliases = [vid] + list(cfg.get("alias") or [])
        _TOML_CACHE[vid] = cfg
        for a in aliases:
            k = re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", str(a).lower())
            if k:
                _INDEX[k] = vid
        _INDEX[re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", vid.lower())] = vid


def vendor_meta(canonical: str) -> dict:
    """取厂商的元信息（name / website / api_docs / cn …）。"""
    c = canonical_vendor(canonical)
    return dict(_TOML_CACHE.get(c) or {})


def _norm_key(s: str) -> str:
    """归一键：小写、去所有非字母数字（保留中日韩文字）。"""
    s = str(s or "").strip().lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", s)


def _build_index() -> None:
    if _INDEX:
        return
    for canonical, aliases in BUILTIN_ALIASES.items():
        _INDEX[_norm_key(canonical)] = canonical
        for a in aliases:
            _INDEX[_norm_key(a)] = canonical


def canonical_vendor(raw: str) -> str:
    """
    归一厂商名。无法识别时返回小写原值（不丢信息）。

    >>> canonical_vendor("deepseek-ai")
    'deepseek'
    >>> canonical_vendor("zai-org")
    'zhipuai'
    >>> canonical_vendor("Moonshot")
    'moonshotai'
    """
    _build_index()
    key = _norm_key(raw)
    if not key:
        return "unknown"
    if key in _INDEX:
        return _INDEX[key]

    # 包含匹配：如 "deepseek-ai-chat" -> deepseek
    for alias_key, canonical in _INDEX.items():
        if len(alias_key) >= 4 and alias_key in key:
            return canonical

    return str(raw).strip().lower()


def is_known_vendor(raw: str) -> bool:
    _build_index()
    return _norm_key(raw) in _INDEX


def vendor_aliases(canonical: str) -> list[str]:
    """取某个厂商的全部别名，用于产出的 aliases 字段。"""
    _build_index()
    c = canonical_vendor(canonical)
    if c in BUILTIN_ALIASES:
        # 去重并保持可读性：去掉纯符号变体
        seen, out = set(), []
        for a in [c] + BUILTIN_ALIASES[c]:
            k = _norm_key(a)
            if k and k not in seen:
                seen.add(k)
                out.append(a)
        return out
    return [canonical]


def merge_vendor_observations(values: Iterable[str]) -> str:
    """
    多个源报了不同写法的厂商名，选举一个 canonical。

    优先能识别的；都不识别则取出现最多的原值。
    """
    from collections import Counter

    known, raw = [], []
    for v in values:
        if not v:
            continue
        if is_known_vendor(v):
            known.append(canonical_vendor(v))
        else:
            raw.append(str(v).strip().lower())

    if known:
        return Counter(known).most_common(1)[0][0]
    if raw:
        return Counter(raw).most_common(1)[0][0]
    return "unknown"
