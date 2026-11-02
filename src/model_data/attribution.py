#!/usr/bin/env python3
"""
厂商归属推断

问题
----
不少模型没带厂商信息，导致 vendor=unknown（实测 1542/3281，47%）。
这些模型其实有线索，只是散落在 id 里：

    alibaba-qwen3-32b              ← 厂商前缀
    apac-anthropic-claude-haiku    ← 区域前缀 apac-
    amazon-nova-2-pro-preview      ← 厂商前缀
    ai21-jamba-1.5-large-v1:0      ← 厂商前缀

不能用的线索
------------
LiteLLM 的 `litellm_provider` 是**渠道**不是发布方：
    bedrock / azure / fireworks_ai / deepinfra / novita
这些都是转售平台，用它判归属会把 331 个 AWS 托管的模型全算成 Amazon。

推断顺序（每一步都要能站得住）
------------------------------
  1. 已有 vendor 观测（models.dev org / HF owned_by）→ 最可靠
  2. 剥离区域前缀（apac- / us- / eu- / global- …）
  3. 剥离渠道前缀（如 databricks-、bedrock-）
  4. 按厂商名前缀匹配（长前缀优先）
  5. 仍无法判定 → unknown（宁可留空，不要瞎猜）
"""

from __future__ import annotations

import re
from typing import Iterable

# 云区域前缀，如 apac-/us-/eu-/global-
REGION_PREFIX = re.compile(
    r"^(apac|us|eu|global|cn|jp|kr|au|ca|uk|me|sa|af|ap|us-gov|us-east|us-west)[-_]",
    re.I,
)

# 渠道前缀：这些是转售方，剥掉后才能看到真正的模型名
CHANNEL_PREFIX = re.compile(
    r"^(databricks|bedrock|azure|vertex|oci|sagemaker|foundry|"
    r"@cf|cf|openrouter|novita|deepinfra|together|fireworks|groq)-",
    re.I,
)

# 厂商名前缀 → canonical 厂商 id。匹配时按长度降序，长前缀优先。
VENDOR_PREFIX: dict[str, str] = {
    # 北美
    "anthropic": "anthropic", "claude": "anthropic",
    "openai": "openai", "chatgpt": "openai", "gpt": "openai",
    "davinci": "openai", "babbage": "openai", "curie": "openai",
    "google": "google", "gemini": "google", "gemma": "google",
    "imagen": "google", "veo": "google", "palm": "google",
    "meta": "meta", "llama": "meta",
    "mistral": "mistral", "codestral": "mistral", "ministral": "mistral",
    "pixtral": "mistral", "devstral": "mistral", "magistral": "mistral",
    "voxtral": "mistral", "mixtral": "mistral",
    "cohere": "cohere", "command": "cohere", "aya": "cohere",
    "xai": "xai", "grok": "xai",
    "nvidia": "nvidia", "nemotron": "nvidia",
    "amazon": "amazon", "titan": "amazon",
    "ai21": "ai21", "jamba": "ai21",
    "perplexity": "perplexity", "sonar": "perplexity", "pplx": "perplexity",
    # 中国
    "deepseek": "deepseek",
    "zhipuai": "zhipuai", "zhipu": "zhipuai", "glm": "zhipuai",
    "chatglm": "zhipuai", "cogview": "zhipuai", "cogvideo": "zhipuai",
    "moonshotai": "moonshotai", "moonshot": "moonshotai", "kimi": "moonshotai",
    "alibaba": "alibaba", "qwen": "alibaba", "qwq": "alibaba", "tongyi": "alibaba",
    "bytedance": "bytedance", "doubao": "bytedance", "seedream": "bytedance",
    "seedance": "bytedance", "volcengine": "bytedance",
    "tencent": "tencent", "hunyuan": "tencent",
    "baidu": "baidu", "ernie": "baidu", "wenxin": "baidu",
    "minimax": "minimax", "abab": "minimax", "hailuo": "minimax",
    "stepfun": "stepfun", "step": "stepfun",
    "xiaomi": "xiaomi", "mimo": "xiaomi",
    "sensenova": "sensenova",
    "inclusionai": "inclusionai",
    "01ai": "01ai", "baichuan": "baichuan",
    "meituan": "meituan", "longcat": "meituan",
}

_SORTED_PREFIX = sorted(VENDOR_PREFIX.items(), key=lambda kv: -len(kv[0]))

# 只匹配这些前缀时容易误伤（如 "step" 会命中 "stepladder"），
# 要求其后必须紧跟数字或短横线。
_NEEDS_BOUNDARY = {"step", "glm", "yi", "aya", "nova", "veo", "palm", "ring", "ling"}


def strip_region(model_id: str) -> str:
    """剥离开头的区域前缀，可能有多层（apac-us-...）。"""
    s = str(model_id or "")
    prev = None
    while prev != s:
        prev = s
        s = REGION_PREFIX.sub("", s, count=1)
    return s


def strip_channel(model_id: str) -> str:
    """剥离渠道/转售前缀。"""
    s = str(model_id or "")
    prev = None
    while prev != s:
        prev = s
        s = CHANNEL_PREFIX.sub("", s, count=1)
    return s


def infer_vendor(model_id: str, known: Iterable[str] = ()) -> str | None:
    """
    从模型 id 推断发布方。返回 None 表示无法判定。

    推断前先剥区域与渠道前缀，避免 apac-anthropic-... 被当成未知。
    """
    mid = str(model_id or "").lower().strip()
    if not mid:
        return None

    known_set = {str(x).lower() for x in known if x}

    base = strip_channel(strip_region(mid))

    # 1) 剥完后命中已知厂商（如 apac-anthropic-claude -> anthropic）
    for k in sorted(known_set, key=len, reverse=True):
        if base == k or base.startswith(k + "-") or base.startswith(k + "_"):
            return k

    # 2) 按厂商名前缀，长前缀优先
    for prefix, vendor in _SORTED_PREFIX:
        if base == prefix:
            return vendor
        if re.match(rf"^{re.escape(prefix)}\d", base):
            return vendor
        if base.startswith(prefix + "-") or base.startswith(prefix + "."):
            if prefix in _NEEDS_BOUNDARY:
                # step-3.7 / glm-5.3 这类可以，step-xl / step-ladder 不行
                if re.match(rf"^{re.escape(prefix)}[.-]\d", base):
                    return vendor
                continue
            return vendor

    return None


def attribution_confidence(method: str) -> float:
    """归属的可信度，供输出时标注。"""
    return {
        "observed": 0.95,   # 来自 models.dev org / HF owned_by
        "prefix": 0.70,     # 由 id 前缀推断
        "none": 0.0,
    }.get(method, 0.5)
