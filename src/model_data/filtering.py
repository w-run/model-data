#!/usr/bin/env python3
"""
收录过滤：哪些模型该进信息库

为什么需要过滤
--------------
信息库的目标是「可信赖、可用于选型的模型」。OpenRouter / LiteLLM 上
还有大量不该进来的条目：

  1. 社区微调 / 无审查版
     deepseek-v4.1-flash-1017-heretic、llama-3.1-8b-abliterated、
     qwen3.8-27b-uncensored —— 个人发布，不是官方模型，能力/价格不可预期

  2. 个人或小型发布者的一次性模型
     thedrummer/Cydonia、sao10k/L3-Euryale、failspy/... —— 角色扮演/实验性质

  3. 聚合商自造的别名条目
     @cf/...、kilo-auto/...、openrouter/... —— 是渠道包装，不是模型本身

  4. 非对话类（除非明确需要）
     图像/语音/嵌入/重排 —— 与 chat 模型混在一起会污染容量统计

判定顺序
--------
  blocklist（厂商/关键词）→ allowlist（正式厂商）→ 规则推断 → 收录

黑名单优先于白名单：即使某正式厂商名下出现了 heretic 变体，也一并排除。
"""

from __future__ import annotations

import re
from typing import Iterable

# ── 1. 名称关键词黑名单（社区微调 / 无审查）────────────────────────────────
NAME_BLOCKLIST = re.compile(
    r"(heretic|abliterat|uncensored|unfiltered|unredacted|nsfw|amoral|"
    r"unaligned|decensored|deleted|uncut|jailbr|"
    r"dolphin|openhermes|noromaid|bagel|euryale|lzlv|goliath|"
    r"chimerax|\brp\b|roleplay|role-play)",
    re.I,
)

# ── 2. 厂商黑名单（个人/微调发布者、聚合商）──────────────────────────────────
# 这些前缀/名称一律不收录。可随时增补，不需要动逻辑。
VENDOR_BLOCKLIST = {
    # 微调/无审查发布者
    "huihui-ai", "huihui", "abliteration-ai", "pqhaz", "failspy", "mlabonne",
    "soob3123", "heabsy", "cognitivecomputations", "thedrummer", "sao10k",
    "nousresearch", "eva-unit-01", "reflection", "spacexai", "publicai",
    "trustedrouter", "scaledown", "kilo-auto", "auto", "sourceful",
    "inception", "liquid", "morph", "recursive", "elorian", "prism-ml",
    "xingchen-agi", "upstage", "aion-labs", "fastino", "sakana",
    "runwayml", "assemblyai", "elevenlabs", "fish-audio", "heygen",
    # 聚合商 / 渠道包装
    "@cf", "cf", "cloudflare", "baseten", "openrouter", "novita", "novita-ai",
    "deepinfra", "together-ai", "together_ai", "fireworks", "fireworks-ai",
    "groq", "hyperbolic", "lambda", "replicate", "anyscale", "lepton",
    "kluster", "featherless", "synthetic", "inference.net", "inferencenet",
}

# ── 3. 非对话模态（默认排除）────────────────────────────────────────────────
NON_CHAT_KIND_PATTERN = re.compile(
    r"(image|video|audio|speech|tts|asr|transcrib|embed|rerank|moderat|"
    r"ocr|vision-?embed|whisper|stable-?diffusion|flux|sdxl|midjourney)",
    re.I,
)


def is_blocked_name(model_id: str, name: str | None = None) -> str | None:
    """命中名称黑名单则返回原因，否则 None。"""
    for text in (model_id or "", name or ""):
        m = NAME_BLOCKLIST.search(text)
        if m:
            return f"社区微调/无审查（{m.group(1)}）"
    return None


def is_blocked_vendor(vendor: str, allowlist: Iterable[str] = ()) -> str | None:
    """命中厂商黑名单则返回原因。allowlist 优先，防止误伤。"""
    v = str(vendor or "").strip().lower()
    if not v:
        return None
    allow = {str(x).lower() for x in allowlist}
    if v in allow:
        return None
    if v in VENDOR_BLOCKLIST:
        return f"非正式发布方（{v}）"
    return None


def is_non_chat(kind: str | None, model_id: str, name: str | None = None) -> str | None:
    """非对话类模型默认排除。"""
    if kind and str(kind).lower() not in ("chat", "completion", "text"):
        return f"非对话模态（{kind}）"
    m = NON_CHAT_KIND_PATTERN.search(model_id or "")
    if m:
        return f"非对话模型（{m.group(1)}）"
    return None


def decide(
    *,
    model_id: str,
    vendor: str = "",
    name: str | None = None,
    kind: str | None = None,
    allowlist: Iterable[str] = (),
    keep_blocked: bool = False,
) -> tuple[bool, str | None]:
    """
    是否收录。返回 (keep, reason_if_dropped)。

    顺序很重要：名称黑名单先于厂商，厂商黑名单先于白名单
    —— 正式厂商名下的 heretic 变体也要排除。
    """
    if keep_blocked:
        return True, None

    reason = is_blocked_name(model_id, name)
    if reason:
        return False, reason

    reason = is_blocked_vendor(vendor, allowlist)
    if reason:
        return False, reason

    reason = is_non_chat(kind, model_id, name)
    if reason:
        return False, reason

    return True, None
