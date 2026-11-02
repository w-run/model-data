#!/usr/bin/env python3
"""
厂商白名单 —— 只有明确认可的厂商才收录

为什么用白名单而不是黑名单
--------------------------
黑名单要不断追着新出现的个人模型跑：今天冒出 thedrummer，明天冒出 sao10k，
后天某个 HF 用户批量上传几十个微调。黑名单永远滞后，而噪音一旦进来就会
污染容量统计和价格分布。

白名单则相反：默认不收录，收录的都必须是我明确认得、能核对官网的厂商。
代价是可能漏收正规小厂，但那是可修复的（加一行），而脏数据不可修复。

判定链
------
  1. 命中 VENDORS 里的 canonical id 或别名 → 收录
  2. 命中 DENY（即便在白名单里也要排除的）→ 排除
  3. 名称/模态关键词过滤（heretic / uncensored / embedding …）→ 排除
  4. 其余一律不收

维护方式
--------
改 data/vendors.toml 的 `[id]` 段即可 —— 那张表同时驱动：
厂商归一、别名展开、vendors.json 产出、以及这里的收录判定。
一处维护，全线生效。
"""

from __future__ import annotations

import re
from typing import Iterable

# ── 名称/模态层过滤（先于厂商判定）──────────────────────────────────────

# 社区微调、无审查版：即便挂在白名单厂商名下也排除
NAME_DENY = re.compile(
    r"(heretic|abliterat|uncensored|unfiltered|unredacted|nsfw|amoral|"
    r"unaligned|decensored|deleted|uncut|jailbr|"
    r"dolphin|openhermes|noromaid|bagel|euryale|lzlv|goliath|"
    r"chimerax|roleplay|role-play)",
    re.I,
)

# 非对话模态：与 chat 模型混在一起会污染容量统计
NON_CHAT = re.compile(
    r"(embed|rerank|moderat|transcrib|tts\b|asr\b|whisper|"
    r"stable-?diffusion|sdxl|flux|midjourney|colpali|bge-|gte-|e5-)",
    re.I,
)

# 明确排除的厂商（即使误加入白名单）
DENY = {
    "unknown", "@cf", "cf", "cloudflare", "openrouter", "baseten",
    "novita", "deepinfra", "together", "fireworks", "groq", "hf",
}


def name_blocked(model_id: str, name: str | None = None) -> str | None:
    for text in (model_id or "", name or ""):
        m = NAME_DENY.search(text)
        if m:
            return f"社区微调/无审查（{m.group(1)}）"
    return None


def non_chat(model_id: str, kind: str | None = None) -> str | None:
    if kind and str(kind).lower() not in ("chat", "completion", "text"):
        return f"非对话模态（{kind}）"
    m = NON_CHAT.search(model_id or "")
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
) -> tuple[bool, str | None]:
    """
    是否收录。返回 (keep, reason_if_dropped)。

    白名单语义：不在名单里就不收。
    allowlist 由调用方传入 —— 实际是 data/vendors.toml 的全部键与别名。
    """
    # 1) 名称层：无审查变体一律排除，不管厂商
    reason = name_blocked(model_id, name)
    if reason:
        return False, reason

    # 2) 模态层
    reason = non_chat(model_id, kind)
    if reason:
        return False, reason

    v = str(vendor or "").strip().lower()
    allow = {str(x).strip().lower() for x in allowlist if x}

    # 3) 明确拒绝
    if not v or v in DENY:
        return False, f"厂商未收录（{v or '空'}）"

    # 4) 白名单：不在名单里就不收
    if v not in allow:
        return False, f"厂商未收录（{v}）"

    return True, None
