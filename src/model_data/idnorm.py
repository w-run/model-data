#!/usr/bin/env python3
"""
模型 id 规范化 —— 全库唯一口径

约定
----
1. **一律小写**
2. **版本号用小数点**：`v4-1` → `v4.1`，`claude-sonnet-4-5` → `claude-sonnet-4.5`
3. 分隔符只保留 `-`，下划线归为 `-`，连续短横线压成一个

为什么需要它
------------
同一个模型在各源写法不同，不规范化会导致证据被拆散、匹配不上：

    SiliconFlow   DeepSeek-V4.1-Flash
    OpenRouter    deepseek/deepseek-v4.1-flash
    models.dev    deepseek-v4-flash      (doc_id)
    LiteLLM       dashscope/deepseek-v4-flash

归一后统一为 `deepseek-v4.1-flash`，投票才能聚到一起。

难点：哪些「数字-数字」是版本号
------------------------------
必须区分版本与型号编号，否则误伤：

    claude-sonnet-4-5    → 4.5   ✅ 版本
    gpt-3-5-turbo        → 3.5   ✅ 版本
    v4-1                 → 4.1   ✅ 版本（段内含数字，v4 是版本段）
    llama-3-1-8b         → 保持   ❌ 3-1 后面跟 8b，是规模编号
    doubao-seed-1-6-...  → 保持   ❌ seed 是型号名，不是版本前缀
    gpt-4o-mini-2024-07  → 保持   ❌ 2024-07 是日期

判定顺序（缺一不可）
--------------------
1. 两段都必须是**短数字**（≤2 位）—— 排除 2024 这类年份
2. 该对之后必须是「版本结束位置」：结束 / 已知后缀词 / 日期段
3. 前面必须是版本前缀：开头 / 含字母的段 / 含数字的版本段(v4)
   但不能是型号名固定词（seed / vl / 8b 之类）
"""

from __future__ import annotations

import re

# 版本号之后常见的后缀词
VERSION_FOLLOWERS = {
    "turbo", "mini", "nano", "micro", "small", "medium", "large", "pro", "ultra",
    "flash", "lite", "max", "plus", "base", "instruct", "chat", "preview",
    "latest", "thinking", "reasoning", "code", "coder", "sol", "terra", "luna",
    "argon", "opus", "sonnet", "haiku", "omni", "vl", "vision", "audio", "tts",
    "exp", "fast", "flex", "air", "turbo",
}

# 型号名里的固定词：出现在这些词之后，数字对**不是**版本号
NOT_VERSION_PREFIX = {
    "seed", "vl", "a", "b", "b-a", "e", "i", "it", "instruct",
    "smol", "mo", "moe",
}

_SEP_RE = re.compile(r"-{2,}")


def canonical_id(raw: str) -> str:
    """
    把任意写法的模型 id 规范成统一口径。

    >>> canonical_id("DeepSeek-V4.1-Flash")
    'deepseek-v4.1-flash'
    >>> canonical_id("claude-sonnet-4-5")
    'claude-sonnet-4.5'
    >>> canonical_id("llama-3-1-8b")
    'llama-3-1-8b'
    """
    s = str(raw or "").strip().lower().replace("_", "-")
    s = _SEP_RE.sub("-", s).strip("-")
    if not s:
        return ""

    parts = s.split("-")
    out: list[str] = []
    i = 0
    while i < len(parts):
        cur = parts[i]
        nxt = parts[i + 1] if i + 1 < len(parts) else None

        # 情况 A：v4-1 —— 前段是「字母+数字」的版本段，后段是纯数字
        if (nxt is not None
                and _is_version_segment(cur) and nxt.isdigit() and len(nxt) <= 2
                and _at_version_end(parts, i + 2)):
            merged = f"{cur}.{nxt}"
            j = i + 2
            while (j + 1 < len(parts) and parts[j].isdigit() and len(parts[j]) <= 2
                   and _at_version_end(parts, j + 1)):
                merged += f".{parts[j]}"
                j += 1
            out.append(merged)
            i = j
            continue

        # 情况 B：4-5 → 4.5 —— 两段都是纯数字
        if (nxt is not None
                and _is_short_number(cur) and _is_short_number(nxt)
                and _at_version_prefix(parts, i)
                and _at_version_end(parts, i + 2)):
            merged = f"{cur}.{nxt}"
            j = i + 2
            # 继续吸收：4-5-1 → 4.5.1（仅当后段是短数字且仍在版本位置）
            while (j + 1 < len(parts)
                   and _is_short_number(parts[j])
                   and _at_version_end(parts, j + 1)):
                merged += f".{parts[j]}"
                j += 1
            out.append(merged)
            i = j
            continue

        out.append(cur)
        i += 1

    return "-".join(out)


def _is_version_segment(token: str) -> bool:
    """
    形如 v4 / qwen3 / gpt4 的版本段：字母开头、以 1-2 位数字结尾。
    与型号段（235b / a22b / 8b）区分：后者数字后还有字母。
    """
    return bool(re.fullmatch(r"[a-z]{1,8}\d{1,2}", token))


def _is_short_number(token: str) -> bool:
    """版本号段必须是纯数字且不超过 2 位（排除 2024 / 20251101 这类年份日期）。"""
    return token.isdigit() and 1 <= len(token) <= 2


def _at_version_prefix(parts: list[str], i: int) -> bool:
    """第 i 段之前是否构成「版本号可以开始」的语境。"""
    if i == 0:
        return True
    prev = parts[i - 1]
    if prev in NOT_VERSION_PREFIX:
        return False
    # 前一段是年份（4 位数字）时，后面的 07-18 是日期，不是版本
    if prev.isdigit() and len(prev) == 4:
        return False
    # 含字母（claude-sonnet / gpt）或含数字（v4 / qwen3）都算版本前缀
    return any(c.isalpha() or c.isdigit() for c in prev)


def _at_version_end(parts: list[str], idx: int) -> bool:
    """
    idx 处是否为「版本号结束」的位置。

    - 到结尾
    - 紧跟已知后缀词（turbo / mini / pro …）
    - 紧跟日期段（claude-opus-4-5-20251101）
    """
    if idx >= len(parts):
        return True
    nxt = parts[idx]
    if nxt in VERSION_FOLLOWERS:
        return True
    if nxt.isdigit() and len(nxt) >= 6:
        return True
    # 紧跟「数字+字母」的规模标记（8b / 27b）→ 说明前面是型号编号，不是版本
    if re.fullmatch(r"\d+[a-z]+", nxt):
        return False
    return False


def is_same_model(a: str, b: str) -> bool:
    """两个写法是否指向同一模型（规范化后比较，忽略分隔符差异）。"""
    ca = canonical_id(a).replace("-", "").replace(".", "")
    cb = canonical_id(b).replace("-", "").replace(".", "")
    return ca == cb
