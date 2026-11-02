#!/usr/bin/env python3
"""
模型身份对齐 —— 以外部权威元数据为准，不做字符串猜测

为什么不能用字符串规则
----------------------
我最初按「去掉尾部日期」推断快照属于哪个基准名，这个思路是错的：

    deepseek-v4-flash    v4     ← 与下面**不是**同一个模型
    deepseek-v4.1-flash  v4.1   ← 版本号不同就是不同模型

    claude-sonnet-4-5    ← 与 claude-sonnet-4-6 / -5 都不同
    gpt-5.2              ← 与 gpt-5.3 不同

版本号是模型的**主要标识**，不是「时间点标签」。官方不并存新旧版本，
但中转商常同时提供，于是同一族里出现多个版本 —— 它们是并列的独立模型。

唯一的例外是「同一个模型的不同日期快照」，例如：
    deepseek-v4.1-flash-1017   ==  deepseek-v4.1-flash-20261017   （月日简写 vs 完整日期）
    claude-sonnet-4-5-20250929 ==  claude-sonnet-4-5 的最新快照

权威来源（均为机器可读，随上游更新）
------------------------------------
1. **OpenRouter `canonical_slug`** —— 直接声明「这个 id 的标准名是 X」。
   实测 433 个模型里 316 个 canonical != id，77 组多对一。
2. **models.dev `canonical_model_id`** —— 同义。
3. LiteLLM 的 provider 前缀同义映射。

用法：优先用外部 canonical，拿不到时才做**保守**的日期折叠，
且只折叠「同一基准名 + 日期形态不同」的情况。
"""

from __future__ import annotations

import re
from typing import Iterable

# 日期快照的几种写法。只处理日期，绝不动版本号。
_DATE_TAIL = re.compile(r"^(?P<base>.+?)[-_.]?(?P<date>\d{8}|\d{4}|\d{2}-\d{4}|\d{4}-\d{2}-\d{2})$")

# 变体后缀：属于**同一模型的不同形态**（计费/队列），可与主体共享证据
VARIANT_SUFFIXES = (":batch", ":free", ":extended", ":nitro", ":floor", ":online",
                    "-free", "-fast", "-flex", "-spot")

# 明确是**不同模型**的后缀，不折叠
DISTINCT_MARKERS = ("vision", "vl", "audio", "tts", "asr", "omni", "embed",
                    "rerank", "moderation", "highspeed", "uncensored", "abliterated",
                    "heretic", "cyber", "marathon", "preview", "exp", "thinking",
                    "reasoning", "distill", "coder", "math")


def strip_variant(model_id: str) -> str | None:
    """
    去掉变体后缀（:batch / :free / -fast …）。返回 None 表示没有变体后缀。

    注意 -fast / -flex / -spot 是同一模型的不同服务档，窗口一致，
    但**价格不同**，所以要保留各自条目，只共享容量证据。
    """
    s = str(model_id or "").lower()
    for suf in VARIANT_SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)]
    return None


def date_base(model_id: str) -> str | None:
    """
    去掉日期快照尾巴，返回基准名。返回 None 表示不是日期快照。

    只认日期形态，不会把 `-4-5` 之类的版本段误当成日期：
        claude-opus-4-5-20251101 → claude-opus-4-5   ✅
        gpt-4o-mini-2024-07-18   → gpt-4o-mini        ✅
        claude-sonnet-4-5        → None               ✅（4-5 是版本）
        gpt-5.2                  → None               ✅
    """
    s = str(model_id or "").lower()
    if any(m in s for m in DISTINCT_MARKERS):
        return None

    m = _DATE_TAIL.match(s)
    if not m:
        return None
    base, date = m.group("base"), m.group("date")

    if not base:
        return None

    # 4 位「年月」型日期（-1017 / -0731 / -2512）与版本号极易混淆：
    #   claude-opus-4-5  → 这里的 "4-5" 不是日期
    # 但 8 位 YYYYMMDD / YYYY-MM-DD / MM-YYYY 形态足够明确。
    # 判据：4 位数字只有当它不像版本段时才认（版本段一般是 1-2 位数字）。
    if len(date) == 4 and base[-1].isdigit():
        # 基准以数字结尾 = 版本段结尾（如 ...-4-5），此时 4 位数字更可能是版本续接
        return None

    # 基准名不能是纯数字残片
    if re.fullmatch(r"[\d.\-_]+", base):
        return None
    return base


class AliasIndex:
    """
    由外部权威数据构建的别名索引。

    数据来源（都可自动获取，无需手写）：
      OpenRouter  models[].id 与 canonical_slug
      models.dev  models[].id 与 canonical_model_id
    """

    def __init__(self) -> None:
        # 任意写法 -> 标准 slug
        self.to_canonical: dict[str, str] = {}
        # 标准 slug -> 所有已知写法
        self.members: dict[str, set[str]] = {}
        self.sources: dict[str, set[str]] = {}

    # ── 构建 ────────────────────────────────────────────────────
    def add(self, raw_id: str, canonical: str | None, source: str) -> None:
        """登记一组 id→canonical 关系。"""
        key = norm_key(raw_id)
        if not key:
            return
        target = norm_key(canonical) if canonical else key
        if not target:
            target = key
        self.to_canonical[key] = target
        self.members.setdefault(target, set()).add(key)
        self.sources.setdefault(key, set()).add(source)

    def add_openrouter(self, data: dict) -> int:
        n = 0
        for m in (data or {}).get("data", []) or []:
            mid = str(m.get("id") or "")
            if not mid or mid.startswith("~"):
                continue
            slug = mid.split("/", 1)[1] if "/" in mid else mid
            canon = m.get("canonical_slug")
            if canon:
                canon = str(canon).split("/", 1)[-1]
            self.add(slug, canon, "openrouter")
            n += 1
        return n

    def add_models_dev(self, data: dict) -> int:
        n = 0
        for provider, pdata in (data or {}).items():
            for doc_id, m in ((pdata or {}).get("models") or {}).items():
                if not isinstance(m, dict):
                    continue
                canon = m.get("canonical_model_id") or m.get("id") or doc_id
                slug = str(canon).split("/")[-1]
                self.add(slug, slug, "models.dev")
                self.add(doc_id.split("/")[-1], slug, "models.dev")
                n += 1
        return n

    # ── 查询 ────────────────────────────────────────────────────
    def resolve(self, model_id: str) -> str:
        """
        返回该 id 的标准 slug。

        顺序：外部 canonical → 同义键 → 自身
        """
        key = norm_key(model_id)
        if key in self.to_canonical:
            return self.to_canonical[key]
        return key

    def siblings(self, model_id: str) -> set[str]:
        """与该 id 属于同一模型的全部已知写法。"""
        return set(self.members.get(self.resolve(model_id), set()))


def norm_key(slug: str) -> str:
    """比较键：小写、小数点与短横线统一（4.5 与 4-5 视为同一写法）。"""
    s = str(slug or "").strip().lower()
    s = s.replace("_", "-").replace(".", "-")
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def build_alias_index(*datasets: tuple[str, dict]) -> AliasIndex:
    """
    从多个源构建索引。

    >>> idx = build_alias_index(("openrouter", or_data), ("models.dev", md_data))
    """
    idx = AliasIndex()
    for name, data in datasets:
        if name == "openrouter":
            idx.add_openrouter(data)
        elif name == "models.dev":
            idx.add_models_dev(data)
    return idx
