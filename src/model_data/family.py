#!/usr/bin/env python3
"""
模型身份：跟随主流信息源，不自己造规则

核心认识
--------
之前我一直在手写特判（剥前缀、转小数点、按代际判断……），每次修完又冒新的
重复。真相是：**主流源已经把「同一模型的不同写法」表达出来了**，直接采用即可。

OpenRouter 的 `canonical_slug` 就是这个表达
------------------------------------------
    请求用的 id                     canonical_slug
    deepseek-v4-flash          →    deepseek-v4-flash-20260423
    deepseek-v4-flash-0731     →    deepseek-v4-flash-20260731
    deepseek-v4.1-flash-1017   →    deepseek-v4.1-flash-20261017
    gpt-6.1-terra              →    gpt-6.1-terra-20261026

语义：
  - **无日期 id 是滚动别名**，canonical 指向它当时提供的那个快照
  - **每个日期快照有自己的 canonical**，是独立身份
  - 短日期 `-0731` 与完整日期 `-20260731` 是同一个快照，canonical 统一为完整形式

这正好满足「同族但不同模型」的需求：
  dsv4f 与 dsv4f-0731 归为一族（使用者只看一个），但 -0731 作为别名保留，
  单独提供时也能识别。

家族键
------
家族 = canonical 去掉尾部日期后的部分：
    deepseek-v4-flash-20260423  →  deepseek-v4-flash
    deepseek-v4-flash-20260731  →  deepseek-v4-flash
两者同族，各自保留 canonical 作为「它是哪个快照」的记录。
"""

from __future__ import annotations

import re
from typing import Any

# 完整日期快照（canonical 里的标准形式）
_FULL_DATE = re.compile(r"-(?P<date>\d{8})$")

# 短日期快照：-0731 / -2509 / -2603（4 位，前两位月份、后两位年份，或反之）
# 这类只在部分源出现，canonical 会给出完整形式，所以这里只用于兜底
_SHORT_DATE = re.compile(r"-(?P<date>\d{4})$")

_PREFIX_RE = re.compile(
    r"^(?:"
    r"apac|us-gov|us-east|us-west|us|eu|global|cn|jp|kr|au|ca|uk|me|sa|af|ap"
    r"|databricks|bedrock|vertex|azure|sagemaker|foundry|oci"
    r"|openrouter|novita|deepinfra|together|fireworks|groq"
    r"|github[-_]copilot|perplexity[-_]agent|aihubmix|edenai|llmgateway"
    r"|nano-gpt|ofox|zenmux|crossmodel|vercel|kilo|requesty|cortecs"
    r"|merge-gateway|coralbricks|iteracompute|modular|scaleway"
    r"|cerebras|baseten|cloudflare|hyperbolic|lambda"
    r")[._-]",
    re.I,
)


def strip_provider_prefix(model_id: str) -> str:
    """
    剥离区域与渠道前缀（可多层）。

        apac-anthropic-claude-opus-4-7   → anthropic-claude-opus-4-7
        databricks-claude-opus-4-8       → claude-opus-4-8
        github_copilot/claude-opus-4.8   → claude-opus-4.8
    """
    s = str(model_id or "")
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    prev = None
    while prev != s:
        prev = s
        s = _PREFIX_RE.sub("", s, count=1)
        # github_copilot 这种下划线渠道
        m = re.match(r"^([a-z][a-z0-9]*)_", s)
        if m:
            s = s[m.end():]
    return s


def family_of(canonical_slug: str) -> str:
    """
    canonical 所属的家族：去掉尾部日期。

        deepseek-v4-flash-20260423  →  deepseek-v4-flash
        gpt-6.1-terra-20261026      →  gpt-6.1-terra
        claude-opus-4-8             →  claude-opus-4-8（无日期，自成一族）
    """
    s = str(canonical_slug or "").strip().lower()
    m = _FULL_DATE.search(s)
    if m:
        return s[: m.start()]
    return s


def expand_short_date(model_id: str, known_canonicals: dict[str, str]) -> str:
    """
    短日期 id 用已登记的 canonical 展开成完整形式。

        deepseek-v4-flash-0731  →  deepseek-v4-flash-20260731

    找不到登记时原样返回 —— 不猜年份。
    """
    s = str(model_id or "").strip().lower()
    if s in known_canonicals:
        return known_canonicals[s]
    return s


def family_of_with_short_date(canonical_slug: str) -> str:
    """
    家族键的兜底：canonical 没有完整日期时，按短日期/滚动别名归族。

    起因：部分 id 只出现在单一源（models.dev），其 canonical_model_id
    就是它自己，没有指向基础名：
        deepseek-v4-flash-0423     ← 只有 models.dev 有
        deepseek-v4-flash-latest   ← 同理
    但它们显然属于 deepseek-v4-flash 家族。

    规则（保守，只处理确定的情况）：
      - 4 位短日期后缀（-0423 / -2509）
      - -latest / -latest 变体
    不做更激进的猜测 —— 版本号仍视为独立模型。
    """
    s = str(canonical_slug or "").strip().lower()
    base = family_of(s)
    if base != s:
        return base
    # 短日期：-0423 / -2509（4 位纯数字，且前面还有内容）
    m = re.search(r"-(\d{4})$", s)
    if m and len(s) > len(m.group(0)) + 3:
        return s[: m.start()]
    # 滚动别名
    m = re.search(r"-latest$", s)
    if m:
        return s[: m.start()]
    return s


def is_snapshot(model_id: str) -> bool:
    """是否带日期快照。"""
    s = str(model_id or "").strip().lower()
    return bool(_FULL_DATE.search(s) or _SHORT_DATE.search(s))


def snapshot_date(model_id: str) -> str | None:
    """取日期部分（可能是 4 位或 8 位）。"""
    s = str(model_id or "").strip().lower()
    m = _FULL_DATE.search(s) or _SHORT_DATE.search(s)
    return m.group("date") if m else None


class CanonicalIndex:
    """
    由外部源构建的 canonical 索引。

    数据来源（全部机器可读、随上游更新，无需手写规则）：
      OpenRouter  models[].id + canonical_slug
      models.dev  models[].id + canonical_model_id
    """

    def __init__(self) -> None:
        # 任意写法 → canonical（完整日期形式）
        self.canonical: dict[str, str] = {}
        # 家族 → 该家族的全部 canonical
        self.family_members: dict[str, set[str]] = {}
        # canonical → 原始写法集合
        self.spellings: dict[str, set[str]] = {}

    def add(self, raw: str, canonical: str | None, source: str) -> None:
        rid = normalize_spelling(raw)
        if not rid:
            return
        canon = normalize_spelling(canonical) if canonical else rid
        # canonical 优先取带完整日期的那一方
        prev = self.canonical.get(rid)
        if prev is None or (len(canon) > len(prev) and canon.endswith(tuple("0123456789"))):
            self.canonical[rid] = canon

        fam = family_of(self.canonical[rid])
        self.family_members.setdefault(fam, set()).add(self.canonical[rid])
        self.spellings.setdefault(self.canonical[rid], set()).add(rid)

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
        """任意写法 → canonical。"""
        rid = normalize_spelling(model_id)
        return self.canonical.get(rid, rid)

    def family(self, model_id: str) -> str:
        """任意写法 → 家族键。"""
        return family_of(self.resolve(model_id))

    def members(self, model_id: str) -> set[str]:
        """同族的全部 canonical。"""
        return set(self.family_members.get(self.family(model_id), set()))

    def aliases_of(self, model_id: str) -> set[str]:
        """同族全部写法（含快照），用于 aliases 字段。"""
        out: set[str] = set()
        for canon in self.members(model_id):
            out |= self.spellings.get(canon, set())
            out.add(canon)
        return out


def normalize_spelling(slug: str) -> str:
    """写法归一：小写、下划线/小数点归一为短横线、剥渠道前缀。"""
    s = str(slug or "").strip().lower()
    s = strip_provider_prefix(s)
    s = s.replace("_", "-").replace(".", "-")
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def build_index(*datasets: tuple[str, dict]) -> CanonicalIndex:
    """
    从多个源构建索引。

    >>> idx = build_index(("openrouter", or_data), ("models.dev", md_data))
    """
    idx = CanonicalIndex()
    for name, data in datasets:
        if name == "openrouter":
            idx.add_openrouter(data)
        elif name == "models.dev":
            idx.add_models_dev(data)
    return idx
