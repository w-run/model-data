#!/usr/bin/env python3
"""
模型 ID 归一 —— 全库唯一的实现

为什么重写
----------
原来有 5 个模块都在做 ID 归一（idnorm / identity / official_rules /
family / attribution），规则互相覆盖，调用方叠了四五层，每层都以为
下一层会处理自己没处理的情况。结果就是一个 bug 修完冒三个新的。

现在只保留这一层：

    原始 id
      ↓ ① 剥渠道/区域前缀      databricks-claude-... → claude-...
      ↓ ② p 转义还原           v4p1 → v4.1
      ↓ ③ 写法归一             大小写/下划线/小数点 → 短横线
      ↓ ④ 官方格式（Claude）   claude-opus-4-0 → claude-opus-4
      ↓ ⑤ 版本号加点           v4-1 → v4.1
    规范 id

对外只暴露三个函数：
    canonical_id(raw)      规范 id（用于产出与族键，必须一致）
    family_key(raw)        家族键（canonical 去掉日期快照）
    snapshot_date(raw)     快照日期（没有则 None）

其余规则（厂商归属、可信度、计价）都不在这一层。
"""

from __future__ import annotations

import re
from functools import lru_cache

# ══ ① 前缀剥离 ═════════════════════════════════════════════════════════════

# 区域/部署标记。这些**永远**不是模型身份的一部分。
_REGION = (
    r"apac|us-gov|us-east|us-west|us|eu|global|cn|jp|kr|au|ca|uk|me|sa|af|ap"
)

# 渠道/中转商。同样是转售层，不是模型名。
_CHANNEL = (
    r"databricks|bedrock|vertex|azure|sagemaker|foundry|oci|"
    r"openrouter|novita|deepinfra|together|fireworks|groq|"
    r"github[-_]copilot|perplexity[-_]agent|aihubmix|edenai|llmgateway|"
    r"nano-gpt|ofox|zenmux|crossmodel|vercel|kilo|requesty|cortecs|"
    r"merge-gateway|coralbricks|iteracompute|modular|scaleway|cerebras|"
    r"baseten|cloudflare|hyperbolic|lambda|fireworks[-_]ai|novita[-_]ai"
)

# 厂商名前缀。判定依据是「后面紧跟的是该厂商的专属系列名」：
#   anthropic-claude-...  → claude-...    （claude 只属于 Anthropic）
#   openai-gpt-...        → gpt-...       （gpt 只属于 OpenAI）
#   deepseek-v4.1-flash   → 不动          （v4.1 是版本号，不是系列名）
VENDOR_SERIES: dict[str, tuple[str, ...]] = {
    "anthropic": ("claude",),
    "openai": ("gpt", "chatgpt", "o1", "o3", "o4"),
    "google": ("gemini", "gemma", "palm", "imagen", "veo"),
    "meta": ("llama",),
    "mistral": ("mistral", "codestral", "ministral", "pixtral", "devstral",
                "magistral", "voxtral", "mixtral"),
    "cohere": ("command", "aya"),
    "amazon": ("nova", "titan"),
    "microsoft": ("phi",),
    "xai": ("grok",),
    "deepseek": ("deepseek",),      # 只有 deepseek-deepseek-r1 这种重复才剥
}

# 只匹配「厂商名 + 分隔符」，用 lookahead 确认后面是该厂商的系列名，
# 但**不消耗系列名** —— anthropic-claude-x 剥成 claude-x，不是 x。
_VENDOR_SERIES_RE = re.compile(
    "^(?:" + "|".join(
        f"{v}(?=[._-](?:{'|'.join(series)})(?:[._-]|$))"
        for v, series in VENDOR_SERIES.items()
    ) + r")(?:[._-])",
    re.I,
)

_RE_STRIP_REGION = re.compile(rf"^(?:{_REGION})[._-]", re.I)
_RE_STRIP_CHANNEL = re.compile(rf"^(?:{_CHANNEL})[._-]", re.I)

# 厂商名 + 分隔符 + 该厂商的专属系列名（允许系列名后紧跟数字：llama2 / llama3）
_VENDOR_TIGHT_RE = re.compile(
    "^(?:"
    + "|".join(
        f"{v}[._-](?=(?:{'|'.join(series)}))"
        for v, series in {
            "anthropic": ("claude",),
            "openai": ("gpt", "chatgpt"),
            "google": ("gemini", "gemma"),
            "meta": ("llama",),
            "mistral": ("mistral", "codestral"),
            "cohere": ("command", "aya"),
            "amazon": ("nova",),
            "microsoft": ("phi",),
            "xai": ("grok",),
        }.items()
    )
    + ")",
    re.I,
)


def strip_prefix(raw: str) -> str:
    """
    剥掉渠道、区域、以及紧随其后的厂商名前缀。

        databricks-claude-opus-4-8          → claude-opus-4-8
        apac-anthropic-claude-opus-4-7      → claude-opus-4-7
        github_copilot/claude-opus-4.8      → claude-opus-4.8
        deepseek-v4.1-flash                 → deepseek-v4.1-flash（不动）
    """
    s = str(raw or "")
    if "/" in s:
        s = s.rsplit("/", 1)[-1]

    stripped_something = False
    prev = None
    while prev != s:
        prev = s
        before = s
        s = _RE_STRIP_CHANNEL.sub("", s, count=1)
        s = _RE_STRIP_REGION.sub("", s, count=1)
        if s != before:
            stripped_something = True

    # 厂商名 + 专属系列名（anthropic-claude-… / openai-gpt-…）→ 只留系列名
    s = _VENDOR_SERIES_RE.sub("", s, count=1)

    # 已经剥过区域/渠道前缀时，裸的厂商名也剥（apac-anthropic-claude → claude）
    if stripped_something:
        s = re.sub(r"^(?:anthropic|openai|google|meta|mistral|cohere|amazon|microsoft|xai)[._-]",
                   "", s, count=1, flags=re.I)

    # 厂商名 + 分隔符 + 系列名（含紧贴数字，如 meta-llama2）
    s = _VENDOR_TIGHT_RE.sub("", s, count=1)
    return s


# ══ ② p 转义还原 ═══════════════════════════════════════════════════════════

# id 字段常不允许点号，于是用 p 代替：
#   deepseek-v4p1-flash == deepseek-v4.1-flash
#   glm-4p5             == glm-4.5
# 只转「数字 p 数字」，避免误伤单词里的 p。
_RE_P_ESCAPE = re.compile(r"(?<=\d)p(?=\d)", re.I)


# ══ ③ 写法归一 ═════════════════════════════════════════════════════════════

_RE_DUP_DASH = re.compile(r"-{2,}")


def _unify_spelling(s: str) -> str:
    """小写、下划线/点号统一为短横线、压掉重复短横线。"""
    s = s.strip().lower()
    s = _RE_P_ESCAPE.sub(".", s)
    s = s.replace("_", "-").replace(".", "-")
    s = _RE_DUP_DASH.sub("-", s)
    return s.strip("-")


# ══ ④ Claude 官方格式 ══════════════════════════════════════════════════════

_CLAUDE = re.compile(r"^claude-(?P<name>[a-z]+)-(?P<major>\d+)(?:-(?P<minor>\d+))?$")
_CLAUDE_NAMES = {"opus", "sonnet", "haiku", "fable", "mythos"}
_CLAUDE_DATELESS_CANONICAL_SINCE = (4, 6)


def parse_claude(s: str) -> dict | None:
    """
    解析 Claude 的规范 id。**点号与短横线都接受** ——
    本库统一输出点号（claude-haiku-4.5），但输入可能是任一写法。

    只认 claude-{name}-{major}[.-{minor}]，name 必须是系列名。
    OpenRouter 那种 claude-4.8-opus-20260528 倒序格式不认。
    """
    s = str(s or "").replace(".", "-")
    m = _CLAUDE.match(s)
    if not m or m.group("name") not in _CLAUDE_NAMES:
        return None
    major = int(m.group("major"))
    minor = int(m.group("minor") or 0)
    return {
        "name": m.group("name"),
        "major": major,
        "minor": minor,
        "dateless_is_canonical": (major, minor) >= _CLAUDE_DATELESS_CANONICAL_SINCE,
    }


# ══ ⑤ 版本号加点 ═══════════════════════════════════════════════════════════

# 版本号后面常见的词：出现这些说明前面的数字对是版本
_VERSION_FOLLOWERS = {
    "turbo", "mini", "nano", "micro", "small", "medium", "large", "pro", "ultra",
    "flash", "lite", "max", "plus", "base", "instruct", "chat", "preview",
    "latest", "thinking", "reasoning", "code", "coder", "sol", "terra", "luna",
    "argon", "opus", "sonnet", "haiku", "omni", "vl", "vision", "audio", "tts",
    "exp", "fast", "flex", "air", "sonic", "realtime", "live", "instant",
}

# 型号名里的固定词：出现在这些词之后的数字对**不是**版本。
# 注意 **不包含 seed** —— doubao-seed-2-0 里的 2-0 确实是版本 2.0，
# 判断依据应看「数字对之后是什么」而不是前面是什么词。
_NOT_VERSION_PREFIX = {"vl", "a", "b", "e", "i", "it"}


_RE_SHORT_NUM = re.compile(r"^\d{1,2}$")
_RE_VERSION_SEG = re.compile(r"^[a-z]{1,8}\d{1,2}$")
_RE_V_SUB = re.compile(r"^v\d{1,2}$")
_RE_SIZE_MARK = re.compile(r"^\d+[a-z]+$")
_RE_ALPHA = re.compile(r"^[a-z]+$")
_RE_DATE6 = re.compile(r"^\d{6,}$")


def _is_short_num(t: str) -> bool:
    return bool(_RE_SHORT_NUM.match(t))


def _at_version_prefix(parts: list[str], i: int) -> bool:
    if i == 0:
        return True
    prev = parts[i - 1]
    if prev in _NOT_VERSION_PREFIX:
        return False
    if prev.isdigit() and len(prev) == 4:      # 年份后接的是日期
        return False
    return True


def _at_version_end(parts: list[str], idx: int) -> bool:
    if idx >= len(parts):
        return True
    nxt = parts[idx]
    if nxt in _VERSION_FOLLOWERS:
        return True
    if _RE_V_SUB.match(nxt):                    # claude-opus-4-6-v1
        return True
    if nxt.isdigit() and len(nxt) >= 6:         # 日期快照
        return True
    if _RE_SIZE_MARK.match(nxt):                # 规模标记 8b / 27b
        return True
    if nxt.isdigit():
        return True
    return bool(_RE_ALPHA.match(nxt))


def _dots_for_version(s: str) -> str:
    """
    横线版本号转小数点：v4-1 → v4.1，claude-sonnet-4-5 → 4.5。

    只转「数字-数字」，且这两段处在版本位置、后面紧跟版本词或结尾。
        llama-3-1-8b   → llama-3.1-8b      （3.1 是版本，8b 是规模）
        doubao-seed-2-0-pro → 2.0
        qwen3-vl-235b-a22b  → 不动（规模标记）
    """
    parts = s.split("-")
    out: list[str] = []
    i = 0
    while i < len(parts):
        cur = parts[i]
        nxt = parts[i + 1] if i + 1 < len(parts) else None
        if (nxt is not None and _is_short_num(cur) and _is_short_num(nxt)
                and _at_version_prefix(parts, i) and _at_version_end(parts, i + 2)):
            merged = f"{cur}.{nxt}"
            j = i + 2
            while (j + 1 < len(parts) and _is_short_num(parts[j])
                   and _at_version_end(parts, j + 1)):
                merged += f".{parts[j]}"
                j += 1
            out.append(merged)
            i = j
            continue
        # v4-1 形态：字母+数字 的版本段紧跟纯数字
        if (nxt is not None and _RE_VERSION_SEG.match(cur)
                and _is_short_num(nxt) and _at_version_end(parts, i + 2)):
            out.append(f"{cur}.{nxt}")
            i += 2
            continue
        out.append(cur)
        i += 1
    return "-".join(out)


# ══ 对外接口 ═══════════════════════════════════════════════════════════════

@lru_cache(maxsize=65536)
def canonical_id(raw: str) -> str:
    """
    规范 id —— 产出与族键**必须都用它**，否则同一模型会分裂成两条。

    >>> canonical_id("databricks-claude-opus-4-8")
    'claude-opus-4-8'
    >>> canonical_id("deepseek-v4p1-flash")
    'deepseek-v4.1-flash'
    >>> canonical_id("llama-3-1-8b")
    'llama-3.1-8b'
    """
    if not raw:
        return ""
    s = strip_prefix(str(raw))
    s = _unify_spelling(s)
    if not s:
        return ""

    # Claude 必须在版本加点**之前**处理：
    #   先摘日期尾巴，否则 claude-opus-4-20250514 的 20250514 会被当成 minor；
    #   也不能走 _dots_for_version，否则 claude-haiku-4-5 会变成 4.5 导致解析失败。
    date = ""
    if m := re.search(r"-(\d{8})$", s):
        date = m.group(1)
        s = s[: m.start()]

    p = parse_claude(s)
    if p:
        # **全库统一口径：版本用小数点**，Claude 也不例外。
        #
        # 这个库的用途是给中转站和工具做映射表，消费方需要的是
        # 「一个模型一个规范 id」，而不是复刻各家的官方写法。
        # 统一成 claude-opus-4.5 后，消费方不必为 Claude 单独写规则。
        #
        # 官方 id（claude-opus-4-5）会作为 alias 保留，双向都能查到。
        # 主版本省略 minor：Claude Opus 4 → claude-opus-4，没有 .0
        ver = str(p["major"])
        if p["minor"]:
            ver += f".{p['minor']}"
        out = f"claude-{p['name']}-{ver}"
        return f"{out}-{date}" if date else out

    return _dots_for_version(s)


_RE_FAMILY_TAIL = re.compile(r"-(?:\d{4}|\d{6}|\d{8})$")
_RE_SNAP_TAIL = re.compile(r"-(\d{4}|\d{6}|\d{8})$")


@lru_cache(maxsize=65536)
def family_key(raw: str) -> str:
    """
    家族键：规范 id 去掉尾部日期快照。

        deepseek-v4.1-flash-20260423  →  deepseek-v4.1-flash
        deepseek-v4.1-flash-1017      →  deepseek-v4.1-flash
        claude-opus-4-6               →  claude-opus-4.6（本身是规范 id）
    """
    s = canonical_id(raw)
    base = _RE_FAMILY_TAIL.sub("", s)
    return base or s


@lru_cache(maxsize=65536)
def snapshot_date(raw: str) -> str | None:
    """尾部日期快照（4/6/8 位纯数字），没有则 None。"""
    s = canonical_id(raw)
    m = _RE_SNAP_TAIL.search(s)
    return m.group(1) if m else None


def is_claude_model(raw: str) -> bool:
    return parse_claude(_unify_spelling(strip_prefix(str(raw or "")))) is not None
