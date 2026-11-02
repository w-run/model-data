#!/usr/bin/env python3
"""
厂商官方的模型 ID 规则

这些规则**不是猜的**，来自官方文档原文：

Anthropic — Model IDs and versioning
(https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions)

  「Starting with the Claude 4.6 generation, model IDs use a dateless format:
    claude-{name}-{major}[-{minor}]」

  「For the 4.6 generation and later, **the dateless ID is the canonical model ID**.
    It maps to a single, fixed model snapshot.」
  「An alias such as claude-sonnet-4-5 is a convenience pointer that resolves to
    the most recent dated snapshot for that minor version.」
  「A 4.6-generation ID such as claude-sonnet-4-6 is **not an alias**. It is the snapshot.」
  「On Claude Models on Google Cloud (Google managed), the date is separated with @:
    claude-haiku-4-5@20251001」← 只是分隔符不同，同一快照

  关键分界：**4.6 及以后的无日期 ID 是规范 ID；4.6 之前的无日期 ID 是别名**
  —— 方向是按代际相反的，不能用一条规则套所有版本。

OpenAI — Models
(https://platform.openai.com/docs/models)
  现行模型的 Model ID 就是无日期形式（gpt-6.1-terra / gpt-6.1-sol）。
  带日期（gpt-4o-mini-2024-07-18）是历史快照的写法。

结论
----
  1. 无日期 ID 本身是**正当的模型标识**，不能因为「缺少日期」就视为别名
  2. 是否把两个 ID 视为同一模型，取决于**代际**，不能只看名称结构
  3. @YYYYMMDD 与 -YYYYMMDD 是同一快照的两种分隔写法，必须合并
"""

from __future__ import annotations

import re

# 先摘日期再解析主结构：否则 claude-opus-4-20250514 里的 20250514
# 会被当成 minor 版本号（8 位数字不可能是 version minor）。
_DATE_SUFFIX = re.compile(r"-(?P<date>\d{8})$")

# claude-{name}-{major}[-{minor}]
_CLAUDE = re.compile(
    r"^claude-(?P<name>[a-z]+)-(?P<major>\d+)(?:-(?P<minor>\d+))?$"
)

# 4.6 是分界代际：此版本起无日期 ID 是规范 ID
CLAUDE_DATELESS_CANONICAL_SINCE = (4, 6)


# 官方结构里，{name} 只能是系列名。用来排除 claude-4.5-opus-20251124
# 这类第三方倒序改写（name 位置上是数字）。
_CLAUDE_NAMES = {"opus", "sonnet", "haiku", "fable", "mythos"}


def parse_claude(model_id: str) -> dict | None:
    """
    解析 Claude 模型 ID。非官方结构返回 None。

    只认官方文档定义的 claude-{name}-{major}[-{minor}][-{YYYYMMDD}]。
    第三方（OpenRouter 等）的 claude-4.5-opus-20251124 倒序写法不认，
    否则会与官方 ID 互相干扰。
    """
    s = normalize_date_separator(model_id).strip().lower()
    # 兼容已经转成小数点的写法（claude-opus-4.7），统一回横线再解析
    s = s.replace(".", "-")
    date = None
    dm = _DATE_SUFFIX.search(s)
    if dm:
        date = dm.group("date")
        s = s[: dm.start()]

    m = _CLAUDE.match(s)
    if not m:
        return None
    if m.group("name") not in _CLAUDE_NAMES:
        return None
    major = int(m.group("major"))
    minor = int(m.group("minor") or 0)
    return {
        "name": m.group("name"),
        "major": major,
        "minor": minor,
        "date": date,
        "dateless": date is None,
        # 4.6 及以后：无日期 ID 就是规范的固定快照
        "dateless_is_canonical": (major, minor) >= CLAUDE_DATELESS_CANONICAL_SINCE,
    }


# 云区域 + 厂商前缀：LiteLLM 里 Bedrock/GCP 的 key 会长成
#   apac-anthropic-claude-opus-4-7
#   us.anthropic.claude-opus-4-7
# 这些前缀不属于模型身份，必须先剥掉。
_REGION_PART = (
    r"(?:apac|us-gov|us-east|us-west|us|eu|global|cn|jp|kr|au|ca|uk|me|sa|af|ap)"
)
# 渠道/中转商前缀：日积月累的品牌名。它们不是模型身份的一部分。
_CHANNEL_PART = (
    r"(?:databricks|bedrock|vertex|azure|sagemaker|foundry|oci|"
    r"openrouter|novita|deepinfra|together|fireworks|groq|github-copilot|"
    r"perplexity-agent|aihubmix|edenai|llmgateway|nano-gpt|ofox|zenmux|"
    r"crossmodel|vercel|kilo|requesty|cortecs|merge-gateway|"
    r"coralbricks|iteracompute|modular|novita-ai|deepinfra)"
)
# 厂商名前缀（在区域/渠道之后出现）
_VENDOR_PART = (
    r"(?:anthropic|openai|google|meta|mistral|cohere|amazon|microsoft|"
    r"deepseek|moonshotai|zhipuai|xai|alibaba|bytedance|tencent|baidu)"
)
_CLOUD_PREFIX = re.compile(
    rf"^(?:(?:{_REGION_PART}|{_CHANNEL_PART})[._-])?"
    rf"(?:{_VENDOR_PART}[._-])?",
    re.I,
)


def strip_cloud_prefix(model_id: str) -> str:
    """
    剥离云区域、渠道、厂商前缀，返回模型本身的写法。

    也处理 LiteLLM 的 `provider/model` 与 `provider_model` 两种写法
    （github_copilot/claude-opus-4.8）。
    """
    s = str(model_id or "")
    # provider/model：取最后一段
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    prev = None
    while prev != s:
        prev = s
        s = _CLOUD_PREFIX.sub("", s, count=1)
        # 下划线分隔的渠道前缀（github_copilot_...）
        s = re.sub(r"^[a-z0-9]+_", "", s, count=1) if re.match(r"^[a-z]+_", s) else s
    return s


def normalize_date_separator(model_id: str) -> str:
    """
    统一日期分隔符：@YYYYMMDD 与 -YYYYMMDD 是同一种快照写法。

        claude-haiku-4-5@20251001  →  claude-haiku-4-5-20251001
    """
    s = str(model_id or "")
    return re.sub(r"@(\d{8})$", r"-\1", s)


def claude_snapshot_base(model_id: str) -> str | None:
    """
    带日期快照所属的「无日期」形式。非快照返回 None。

        claude-sonnet-4-5-20250929  →  claude-sonnet-4-5
        claude-opus-4-20250514      →  claude-opus-4
    """
    p = parse_claude(normalize_date_separator(model_id))
    if not p or not p["date"]:
        return None
    if p["minor"]:
        return f"claude-{p['name']}-{p['major']}-{p['minor']}"
    return f"claude-{p['name']}-{p['major']}"


def canonical_claude_id(model_id: str) -> str | None:
    """
    按官方规则输出规范 id。

    官方格式：claude-{name}-{major}[-{minor}]
      主版本发布省略 minor：Claude Opus 4 就是 claude-opus-4，**没有 -0**
      （官方文档原文：Major-version releases such as Claude Sonnet 5 and
        Claude Opus 5 omit the minor segment.）

    所以 claude-opus-4-0 应归一为 claude-opus-4，而不是 claude-opus-4.0。
    返回 None 表示不是官方结构。
    """
    p = parse_claude(model_id)
    if not p:
        return None
    parts = ["claude", p["name"], str(p["major"])]
    if p["minor"]:
        parts.append(str(p["minor"]))
    if p["date"]:
        parts.append(p["date"])
    return "-".join(parts)


def should_merge_dateless_with_snapshot(dateless_id: str) -> bool:
    """
    无日期 ID 是否应当并入它的日期快照。

    **只有 4.6 之前的代际**才是别名（指向最新快照），需要合并。
    4.6 及以后的无日期 ID 是规范 ID，与任何快照都不同，不能合并。
    """
    p = parse_claude(dateless_id)
    if not p or not p["dateless"]:
        return False
    return not p["dateless_is_canonical"]
