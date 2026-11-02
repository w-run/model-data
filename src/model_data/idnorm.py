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
    # 语音/多模态档位
    "sonic", "realtime", "live", "nano", "instant", "previewv2",
    # 能力/规模描述
    "reasoning", "thinking", "instruct", "chat", "base", "it", "vision",
}

# 型号名里的固定词：出现在这些词之后，数字对**不是**版本号
NOT_VERSION_PREFIX = {
    "seed", "vl", "a", "b", "b-a", "e", "i", "it", "instruct",
    "smol", "mo", "moe",
}

_SEP_RE = re.compile(r"-{2,}")


# 云区域前缀：global- / us- / eu- / apac- / cn- …
# 这些是**部署区域**标记，不是模型身份的一部分，必须剥离，
# 否则 global-openai-gpt-6.1-terra 与 gpt-6.1-terra 会被当成两个模型。
_REGION_PREFIX_RE = re.compile(
    r"^(?:apac|us-gov|us-east|us-west|us|eu|global|cn|jp|kr|au|ca|uk|me|sa|af|ap)[-_]",
    re.I,
)

# 变体后缀：:batch / :free / :thinking …（冒号分隔，不参与版本号解析）
_VARIANT_TAIL_RE = re.compile(r"(:[a-z0-9_-]+)+$", re.I)

# 渠道/部署前缀：这些是**转售渠道或服务档**，不是模型身份的一部分
#   databricks-gpt-6.1-terra  → gpt-6.1-terra
#   coding-xiaomi-mimo-v2.5   → xiaomi-mimo-v2.5（再经冗余厂商前缀 → mimo-v2.5）
#   open-mistral-nemo         → mistral-nemo
_CHANNEL_PREFIX_RE = re.compile(
    r"^(?:databricks|bedrock|azure|vertex|oci|sagemaker|foundry|"
    r"openrouter|novita|deepinfra|together|fireworks|groq|"
    r"coding|open|pro|free|fast|turbo|"
    r"@cf|cf|global)-",
    re.I,
)

# 厂商别名前缀：中转商常把厂商别名拼在前面（zai-glm-5.3、qwen-qwen3-max）
_VENDOR_ALIAS_PREFIX_RE = re.compile(
    r"^(?:zai|z-ai|zhipu|bigmodel|glm)(?=-glm)-"
    r"|^(?:qwen|aliyun|tongyi)(?=-qwen)-"
    r"|^(?:moonshot|kimi)(?=-kimi)-"
    r"|^(?:volcengine|doubao)(?=-doubao)-",
    re.I,
)

# 厂商名冗余前缀：厂商名与模型自身名字重复时剥掉，不损失识别度。
# 判据是「该前缀之后的部分本身就能唯一定位到这家厂商」：
#     openai-gpt-…        → gpt- 只属于 OpenAI
#     anthropic-claude-…  → claude- 只属于 Anthropic
#     alibaba-qwen3-…     → qwen 只属于阿里（实测全库 qwen 系列均为 alibaba）
#     moonshotai-kimi-…   → kimi 只属于月之暗面
# 不剥的情况：前缀之后的名字**跨厂商使用**，例如
#     deepseek-r1-distill-qwen-32b  → 剥掉 deepseek- 会变成 qwen-，归属就错了
_REDUNDANT_VENDOR_PREFIX_RE = re.compile(
    r"^(?:openai(?=-gpt|-chatgpt)|anthropic(?=-claude)|google(?=-gemini|-gemma)"
    r"|meta(?=-llama)|mistralai(?=-mistral|-codestral)|moonshotai(?=-kimi)"
    r"|zhipuai(?=-glm)|xai(?=-grok)|alibaba(?=-qwen|-qwq|-tongyi)"
    r"|bytedance(?=-doubao)|tencent(?=-hunyuan)|baidu(?=-ernie|-wenxin)"
    r"|xiaomi(?=-mimo)|nvidia(?=-nemotron)|amazon(?=-nova)|cohere(?=-command)"
    r"|ai21(?=-jamba)|moonshot(?=-moonshot))-",
    re.I,
)


def strip_region_prefix(s: str) -> str:
    """剥离可能多层的区域前缀。"""
    prev = None
    while prev != s:
        prev = s
        s = _REGION_PREFIX_RE.sub("", s, count=1)
    return s


def canonical_id(raw: str) -> str:
    """
    把任意写法的模型 id 规范成统一口径。

    >>> canonical_id("DeepSeek-V4.1-Flash")
    'deepseek-v4.1-flash'
    >>> canonical_id("claude-sonnet-4-5")
    'claude-sonnet-4.5'
    >>> canonical_id("llama-3-1-8b")
    'llama-3-1-8b'
    >>> canonical_id("global-openai-gpt-6.1-terra")
    'openai-gpt-6.1-terra'
    >>> canonical_id("gpt-6-1-terra:batch")
    'gpt-6.1-terra:batch'
    """
    s = str(raw or "").strip().lower().replace("_", "-")
    s = _SEP_RE.sub("-", s).strip("-")
    if not s:
        return ""

    # 变体后缀先摘掉：`gpt-6-1-terra:batch` 里的 `:batch` 会挡住版本号识别
    variant = ""
    m_var = _VARIANT_TAIL_RE.search(s)
    if m_var:
        variant = m_var.group(0)
        s = s[: m_var.start()]

    # 区域前缀剥离：global-/us-/apac- 是部署标记，不属于模型身份
    s = strip_region_prefix(s)

    # 渠道前缀剥离：databricks-gpt-6.1-terra 就是 gpt-6.1-terra
    s = _CHANNEL_PREFIX_RE.sub("", s, count=1)

    # 厂商别名前缀：zai-glm-5.3 → glm-5.3
    s = _VENDOR_ALIAS_PREFIX_RE.sub("", s, count=1)

    # 冗余厂商前缀：openai-gpt-… / anthropic-claude-… 剥掉不损失识别度
    s = _REDUNDANT_VENDOR_PREFIX_RE.sub("", s, count=1)

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

    # 变体后缀原样拼回（:batch 等是同一模型的不同服务档，保留以区分计费）
    return "-".join(out) + variant


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
    """
    第 i 段之前是否构成「版本号可以开始」的语境。

    关键是别把「型号编号」误当版本：
        llama-3-1-8b      3-1 后面跟 8b（规模），是型号编号，不转
        doubao-seed-2-0   2-0 后面是 pro/lite/结尾，是版本号，要转
        doubao-seed-1-6   1-6 后面是 thinking 这类后缀，也是版本号
    所以判断依据是**数字对之后是什么**，而不是前面是什么词。
    seed/vl 这类词本身不能一票否决。
    """
    if i == 0:
        return True
    prev = parts[i - 1]
    # 前一段是年份（4 位数字）时，后面的 07-18 是日期，不是版本
    if prev.isdigit() and len(prev) == 4:
        return False
    # 前一段必须含字母或数字（纯符号不可能）
    if not any(c.isalnum() for c in prev):
        return False
    return True


def _at_version_end(parts: list[str], idx: int) -> bool:
    """
    idx 处是否**不阻碍**前面的数字对构成版本号。

    这里只排除两类明确不是版本的场景：
      - 紧跟 2 位日期段（claude-opus-4-5-20251101 的 20251101 是快照）
    其余都放行 —— 包括规模标记（llama-3-1-8b 的 8b）：
    `llama-3-1-8b` 是 Llama **3.1** 8B，3-1 确实是版本号，
    后面跟参数量并不否定前面是版本。
    """
    if idx >= len(parts):
        return True
    nxt = parts[idx]
    if nxt in VERSION_FOLLOWERS:
        return True
    # v1 / v2 子版本标记（claude-opus-4-6-v1）
    if re.fullmatch(r"v\d{1,2}", nxt):
        return True
    # 日期快照：6 位以上纯数字（claude-opus-4-5-20251101）
    # 它是**版本号的合法结束位置**（4-5 转成 4.5 后停住），
    # 但不能被继续吸收 —— 吸收循环里有独立的长度判断拦它。
    if nxt.isdigit() and len(nxt) >= 6:
        return True
    # 规模标记（8b / 27b / 235b）—— 版本号之后的常见跟随，不构成否定
    if re.fullmatch(r"\d+[a-z]+", nxt):
        return True
    # 纯数字段：可能是 4-5-1 的多级版本，交给调用方继续吸收
    if nxt.isdigit():
        return True
    # 普通单词（pro / sonic / thinking …）也不阻碍
    return bool(re.fullmatch(r"[a-z]+", nxt))


def is_same_model(a: str, b: str) -> bool:
    """两个写法是否指向同一模型（规范化后比较，忽略分隔符差异）。"""
    ca = canonical_id(a).replace("-", "").replace(".", "")
    cb = canonical_id(b).replace("-", "").replace(".", "")
    return ca == cb
