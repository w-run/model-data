#!/usr/bin/env python3
"""
models.dev 源仓库解析器

数据源是 https://github.com/sst/models.dev 的 TOML 文件，不是 API。
理由：TOML 里有 API 没有的东西 —— 每个数值的**出处注释**和**核对日期**：

    # GET /models reports context_window 1048576 and max_output_tokens 393216.
    # Sources (api-docs.deepseek.com, accessed 2026-10-19): ...
    [limit]
    context = 1_048_576
    output = 393_216

这是唯一能支撑「可信度计算」的格式。API 只给一个裸数字，无从判断真假。

两种模型条目
------------
1. 自包含：直接带 [limit] / [cost]
2. 继承：带 base_model = "org/model"，指向另一个条目

86% 的条目是第 2 种，所以必须实现继承解析，否则只能看到一半数据。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # py<3.11
    import tomli as tomllib  # type: ignore

# 真正研发模型的厂商。与 data/vendors.toml 对应，但这里独立维护，
# 因为需要区分「官方直营」和「官方云的转售」。
FIRST_PARTY_PROVIDERS = {
    "openai", "anthropic", "google", "deepseek", "alibaba", "alibaba-cn",
    "xai", "mistral", "moonshotai", "zhipuai", "minimax", "stepfun",
    "nvidia", "meta", "cohere", "perplexity", "sarvam", "xiaomi",
    "bytedance", "tencent", "baidu", "amazon",
}

# 官方云转售：同一个模型，但是二手渠道，可能有平台侧限制
OFFICIAL_CLOUD_PROVIDERS = {
    "azure", "azure-cognitive-services", "amazon-bedrock", "aws-bedrock",
    "google-vertex", "google-vertex-anthropic", "google-vertex-ai",
    "oci", "cloudflare-workers-ai", "cloudflare-ai-gateway",
    "github-copilot",
}

# 官方转售里，这些是**真·厂商自营云**，可信度高于一般聚合
def tier_of(provider: str) -> str:
    if provider in FIRST_PARTY_PROVIDERS:
        return "first_party"
    if provider in OFFICIAL_CLOUD_PROVIDERS:
        return "official_cloud"
    return "aggregator"


@dataclass
class Doc:
    """一个 TOML 文件的解析结果。"""

    provider: str
    doc_id: str          # 例如 "deepseek-v4.1-flash"
    path: str
    data: dict[str, Any]
    comments: str        # 文件头注释，用于提取出处
    sources: list[str] = field(default_factory=list)   # 提取到的 URL
    accessed_at: str | None = None                     # "accessed 2026-10-19"


_URL_RE = re.compile(r"https?://[^\s)\]，,；;]+")
_ACCESSED_RE = re.compile(r"accessed\s+(\d{4}-\d{2}-\d{2})", re.I)


def parse_doc(path: str, provider: str, doc_id: str) -> Doc | None:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None

    # 注释 = 出现第一个 TOML 键之前的所有 # 行
    header_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("#"):
            header_lines.append(s.lstrip("#").strip())
        elif s:
            break
    comments = "\n".join(header_lines)

    try:
        data = tomllib.loads(raw)
    except Exception:  # noqa: BLE001  个别文件可能语法损坏，跳过而非中断
        return None

    urls = _URL_RE.findall(comments)
    m = _ACCESSED_RE.search(comments)
    return Doc(
        provider=provider,
        doc_id=doc_id,
        path=path,
        data=data,
        comments=comments,
        sources=urls,
        accessed_at=m.group(1) if m else None,
    )


def scan(repo_dir: str) -> dict[str, list[Doc]]:
    """
    扫描仓库，返回 {provider: [Doc, ...]}。

    目录结构：
        providers/<provider>/models/<model>.toml
        providers/<provider>/models/<org>/<model>.toml   （嵌套，272 个）
    """
    root = Path(repo_dir) / "providers"
    result: dict[str, list[Doc]] = {}
    if not root.is_dir():
        return result

    for provider_dir in sorted(root.iterdir()):
        if not provider_dir.is_dir():
            continue
        models_dir = provider_dir / "models"
        if not models_dir.is_dir():
            continue
        provider = provider_dir.name
        for path in sorted(models_dir.rglob("*.toml")):
            rel = path.relative_to(models_dir).with_suffix("")
            doc_id = "/".join(rel.parts)
            d = parse_doc(str(path), provider, doc_id)
            if d:
                result.setdefault(provider, []).append(d)
    return result


def org_of(doc: Doc) -> str | None:
    """
    模型归属厂商。

    models.dev 有两层信息：
      providers/<provider>/   → 谁在**提供**这个模型（可能是转售云）
      doc_id 里的 <org>/<model> → 谁**发布**了这个模型

    例：providers/alibaba/models/qwen/qwen3-max.toml
        provider=alibaba（阿里云在提供）
        org=qwen（通义千问发布会）

    有嵌套 org 时取 org，否则退回 provider。
    """
    if "/" in doc.doc_id:
        return doc.doc_id.split("/")[0].lower()
    return None


def canonical_key(doc: Doc) -> str:
    """该条目最终归属的模型 slug（用于跨 provider 对齐）。"""
    base = doc.data.get("base_model")
    if isinstance(base, str) and base:
        return base.split("/")[-1]
    return doc.doc_id.split("/")[-1]


def resolve_limit(doc: Doc, index: dict[str, Doc], _seen: set[str] | None = None) -> dict[str, Any] | None:
    """
    沿 base_model 链解析出该条目的 limit。

    返回 {"context": int, "output": int, "inherited_from": str|None, "hops": int}
    自身没有 [limit] 时向上找，最多 8 跳防环。
    """
    _seen = _seen or set()
    key = f"{doc.provider}/{doc.doc_id}"
    if key in _seen or len(_seen) > 8:
        return None
    _seen.add(key)

    limit = doc.data.get("limit") or {}
    ctx = limit.get("context")
    out = limit.get("output")

    if ctx:
        return {
            "context": ctx,
            "output": out,
            "inherited_from": None if len(_seen) == 1 else None,
            "hops": len(_seen) - 1,
        }

    base = doc.data.get("base_model")
    if not isinstance(base, str) or not base:
        return None

    target = index.get(base)
    if not target:
        # 允许只写 model 名，不带 org/
        target = index.get(base.split("/")[-1])
    if not target:
        return None

    resolved = resolve_limit(target, index, _seen)
    if not resolved:
        return None
    return {
        "context": resolved["context"],
        "output": resolved["output"],
        "inherited_from": base,
        "hops": len(_seen) - 1,
    }


def build_index(docs: dict[str, list[Doc]]) -> dict[str, Doc]:
    """建立 provider/doc_id -> Doc 的索引，供 base_model 解析。"""
    index: dict[str, Doc] = {}
    for provider, items in docs.items():
        for d in items:
            index[f"{provider}/{d.doc_id}"] = d
    return index
