#!/usr/bin/env python3
"""
智谱 AI（BigModel）官方规格源

https://docs.bigmodel.cn/cn/guide/start/model-overview

页面是静态 HTML 表格，无需登录、无需 API key。表格结构：

    | 模型 | 特点 | 上下文 | 最大输出 |
    | GLM-5.3 | ... | 1M | 128K |

价值
----
智谱的 [limit] 在 models.dev 上覆盖不全（22 条里 0 条带 limit），
只能靠聚合商投票，导致 glm-5.3 的输出上限被各种乱猜（16384 ~ 1048576）。
官方文档直接给出 1M / 128K，是**唯一权威**。

注意
----
- 表格里的 1M 是厂商标称的十进制 1M（1_000_000），不是 1Mi
- 官方同族模型（Flash / FlashX / Legal）规格可能一致，但**各自独立**
- OCR / ASR / TTS / 图像 / 视频模型没有 token 上下文，跳过
"""

from __future__ import annotations

import html
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from ..trust.engine import Observation

URL = "https://docs.bigmodel.cn/cn/guide/start/model-overview"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122 Safari/537.36")
CACHE = Path(os.environ.get("MODEL_DATA_CACHE", Path.home() / ".cache" / "model-data"))

# 非对话模型：没有 token 上下文，不进库
_SKIP = re.compile(r"(image|ocr|asr|tts|embedding|video|cogvideo|rerank|moderation)", re.I)

_UNIT = {"K": 1000, "M": 1_000_000, "B": 1_000_000_000}


def fetch_html(offline: bool = False, timeout: int = 45) -> str:
    cached = CACHE / "bigmodel_overview.html"
    if offline and cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", errors="replace")
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(text, encoding="utf-8")
        return text
    except Exception:
        if cached.exists():
            return cached.read_text(encoding="utf-8", errors="replace")
        raise


def _cell(raw: str) -> str:
    t = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def _parse_size(text: str) -> int | None:
    """'1M' → 1000000，'128K' → 128000，'8K' → 8000。"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*([KMB]?)", (text or "").upper())
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    return int(val * _UNIT.get(m.group(2), 1))


def parse(html_text: str) -> dict[str, dict[str, int]]:
    """解析出 {model_id: {"context": n, "output": n}}。"""
    out: dict[str, dict[str, int]] = {}

    for table in re.findall(r"<table.*?</table>", html_text, re.S):
        rows = re.findall(r"<tr.*?</tr>", table, re.S)
        if len(rows) < 2:
            continue

        header = [_cell(c) for c in
                  re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", rows[0], re.S)]
        # 只处理「模型 / 上下文 / 最大输出」这张表
        if not any("上下文" in h for h in header):
            continue
        try:
            col_model = next(i for i, h in enumerate(header) if "模型" in h)
            col_ctx = next(i for i, h in enumerate(header) if "上下文" in h)
            col_out = next(i for i, h in enumerate(header) if "最大输出" in h)
        except StopIteration:
            continue

        for row in rows[1:]:
            cells = [_cell(c) for c in
                     re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)]
            if len(cells) <= max(col_model, col_ctx, col_out):
                continue

            name = cells[col_model].strip()
            if not name or _SKIP.search(name):
                continue
            # 模型名转 id：GLM-5.3 → glm-5.3
            mid = name.lower()

            ctx = _parse_size(cells[col_ctx])
            out_size = _parse_size(cells[col_out])
            if ctx is None:
                continue
            entry: dict[str, int] = {"context": ctx}
            if out_size:
                entry["output"] = out_size
            out[mid] = entry

    return out


def observations(html_text: str | None = None, *,
                 offline: bool = False) -> dict[str, list[Observation]]:
    """
    产出观测。tier=first_party（官方文档），带 pricing_url 出处加成。

    注意 provenance 用 doc_url —— 这是官方规格页，不是定价页。
    """
    html_text = html_text if html_text is not None else fetch_html(offline=offline)
    parsed = parse(html_text)

    out: dict[str, list[Observation]] = {}
    for mid, lim in parsed.items():
        base = {
            "source": "bigmodel:official",
            "tier": "first_party",
            "provenance": "doc_url",
            "url": URL,
            "observed_at": None,
            "valid": True,
            "currency": "CNY",
        }
        if lim.get("context"):
            out.setdefault(mid, []).append(Observation(
                field_name="context", value=lim["context"], **base))
        if lim.get("output"):
            out.setdefault(mid, []).append(Observation(
                field_name="max_output", value=lim["output"], **base))

    return out
