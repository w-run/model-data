#!/usr/bin/env python3
"""
MiniMax 官方规格源

https://platform.minimax.cn/docs/guides/text-generation

页面静态 HTML 表格：

    | 模型名称 | 上下文窗口 | 模型介绍 |
    | MiniMax-M3.1-Flash | 1,000,000 | ... |

为什么需要它
------------
models.dev 上 MiniMax 官方条目**只声明了 output（524288），没有 context**，
导致上下文只能靠聚合商投票 —— nvidia 报 262144、cortecs 报 1048576，
最后选出一个远小于真实的 256000。官方文档明确是 1,000,000。

注意
----
- 文档只给上下文窗口，不给输出上限（输出由 models.dev 官方条目的
  output = 524288 提供），两者互补
- 模型名带大小写（MiniMax-M3.1-Flash），走 ids 归一后落成 minimax-m3.1-flash
"""

from __future__ import annotations

import html
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from ..trust.engine import Observation

URL = "https://platform.minimax.cn/docs/guides/text-generation"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122 Safari/537.36")
CACHE = Path(os.environ.get("MODEL_DATA_CACHE", Path.home() / ".cache" / "model-data"))


def fetch_html(offline: bool = False, timeout: int = 45) -> str:
    cached = CACHE / "minimax_textgen.html"
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


def _parse_int(text: str) -> int | None:
    """'1,000,000' → 1000000。"""
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else None


def parse(html_text: str) -> dict[str, dict[str, int]]:
    """解析出 {model_name: {"context": n}}。"""
    out: dict[str, dict[str, int]] = {}

    for table in re.findall(r"<table.*?</table>", html_text, re.S):
        rows = re.findall(r"<tr.*?</tr>", table, re.S)
        if len(rows) < 2:
            continue

        header = [_cell(c) for c in
                  re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", rows[0], re.S)]
        if not any("上下文" in h for h in header):
            continue
        try:
            col_name = next(i for i, h in enumerate(header) if "模型" in h)
            col_ctx = next(i for i, h in enumerate(header) if "上下文" in h)
        except StopIteration:
            continue

        for row in rows[1:]:
            cells = [_cell(c) for c in
                     re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)]
            if len(cells) <= max(col_name, col_ctx):
                continue
            name = cells[col_name].strip()
            ctx = _parse_int(cells[col_ctx])
            if name and ctx:
                out[name] = {"context": ctx}

    return out


def observations(html_text: str | None = None, *,
                 offline: bool = False) -> dict[str, list[Observation]]:
    html_text = html_text if html_text is not None else fetch_html(offline=offline)
    parsed = parse(html_text)

    out: dict[str, list[Observation]] = {}
    for name, lim in parsed.items():
        out.setdefault(name, []).append(Observation(
            source="minimax:official",
            tier="first_party",
            field_name="context",
            value=lim["context"],
            provenance="doc_url",
            url=URL,
            observed_at=None,
            valid=True,
            currency="CNY",
        ))
    return out
