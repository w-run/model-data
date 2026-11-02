#!/usr/bin/env python3
"""
阿里云百炼（Model Studio）定价源

https://help.aliyun.com/zh/model-studio/model-pricing

这是目前**唯一能直采到人民币原价 + 阶梯**的国内源：
页面是静态 HTML 表格（546 个），不依赖 JS，可稳定解析。

表结构
------
    | 模型 ID | 模式 | 单次请求的输入 Token 数 | 输入单价（每百万Token） | 输出单价（每百万Token） | 免费额度 |
    | qwen3.8-max | 非思考和思考模式 | 0<Token≤1M | 12元 | 36元 | 100万Token |
    | qwen-max    | 仅非思考模式   | 无阶梯计价   | 2.4元 | 9.6元 | 无        |

要点
----
1. 价格单位就是**元 / 百万 Token**，与既有产出口径一致
2. 阶梯用「单次请求的输入 Token 数」列表达（`0<Token≤1M`、`无阶梯计价`）
   这一列同时给出阈值，比 models.dev 的 tiers 语义更明确
3. 部分模型有 Batch 半价、上下文缓存折扣，写在模型名旁的注释里

已知不足
--------
页面上各表分属不同分类（文本生成 / 视觉 / 语音 …），单元格里偶有
换行与「免费额度」等干扰文本，解析时按「含元且形如数字」过滤。
"""

from __future__ import annotations

import html
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from ..trust.engine import Observation

URL = "https://help.aliyun.com/zh/model-studio/model-pricing"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122 Safari/537.36")
CACHE = Path(os.environ.get("MODEL_DATA_CACHE", Path.home() / ".cache" / "model-data"))

# Token 数简写 -> 实际值
_UNIT = {"k": 1000, "K": 1000, "万": 10_000, "m": 1_000_000, "M": 1_000_000,
         "亿": 100_000_000}


def fetch_html(offline: bool = False, timeout: int = 45) -> str:
    cached = CACHE / "aliyun_pricing.html"
    if offline and cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")
    try:
        req = urllib.request.Request(URL, headers={
            "User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", errors="replace")
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(text, encoding="utf-8")
        return text
    except Exception:
        if cached.exists():
            return cached.read_text(encoding="utf-8", errors="replace")
        raise


def _cell_text(raw: str) -> str:
    """
    单元格转纯文本。

    注意保留换行信息：模型 ID 单元格常是
        "qwen3.7-max \n 当前能力等同于 qwen3.7-max-0902"
    只取第一行才是模型 ID，所以 <br> 要变成 \n 而不是空格。
    """
    t = re.sub(r"<br\s*/?>", "\n", raw, flags=re.I)
    t = re.sub(r"</p>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]
    return "\n".join(x for x in lines if x)


_PRICE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*元")


def _parse_price(text: str) -> float | None:
    """'12元' -> 12.0；'0.0008元' -> 0.0008；'无' / '免费' -> None。"""
    m = _PRICE_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


_TIER_RE = re.compile(r"([0-9.]+)\s*([kKmM万]?)\s*<\s*Token\s*[≤<=]\s*([0-9.]+)\s*([kKmM万]?)")


_OVER_RE = re.compile(r"([0-9.]+)\s*([kKmM万]?)\s*Token\s*[：:]\s*([0-9.]+)\s*元")


def _parse_tiered_prices(text: str) -> list[tuple[int, float]]:
    """
    解析「输入不超过 512k Token：2.1 元 输入超过 512k Token：4.2 元」。

    返回 [(阈值 token 数, 单价), ...]，按阈值升序。
    第一档（不超过 X）用 over=0 表示，即基础价。
    """
    out: list[tuple[int, float]] = []
    # 「超过 X Token：Y 元」
    for m in re.finditer(r"超过\s*([0-9.]+)\s*([kKmM万]?)\s*Token\s*[：:]\s*([0-9.]+)\s*元", text or ""):
        try:
            over = int(float(m.group(1)) * _UNIT.get(m.group(2), 1))
        except ValueError:
            continue
        out.append((over, float(m.group(3))))
    # 「不超过 X Token：Y 元」→ 基础价
    for m in re.finditer(r"不超过\s*([0-9.]+)\s*([kKmM万]?)\s*Token\s*[：:]\s*([0-9.]+)\s*元", text or ""):
        out.append((0, float(m.group(3))))
    return sorted(set(out))


def _parse_tier(text: str) -> dict[str, Any] | None:
    """
    解析「单次请求的输入 Token 数」列。

        '0<Token≤1M'    -> {"over": 1, "size": 1000000}
        '无阶梯计价'     -> None
    """
    t = (text or "").strip()
    if not t or "无阶梯" in t:
        return None
    m = _TIER_RE.search(t)
    if not m:
        return None
    lo_num, lo_unit, hi_num, hi_unit = m.groups()
    try:
        lo = float(lo_num) * _UNIT.get(lo_unit, 1)
        hi = float(hi_num) * _UNIT.get(hi_unit, 1)
    except ValueError:
        return None
    return {"over": int(lo), "size": int(hi)}


def parse(html_text: str) -> dict[str, dict[str, Any]]:
    """
    解析出 {model_id: {input, output, tiers: [...]}}。

    同一模型可能在多张表里重复出现（不同计费档位），按 tier 合并。
    """
    result: dict[str, dict[str, Any]] = {}

    for table in re.findall(r"<table.*?</table>", html_text, re.S):
        rows = re.findall(r"<tr.*?</tr>", table, re.S)
        if len(rows) < 2:
            continue

        header = [_cell_text(c) for c in
                  re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", rows[0], re.S)]
        col_in = next((i for i, h in enumerate(header) if "输入单价" in h), None)
        col_out = next((i for i, h in enumerate(header) if "输出单价" in h), None)
        col_tier = next((i for i, h in enumerate(header) if "Token" in h and "输入" in h), None)
        if col_in is None or col_out is None:
            # 表头格式不同（如视觉/语音表），退回按位置猜
            col_in = 3 if col_in is None else col_in
            col_out = 4 if col_out is None else col_out
        if col_tier is None:
            col_tier = 2   # 非标准表可能没有「Token 阶梯」列

        for row in rows[1:]:
            cells = [_cell_text(c) for c in
                     re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)]
            if len(cells) <= max(col_in, col_out):
                continue

            model_id = cells[0].split("\n")[0].strip()
            if not model_id or any(ch in model_id for ch in "（）|"):
                continue
            if not re.fullmatch(r"[A-Za-z0-9._\-/]+", model_id):
                continue

            tier_text = cells[col_tier] if col_tier < len(cells) else ""
            in_price = _parse_price(cells[col_in])
            out_price = _parse_price(cells[col_out])
            if in_price is None and out_price is None:
                continue

            entry = result.setdefault(model_id, {"tiers": []})

            # 文字阶梯：输入列与输出列**各自**带阈值，要配对
            in_tiers = _parse_tiered_prices(cells[col_in])
            out_tiers = _parse_tiered_prices(cells[col_out])
            if len(in_tiers) > 1 or len(out_tiers) > 1:
                out_by_over = {o: p for o, p in out_tiers}
                in_by_over = {o: p for o, p in in_tiers}
                seen_over: set[int] = set()
                for over in sorted(set(in_by_over) | set(out_by_over)):
                    if over in seen_over:
                        continue
                    seen_over.add(over)
                    if over == 0:
                        # 第一档 = 基础价
                        if over in in_by_over:
                            entry.setdefault("input", in_by_over[over])
                        if over in out_by_over:
                            entry.setdefault("output", out_by_over[over])
                        continue
                    entry["tiers"].append({
                        "over": over,
                        "type": "context",
                        "input": in_by_over.get(over),
                        "output": out_by_over.get(over),
                    })
                continue

            tier = _parse_tier(tier_text)
            if tier:
                # 「0<Token≤1M」表示从 0 到 1M 都是这个价 —— 这就是**基础价**，
                # 不是额外档位。只有下界 > 0 才是真正的阶梯。
                if tier["over"] == 0:
                    entry.setdefault("input", in_price)
                    entry.setdefault("output", out_price)
                    entry["base_range"] = tier["size"]
                else:
                    entry["tiers"].append({
                        "over": tier["size"],
                        "type": "context",
                        "input": in_price,
                        "output": out_price,
                    })
            else:
                entry.setdefault("input", in_price)
                entry.setdefault("output", out_price)

    # 同一模型可能在多张表里重复出现（分属不同分类），阶梯去重
    for entry in result.values():
        tiers = entry.get("tiers") or []
        if tiers:
            seen, uniq = set(), []
            for t in tiers:
                k = (t.get("over"), t.get("input"), t.get("output"))
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(t)
            entry["tiers"] = uniq
    return result


def observations(html_text: str | None = None, *, offline: bool = False) -> dict[str, list[Observation]]:
    """产出观测，价格标 currency='CNY'，阶梯带 price_tier。"""
    html_text = html_text if html_text is not None else fetch_html(offline=offline)
    parsed = parse(html_text)

    out: dict[str, list[Observation]] = {}
    for model_id, info in parsed.items():
        slug = model_id.split("/")[-1]
        base = {"source": "aliyun:model-studio", "tier": "first_party",
                "provenance": "pricing_url", "url": URL, "observed_at": None,
                "valid": True, "currency": "CNY"}

        if info.get("input") is not None:
            out.setdefault(slug, []).append(Observation(
                field_name="price_input", value=float(info["input"]), **base))
        if info.get("output") is not None:
            out.setdefault(slug, []).append(Observation(
                field_name="price_output", value=float(info["output"]), **base))

        for t in info.get("tiers", []):
            out.setdefault(slug, []).append(Observation(
                field_name="price_tier", value=t, **base))

    return out
