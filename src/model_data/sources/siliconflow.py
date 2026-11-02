#!/usr/bin/env python3
"""
硅基流动（SiliconFlow）定价源

数据位置
--------
`https://siliconflow.cn/pricing` 是 Next.js 页面，模型数据以 flight 流
（`self.__next_f.push([1,"..."])`）内嵌在 HTML 里，无需登录、无需 API key。

公开 API 只给 `id/object/created/owned_by`，**不返回价格、上下文、缓存计费** ——
所以这里必须解析页面内嵌数据，不能用 API。

字段（实测）
-----------
    {
      "modelName": "deepseek-ai/DeepSeek-V4.1-Flash",
      "mf": "deepseek-ai",                    # 模型厂商
      "contextLen": 1048576,                  # 上下文
      "price": "8",                           # 输出价（见下）
      "currency": "¥",                        # 人民币
      "priceUnit": "/ M Tokens",
      "priceInfo": [                          # 按 prompt/completion 拆分
        {"price": "8", "specification": "completion", "unitOfGood": "/ M Tokens"},
        {"price": "2", "specification": "prompt",     "unitOfGood": "/ M Tokens"}
      ]
    }

价格是**每百万 token 的人民币**，且是干净整数（8 / 2）—— 比反推换算可靠得多。

缓存计费
--------
硅基流动部分模型有缓存命中价，在 flight 流的 `cache` 相关字段里，
本模块会尽量提取；缺省则留空，不猜。
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from ..trust.engine import Observation

URL = "https://siliconflow.cn/pricing"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122 Safari/537.36")
CACHE = Path(os.environ.get("MODEL_DATA_CACHE", Path.home() / ".cache" / "model-data"))


def fetch_html(offline: bool = False, timeout: int = 45) -> str:
    cached = CACHE / "siliconflow_pricing.html"
    if offline and cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            html = r.read().decode("utf-8", errors="replace")
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(html, encoding="utf-8")
        return html
    except Exception:
        if cached.exists():
            return cached.read_text(encoding="utf-8", errors="replace")
        raise


def _flight_payload(html: str) -> str:
    """
    把 flight 流拼成一整段可解析的文本。

    Next.js 把数据切成多段 `self.__next_f.push([1,"<json 片段>"])`，
    片段本身是 JSON 字符串（转义过），需要先逐段还原再拼接。
    """
    parts = re.findall(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)', html)
    out = []
    for p in parts:
        try:
            out.append(json.loads(p))
        except json.JSONDecodeError:
            continue
    return "".join(out)


def parse_models(html: str) -> list[dict[str, Any]]:
    """
    从 flight 文本里提取模型条目。

    条目形如:
        {"modelId":"...","modelName":"deepseek-ai/DeepSeek-V4.1-Flash",
         "mf":"deepseek-ai","contextLen":1048576,"price":"8",
         "currency":"¥","priceUnit":"/ M Tokens",...}
    """
    text = _flight_payload(html)
    models: list[dict[str, Any]] = []

    # 逐个找 modelName 起点，向前后取平衡的 {...}
    for m in re.finditer(r'\{"modelId"', text):
        start = m.start()
        depth = 0
        end = start
        for i in range(start, min(len(text), start + 20000)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end <= start:
            continue
        try:
            obj = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("modelName"):
            models.append(obj)

    return models


def _price_map(html: str) -> dict[str, dict[str, float]]:
    """
    提取 prompt / completion / cache 的价格表。

    flight 流里价格是**引用式**存储：
        21:{"price":"8","specification":"completion","unitOfGood":"/ M Tokens"}
        22:{"price":"2","specification":"prompt","unitOfGood":"/ M Tokens"}
        20:["$21","$22"]
    模型条目里则是 priceInfo: "$20" 这样的引用。
    所以先建 (行号 → 价格条目) 表，再由引用解析。
    """
    text = _flight_payload(html)

    # 两层结构：
    #   21:{"price":"8","specification":"completion","unitOfGood":"/ M Tokens"}
    #   20:["$21","$22"]
    # 模型条目用 "pricing":"$20" 引用中间那层。
    leaf: dict[str, dict[str, float]] = {}
    for m in re.finditer(
            r'([0-9a-f]+):\{"price":"([0-9.]+)","specification":"(\w+)"', text):
        line, price, spec = m.group(1), float(m.group(2)), m.group(3)
        leaf.setdefault(line, {})[spec] = price

    grouped: dict[str, dict[str, float]] = {}
    for m in re.finditer(r'([0-9a-f]+):\[((?:"\$[0-9a-f]+",?)+)\]', text):
        line = m.group(1)
        refs = re.findall(r'\$([0-9a-f]+)', m.group(2))
        merged: dict[str, float] = {}
        for r in refs:
            merged.update(leaf.get(r, {}))
        if merged:
            grouped[line] = merged

    # 合并：直接引用叶子行也能命中
    grouped.update({k: v for k, v in leaf.items() if k not in grouped})
    return grouped


def observations(html: str | None = None, *, offline: bool = False) -> dict[str, list[Observation]]:
    """
    返回 {slug: [Observation, ...]}，价格标 currency="CNY"。
    """
    html = html if html is not None else fetch_html(offline=offline)
    models = parse_models(html)
    pmap = _price_map(html)

    out: dict[str, list[Observation]] = {}

    seen: set[str] = set()
    for m in models:
        name = str(m.get("modelName") or "")
        if not name:
            continue
        # flight 流里同一模型出现两次（首屏 + 补充段），只取一次
        mkey = f"{name}|{m.get('modelId')}"
        if mkey in seen:
            continue
        seen.add(mkey)
        slug = name.split("/")[-1]
        mf = str(m.get("mf") or "").strip()

        ctx = m.get("contextLen")
        if isinstance(ctx, int) and ctx > 0:
            out.setdefault(slug, []).append(Observation(
                source=f"siliconflow:{mf or 'sf'}",
                tier="aggregator",       # 硅基流动是转售平台，不是模型作者
                field_name="context",
                value=ctx,
                provenance="pricing_url",
                url="https://siliconflow.cn/pricing",
                observed_at=None,
                valid=True,
            ))

        # 归属厂商：mf 字段就是发布方
        if mf:
            out.setdefault(slug, []).append(Observation(
                source=f"siliconflow:{mf}",
                tier="aggregator",
                field_name="vendor",
                value=mf.lower(),
                provenance="pricing_url",
                url="https://siliconflow.cn/pricing",
                observed_at=None,
                valid=True,
            ))

        # 价格：模型条目用 "pricing":"$<行号>" 引用价格表
        prices: dict[str, float] = {}
        for field in ("pricing", "priceInfo"):
            ref = m.get(field)
            if isinstance(ref, str) and ref.startswith("$"):
                prices.update(pmap.get(ref.lstrip("$"), {}))
                break
            if isinstance(ref, list):
                for item in ref:
                    if isinstance(item, dict) and item.get("specification"):
                        try:
                            prices[item["specification"]] = float(item["price"])
                        except (TypeError, ValueError):
                            pass
                break
        if not prices:
            top = m.get("price")
            try:
                if top is not None:
                    prices["completion"] = float(top)
            except (TypeError, ValueError):
                pass

        for spec, field_name in (("prompt", "price_input"),
                                 ("completion", "price_output"),
                                 ("cache", "price_cache_read")):
            val = prices.get(spec)
            if isinstance(val, (int, float)) and val > 0:
                out.setdefault(slug, []).append(Observation(
                    source=f"siliconflow:{mf or 'sf'}",
                    tier="aggregator",
                    field_name=field_name,
                    value=float(val),
                    provenance="pricing_url",
                    url="https://siliconflow.cn/pricing",
                    observed_at=None,
                    valid=True,
                    currency="CNY",      # ¥，页面明确标注
                ))

    return out
