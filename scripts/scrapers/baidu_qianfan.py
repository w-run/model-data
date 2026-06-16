#!/usr/bin/env python3
"""
百度千帆数据源

模型列表页面：
  https://cloud.baidu.com/doc/qianfan/s/rmh4stp0j
  该页面是 Gatsby 生成的静态页面，HTML 中包含完整的模型表格数据。

价格页面：
  https://cloud.baidu.com/doc/qianfan/s/wmh4sv6ya
  该页面也是 Gatsby 静态页面，HTML 中包含按量计费价格表格。

百度千帆是厂商专属数据源（vendor_id = baidu），所有模型归属百度。
"""

import re
from typing import Any

from .base import BaseSource, register_source


def _parse_price_value(val: str) -> float | None:
    """解析价格数值字符串"""
    val = val.strip().replace(",", "").replace("，", "")
    if not val or val == "-":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _split_version_names(raw: str) -> list[str]:
    """
    分割可能合并在一起的多个版本名

    如 "ERNIE-4.5-Turbo-VL ERNIE-4.5-Turbo-VL-32K" -> ["ERNIE-4.5-Turbo-VL", "ERNIE-4.5-Turbo-VL-32K"]
    """
    raw = raw.strip()
    if " " in raw:
        return [p.strip() for p in raw.split() if p.strip()]

    parts = re.split(r'(?=(?:ernie-|ERNIE-|qianfan-|Qianfan-|DeepSeek-|deepseek-))', raw)
    return [p for p in parts if p]


@register_source
class BaiduQianfanSource(BaseSource):
    source_id = "baidu"
    name = "百度千帆"
    is_aggregator = False
    vendor_id = "baidu"  # 厂商专属数据源的厂商 ID

    doc_url = "https://cloud.baidu.com/doc/qianfan/s/rmh4stp0j"
    price_url = "https://cloud.baidu.com/doc/qianfan/s/wmh4sv6ya"

    def fetch(self, alias_map: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
        """从百度千帆文档获取模型列表"""
        models = self._fetch_models()
        return {self.vendor_id: models}

    def fetch_prices(self) -> dict[str, dict[str, Any]]:
        """从百度千帆价格页面获取模型价格"""
        return self._scrape_prices()

    def _fetch_models(self) -> list[dict[str, Any]]:
        """从文档页面抓取模型列表"""
        html = self.http_get_html(self.doc_url)

        tables = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
        if not tables:
            raise RuntimeError("No tables found in Baidu Qianfan docs page")

        all_models: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for table_html in tables:
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL)
            if not rows:
                continue

            header_cells = re.findall(r"<th[^>]*>(.*?)</th>", rows[0], re.DOTALL)
            if not header_cells:
                continue

            header_clean = [self.clean_html(c) for c in header_cells]
            has_name = any("模型名称" in h or "模型" in h for h in header_clean)
            has_id = any("接入点" in h or "model" in h.lower() or "ID" in h for h in header_clean)
            if not (has_name or has_id):
                continue

            for row in rows[1:]:
                cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
                cells_clean = [self.clean_html(c).strip() for c in cells]

                if len(cells_clean) < 2:
                    continue

                model_name = cells_clean[0]
                model_id = cells_clean[1] if len(cells_clean) > 1 else ""

                if not model_name or not model_id:
                    continue
                skip_keywords = ["使用场景", "核心定位", "上下文长度", "最大输出", "Token数"]
                if any(kw in model_id for kw in skip_keywords):
                    continue
                if any(kw in model_name for kw in skip_keywords):
                    continue
                if not re.search(r"[a-zA-Z]", model_id):
                    continue

                if model_id in seen_ids:
                    continue
                seen_ids.add(model_id)

                if not (model_id.startswith("ernie") or model_id.startswith("qianfan")):
                    continue

                entry: dict[str, Any] = {
                    "id": model_id,
                    "vendor_id": self.vendor_id,
                    "name": model_name,
                }

                if len(cells_clean) > 2:
                    ctx = self.parse_token_count(cells_clean[2])
                    if ctx:
                        entry["context_length"] = ctx

                if len(cells_clean) > 3:
                    max_in = self.parse_token_count(cells_clean[3])
                    if max_in:
                        entry["max_input"] = max_in

                if len(cells_clean) > 4:
                    max_out_str = cells_clean[4]
                    bracket_match = re.search(r"[\[【](\d+)[，,](\d+)[\]】]", max_out_str)
                    if bracket_match:
                        entry["max_output"] = int(bracket_match.group(2))
                    else:
                        max_out = self.parse_token_count(max_out_str)
                        if max_out:
                            entry["max_output"] = max_out

                if "vl" in model_id.lower() or "vision" in model_id.lower():
                    entry["modalities"] = {"input": ["text", "image"], "output": ["text"]}
                    entry["family"] = model_name.rsplit(" ", 1)[0] if " " in model_name else model_name
                elif "speed" in model_id:
                    entry["family"] = "ERNIE Speed"
                    entry["modalities"] = {"input": ["text"], "output": ["text"]}
                elif "lite" in model_id:
                    entry["family"] = "ERNIE Lite"
                    entry["modalities"] = {"input": ["text"], "output": ["text"]}
                elif "char" in model_id:
                    entry["family"] = "ERNIE Character"
                    entry["modalities"] = {"input": ["text"], "output": ["text"]}
                elif "thinking" in model_id:
                    entry["family"] = model_name
                    entry["modalities"] = {"input": ["text"], "output": ["text"]}
                    entry["capabilities"] = {"reasoning": True}
                elif "turbo" in model_id:
                    entry["family"] = model_name
                    entry["modalities"] = {"input": ["text"], "output": ["text"]}
                else:
                    entry["family"] = model_name
                    entry["modalities"] = {"input": ["text"], "output": ["text"]}

                caps = entry.get("capabilities", {})
                caps["tool_call"] = True
                caps["structured_output"] = True
                entry["capabilities"] = caps

                all_models.append(entry)

        if not all_models:
            raise RuntimeError("No ERNIE models found in Baidu Qianfan docs page")

        return all_models

    def _scrape_prices(self) -> dict[str, dict[str, Any]]:
        """从百度千帆价格页面抓取模型价格"""
        html = self.http_get_html(self.price_url)

        tables = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
        if not tables:
            raise RuntimeError("No tables found in Baidu Qianfan price page")

        table_html = tables[0]
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL)
        if not rows:
            raise RuntimeError("No rows found in price table")

        prices: dict[str, dict[str, Any]] = {}
        current_version: str | None = None

        for row_idx, row in enumerate(rows):
            if row_idx == 0:
                continue

            raw_cells = re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", row, re.DOTALL)
            cells_clean = [self.clean_html(c).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)]

            if not cells_clean:
                continue

            for cell_idx, (attrs, content) in enumerate(raw_cells):
                cls_match = re.search(r'class="([^"]*)"', attrs)
                cls_name = cls_match.group(1) if cls_match else ""
                rowspan = re.search(r'rowspan="(\d+)"', attrs)

                if "version-name" in cls_name:
                    content_clean = self.clean_html(content).strip()
                    if content_clean:
                        versions = _split_version_names(content_clean)
                        current_version = versions[0] if versions else content_clean
                elif "model-name" in cls_name:
                    pass
                elif rowspan and cell_idx <= 1:
                    content_clean = self.clean_html(content).strip()
                    if content_clean and re.match(r'^[A-Za-z]', content_clean):
                        if re.search(r'[A-Za-z]-\d', content_clean) or re.search(r'[A-Za-z]-[A-Za-z]', content_clean):
                            versions = _split_version_names(content_clean)
                            if versions and not any(kw in versions[0] for kw in ["推理", "输入", "输出", "命中", "搜索"]):
                                current_version = versions[0]

            if not current_version:
                continue

            sub_patterns = [
                ("搜索增强", "web_search"),
                ("命中缓存", "cache_input"),
                ("缓存命中", "cache_input"),
                ("输出", "output"),
                ("输入", "input"),
            ]
            sub_idx = -1
            sub_label = ""
            for i, cell in enumerate(cells_clean):
                for pattern, label in sub_patterns:
                    if cell.startswith(pattern) or (cell.startswith("（") and pattern in cell):
                        sub_idx = i
                        sub_label = label
                        break
                if sub_idx >= 0:
                    break

            if sub_idx < 0:
                continue

            price_idx = sub_idx + 1
            if price_idx < len(cells_clean) and cells_clean[price_idx] == "触发":
                price_idx += 1

            online_price = cells_clean[price_idx] if price_idx < len(cells_clean) else ""

            unit = ""
            for i in range(sub_idx + 1, len(cells_clean)):
                if "元/" in cells_clean[i]:
                    unit = cells_clean[i]
                    break

            price_val = _parse_price_value(online_price)
            if price_val is None:
                continue

            version_key = current_version.strip()
            if not version_key:
                continue

            if version_key not in prices:
                prices[version_key] = {}

            is_per_ktoken = "千token" in unit
            is_per_time = "元/次" in unit

            if is_per_ktoken:
                price_per_m = round(price_val * 1000, 6)
            elif is_per_time:
                price_per_m = price_val
            else:
                price_per_m = round(price_val * 1000, 6)

            if sub_label == "web_search":
                prices[version_key][sub_label] = price_per_m
            elif sub_label in ("input", "output", "cache_input"):
                if sub_label not in prices[version_key]:
                    prices[version_key][sub_label] = price_per_m

        prices = {k: v for k, v in prices.items() if v}

        normalized_prices: dict[str, dict[str, Any]] = {}
        for k, v in prices.items():
            key_lower = k.lower()
            if key_lower.startswith("ernie") or key_lower.startswith("qianfan"):
                normalized_prices[key_lower] = v

        return normalized_prices
