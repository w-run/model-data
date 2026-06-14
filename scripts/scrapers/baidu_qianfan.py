#!/usr/bin/env python3
"""
百度千帆模型列表及价格爬虫

模型列表页面：
  https://cloud.baidu.com/doc/qianfan/s/rmh4stp0j
  该页面是 Gatsby 生成的静态页面，HTML 中包含完整的模型表格数据。

价格页面：
  https://cloud.baidu.com/doc/qianfan/s/wmh4sv6ya
  该页面也是 Gatsby 静态页面，HTML 中包含按量计费价格表格。

价格表格结构（第一个表格为按量计费）：
  | 模型名称 | 版本名称 | 服务内容 | 子项 | 在线推理 | 批量推理 | 单位 |

  由于 rowspan，某些行会缺少部分列。需要通过 version-name class 或 rowspan 属性
  来追踪当前版本名称。

  子项含义：
    "输入" -> 输入价格
    "输出" -> 输出价格
    "命中缓存" -> 命中缓存时的输入价格
    "搜索增强" -> 联网搜索价格（元/次）

  单位：
    "元/千tokens" -> 换算为 "元/百万tokens"（×1000）
    "元/次" -> 保持不变
"""

import re
from typing import Any

from .base import BaseScraper, register_scraper


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
    如 "ernie-char-8kernie-char-fiction-8k" -> ["ernie-char-8k", "ernie-char-fiction-8k"]
    如 "ERNIE-5.0ERNIE-5.0-Thinking-PreviewERNIE-5.0-Thinking-LatestERNIE-5.0-Thinking-Exp"
       -> ["ERNIE-5.0", "ERNIE-5.0-Thinking-Preview", "ERNIE-5.0-Thinking-Latest", "ERNIE-5.0-Thinking-Exp"]
    """
    raw = raw.strip()
    # 先尝试按空格分割
    if " " in raw:
        return [p.strip() for p in raw.split() if p.strip()]

    # 按常见前缀分割
    parts = re.split(r'(?=(?:ernie-|ERNIE-|qianfan-|Qianfan-|DeepSeek-|deepseek-))', raw)
    return [p for p in parts if p]


@register_scraper
class BaiduQianfanScraper(BaseScraper):
    vendor_id = "baidu"
    name = "百度千帆"
    doc_url = "https://cloud.baidu.com/doc/qianfan/s/rmh4stp0j"
    price_url = "https://cloud.baidu.com/doc/qianfan/s/wmh4sv6ya"

    def scrape_models(self) -> list[dict[str, Any]]:
        html = self.http_get_html(self.doc_url)

        # 提取所有表格
        tables = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
        if not tables:
            raise RuntimeError("No tables found in Baidu Qianfan docs page")

        all_models: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for table_html in tables:
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL)
            if not rows:
                continue

            # 检查表头是否包含模型相关信息
            header_cells = re.findall(r"<th[^>]*>(.*?)</th>", rows[0], re.DOTALL)
            if not header_cells:
                continue

            header_clean = [self.clean_html(c) for c in header_cells]
            has_name = any("模型名称" in h or "模型" in h for h in header_clean)
            has_id = any("接入点" in h or "model" in h.lower() or "ID" in h for h in header_clean)
            if not (has_name or has_id):
                continue

            # 解析数据行
            for row in rows[1:]:
                cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
                cells_clean = [self.clean_html(c).strip() for c in cells]

                if len(cells_clean) < 2:
                    continue

                model_name = cells_clean[0]
                model_id = cells_clean[1] if len(cells_clean) > 1 else ""

                # 过滤掉非模型行
                if not model_name or not model_id:
                    continue
                skip_keywords = ["使用场景", "核心定位", "上下文长度", "最大输出", "Token数"]
                if any(kw in model_id for kw in skip_keywords):
                    continue
                if any(kw in model_name for kw in skip_keywords):
                    continue
                if not re.search(r"[a-zA-Z]", model_id):
                    continue

                # 去重
                if model_id in seen_ids:
                    continue
                seen_ids.add(model_id)

                # 只保留百度自研的 ERNIE 系列模型和 Qianfan 系列
                if not (model_id.startswith("ernie") or model_id.startswith("qianfan")):
                    continue

                entry: dict[str, Any] = {
                    "id": model_id,
                    "name": model_name,
                }

                # 上下文长度
                if len(cells_clean) > 2:
                    ctx = self.parse_token_count(cells_clean[2])
                    if ctx:
                        entry["context_length"] = ctx

                # 最大输入
                if len(cells_clean) > 3:
                    max_in = self.parse_token_count(cells_clean[3])
                    if max_in:
                        entry["max_input"] = max_in

                # 最大输出
                if len(cells_clean) > 4:
                    max_out_str = cells_clean[4]
                    bracket_match = re.search(r"[\[【](\d+)[，,](\d+)[\]】]", max_out_str)
                    if bracket_match:
                        entry["max_output"] = int(bracket_match.group(2))
                    else:
                        max_out = self.parse_token_count(max_out_str)
                        if max_out:
                            entry["max_output"] = max_out

                # 家族和模态推断
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

                # 通用能力
                caps = entry.get("capabilities", {})
                caps["tool_call"] = True
                caps["structured_output"] = True
                entry["capabilities"] = caps

                all_models.append(entry)

        if not all_models:
            raise RuntimeError("No ERNIE models found in Baidu Qianfan docs page")

        return all_models

    def scrape_prices(self) -> dict[str, dict[str, Any]]:
        """
        从百度千帆价格页面抓取模型价格信息

        返回: {version_name: {input, output, cache_input, web_search, ...}}
        价格单位: 元/百万token（从元/千token换算×1000），搜索增强为 元/次
        """
        html = self.http_get_html(self.price_url)

        # 提取第一个表格（按量计费 - 在线推理）
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
            # 跳过表头
            if row_idx == 0:
                continue

            # 解析所有 <td> 元素，包括属性
            raw_cells = re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", row, re.DOTALL)
            cells_clean = [self.clean_html(c).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)]

            if not cells_clean:
                continue

            # 检测当前行是否有新的版本名称
            # 方式1：version-name class
            # 方式2：rowspan 且不是子项行（含有数字和字母的ID格式）
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
                    # model-name 不更新 current_version，它只是分组标题
                    pass
                elif rowspan and cell_idx <= 1:
                    # 有 rowspan 的前两列可能是版本名或模型名
                    content_clean = self.clean_html(content).strip()
                    # 检查是否像版本名（包含字母和可能的数字、连字符）
                    if content_clean and re.match(r'^[A-Za-z]', content_clean):
                        # 可能是版本名
                        if re.search(r'[A-Za-z]-\d', content_clean) or re.search(r'[A-Za-z]-[A-Za-z]', content_clean):
                            versions = _split_version_names(content_clean)
                            if versions and not any(kw in versions[0] for kw in ["推理", "输入", "输出", "命中", "搜索"]):
                                current_version = versions[0]

            if not current_version:
                continue

            # 查找子项关键词（顺序很重要！先匹配更精确的词）
            # 注意："输出（输入<=32k）"同时包含"输出"和"输入"，需要先匹配"输出"
            # 重要：需要排除"推理服务  输入Token数"这样的列，它们不是子项
            # 策略：子项cell必须以关键词开头（如"输入"、"输出"、"命中缓存"等），
            # 而不是包含关键词（如"推理服务  输入Token数"不以"输入"开头）
            sub_patterns = [
                ("搜索增强", "web_search"),
                ("命中缓存", "cache_input"),
                ("缓存命中", "cache_input"),
                ("输出", "output"),
                ("输入", "input"),
            ]
            sub_item = ""
            sub_label = ""
            sub_idx = -1
            for i, cell in enumerate(cells_clean):
                for pattern, label in sub_patterns:
                    # 子项cell必须以关键词开头
                    if cell.startswith(pattern) or cell.startswith("（") and pattern in cell:
                        sub_idx = i
                        sub_item = cell
                        sub_label = label
                        break
                if sub_idx >= 0:
                    break

            if sub_idx < 0:
                continue

            # 在线推理价格在子项之后（跳过"触发"等子子项）
            price_idx = sub_idx + 1
            if price_idx < len(cells_clean) and cells_clean[price_idx] == "触发":
                price_idx += 1

            online_price = cells_clean[price_idx] if price_idx < len(cells_clean) else ""

            # 单位：搜索"元/"开头的列
            unit = ""
            for i in range(sub_idx + 1, len(cells_clean)):
                if "元/" in cells_clean[i]:
                    unit = cells_clean[i]
                    break

            # 解析价格
            price_val = _parse_price_value(online_price)
            if price_val is None:
                continue

            # 版本名称处理
            version_key = current_version.strip()
            if not version_key:
                continue

            # 初始化该版本的价格记录
            if version_key not in prices:
                prices[version_key] = {}

            # 确定价格类型和单位
            is_per_ktoken = "千token" in unit
            is_per_time = "元/次" in unit

            if is_per_ktoken:
                price_per_m = round(price_val * 1000, 6)
            elif is_per_time:
                price_per_m = price_val
            else:
                price_per_m = round(price_val * 1000, 6)

            if sub_label == "web_search":
                # 搜索增强是元/次
                prices[version_key][sub_label] = price_per_m
            elif sub_label in ("input", "output", "cache_input"):
                # 百度有分段输入/输出价格，取最低区间的价格作为标准价格
                if sub_label not in prices[version_key]:
                    prices[version_key][sub_label] = price_per_m

        # 清理空记录
        prices = {k: v for k, v in prices.items() if v}

        # 规范化 key：转小写
        normalized_prices: dict[str, dict[str, Any]] = {}
        for k, v in prices.items():
            key_lower = k.lower()
            # 只保留百度自研的 ERNIE 和 Qianfan 系列模型
            if key_lower.startswith("ernie") or key_lower.startswith("qianfan"):
                normalized_prices[key_lower] = v

        return normalized_prices
