#!/usr/bin/env python3
"""
火山引擎豆包数据源

模型列表页面：
  https://www.volcengine.com/docs/82379/1330310
  SPA 应用，HTML 中通过 window._ROUTER_DATA 注入了文档内容。

价格页面：
  https://www.volcengine.com/docs/82379/1544106?lang=zh
  同样通过 window._ROUTER_DATA 注入，内容为 MD 格式（Markdown 表格）。

火山引擎豆包是厂商专属数据源（vendor_id = bytedance），所有模型归属字节跳动。
"""

import json
import re
from typing import Any

from .base import BaseSource, register_source


def _extract_router_data_json(html: str) -> dict[str, Any]:
    """从火山引擎 HTML 中提取 window._ROUTER_DATA JSON"""
    start = html.find("window._ROUTER_DATA")
    if start < 0:
        raise RuntimeError("window._ROUTER_DATA not found in HTML")

    eq = html.find("=", start)
    json_start = eq + 1
    while json_start < len(html) and html[json_start] in " \t\n":
        json_start += 1

    depth = 0
    i = json_start
    while i < len(html):
        ch = html[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        elif ch == '"':
            i += 1
            while i < len(html) and html[i] != '"':
                if html[i] == "\\":
                    i += 1
                i += 1
        i += 1

    json_str = html[json_start : i + 1]
    return json.loads(json_str)


def _extract_md_content(html: str) -> str:
    """从火山引擎 HTML 中提取 MDContent"""
    router_data = _extract_router_data_json(html)
    loader_data = router_data.get("loaderData", {})

    page_key = None
    for key in loader_data:
        if "(docid" in key:
            page_key = key
            break

    if not page_key:
        raise RuntimeError("No doc page key found in loaderData")

    page_data = loader_data[page_key]
    cur_doc = page_data.get("curDoc", {})
    return cur_doc.get("MDContent", "")


def _extract_ordered_blocks(router_data: dict[str, Any]) -> list[tuple[str, str]]:
    """从 _ROUTER_DATA 中提取有序的文本块列表"""
    page_key = None
    for key in router_data.get("loaderData", {}):
        if "(docid" in key:
            page_key = key
            break

    if not page_key:
        raise RuntimeError("No doc page key found in loaderData")

    page_data = router_data["loaderData"][page_key]
    cur_doc = page_data.get("curDoc", {})
    content_str = cur_doc.get("Content", "")

    if not content_str:
        raise RuntimeError("curDoc.Content is empty")

    content = json.loads(content_str)
    data = content.get("data", {})

    ordered_blocks: list[tuple[str, str]] = []
    for key, val in data.items():
        if isinstance(val, dict) and val.get("zoneType") == "Z":
            ops = val.get("ops", [])
            text_parts: list[str] = []
            for op in ops:
                if "insert" in op:
                    t = op["insert"]
                    if isinstance(t, str):
                        text_parts.append(t)
            text = "".join(text_parts).strip()
            if text:
                ordered_blocks.append((key, text))

    return ordered_blocks


def _extract_full_text(router_data: dict[str, Any]) -> str:
    """从 _ROUTER_DATA 中提取文档全文"""
    page_key = None
    for key in router_data.get("loaderData", {}):
        if "(docid" in key:
            page_key = key
            break

    if not page_key:
        raise RuntimeError("No doc page key found in loaderData")

    page_data = router_data["loaderData"][page_key]
    cur_doc = page_data.get("curDoc", {})
    content_str = cur_doc.get("Content", "")

    if not content_str:
        raise RuntimeError("curDoc.Content is empty")

    content = json.loads(content_str)
    data = content.get("data", {})

    text_parts: list[str] = []
    for _key, val in data.items():
        if isinstance(val, dict) and "ops" in val:
            for op in val["ops"]:
                if "insert" in op:
                    text = op["insert"]
                    if isinstance(text, str):
                        text_parts.append(text)
                    elif isinstance(text, dict):
                        text_parts.append(json.dumps(text, ensure_ascii=False))

    return "".join(text_parts)


def _parse_price_value(val: str) -> float | None:
    """解析价格数值字符串"""
    val = val.strip().replace("\\", "").replace(",", "").replace("，", "")
    if not val or val == "-":
        return None
    try:
        return float(val)
    except ValueError:
        return None


_MODEL_ID_PATTERN = re.compile(r"\*(doubao[a-zA-Z0-9\-]+(?:-\d+k)?)", re.IGNORECASE)


def _convert_version_hyphens(mid: str) -> str:
    """将模型 ID 中的版本号连字符转为点号"""
    version_prefixes = [
        "doubao-seed-",
        "doubao-seedance-",
        "doubao-seedream-",
        "doubao-seed3d-",
    ]
    for prefix in version_prefixes:
        m = re.match(r"(" + re.escape(prefix) + r")(\d+)-(\d+)(.*)", mid)
        if m:
            return f"{m.group(1)}{m.group(2)}.{m.group(3)}{m.group(4)}"

    m = re.match(r"(doubao-)(\d+)-(\d+)(.*)", mid)
    if m and "seed" not in mid:
        return f"{m.group(1)}{m.group(2)}.{m.group(3)}{m.group(4)}"

    return mid


def normalize_doubao_id(model_id: str) -> str:
    """将火山引擎模型ID规范化为基础ID格式"""
    mid = model_id.lower().strip()
    mid = re.sub(r"-preview(?:-\d+)?$", "", mid)
    mid = re.sub(r"-\d{6}$", "", mid)
    mid = _convert_version_hyphens(mid)
    return mid


@register_source
class VolcengineDoubaoSource(BaseSource):
    source_id = "volcengine-doubao"
    name = "火山引擎豆包"
    is_aggregator = False
    vendor_id = "bytedance"

    doc_url = "https://www.volcengine.com/docs/82379/1330310"
    price_url = "https://www.volcengine.com/docs/82379/1544106?lang=zh"

    def fetch(self, alias_map: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
        """从火山引擎文档获取豆包模型列表"""
        models = self._fetch_models()
        return {self.vendor_id: models}

    def fetch_prices(self) -> dict[str, dict[str, Any]]:
        """从火山引擎价格页面获取豆包模型价格"""
        return self._scrape_prices()

    def _fetch_models(self) -> list[dict[str, Any]]:
        """从文档页面抓取模型列表"""
        html = self.http_get_html(self.doc_url)
        router_data = _extract_router_data_json(html)

        ordered_blocks = _extract_ordered_blocks(router_data)
        model_specs = self._extract_model_specs_from_blocks(ordered_blocks)

        full_text = _extract_full_text(router_data)
        model_apis = self._extract_model_apis_from_text(full_text)

        all_models: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        matches = list(_MODEL_ID_PATTERN.finditer(full_text))
        for m in matches:
            model_id = m.group(1)
            clean_id = re.sub(r"[^a-zA-Z0-9\-].*$", "", model_id)

            normalized_id = normalize_doubao_id(clean_id)

            if normalized_id in seen_ids:
                continue
            seen_ids.add(normalized_id)

            after_text = full_text[m.end() : m.end() + 800]

            entry: dict[str, Any] = {
                "id": normalized_id,
                "vendor_id": self.vendor_id,
                "name": self._id_to_display_name(normalized_id),
            }

            aliases: list[str] = []
            seen_alias: set[str] = {normalized_id}

            clean_lower = clean_id.lower()
            if clean_lower != normalized_id and clean_lower not in seen_alias:
                aliases.append(clean_lower)
                seen_alias.add(clean_lower)

            no_date = re.sub(r"-\d{6}$", "", clean_lower)
            no_date = re.sub(r"-preview(?:-\d+)?$", "", no_date)
            if no_date != normalized_id and no_date != clean_lower and no_date not in seen_alias:
                aliases.append(no_date)
                seen_alias.add(no_date)

            if aliases:
                entry["model_aliases"] = aliases

            specs = model_specs.get(clean_id, {})
            for k, v in specs.items():
                entry[k] = v

            capabilities: dict[str, Any] = {}
            if "深度思考" in after_text[:400]:
                capabilities["reasoning"] = True
            if "工具调用" in after_text[:400]:
                capabilities["tool_call"] = True
            if "结构化输出" in after_text[:400]:
                capabilities["structured_output"] = True
            if "多模态理解" in after_text[:400]:
                capabilities["vision"] = True

            api_info = model_apis.get(clean_id, [])
            if api_info:
                entry["supported_apis"] = api_info

            family, modalities = self._infer_family_and_modality(normalized_id)
            entry["family"] = family
            entry["modalities"] = modalities

            if capabilities:
                entry["capabilities"] = capabilities

            generation_specs = self._extract_generation_specs(clean_id, ordered_blocks)
            if generation_specs:
                entry["generation_specs"] = generation_specs

            all_models.append(entry)

        if not all_models:
            raise RuntimeError("No Doubao models found in Volcengine docs page")

        return all_models

    def _scrape_prices(self) -> dict[str, dict[str, Any]]:
        """从价格页面抓取模型价格"""
        html = self.http_get_html(self.price_url)
        md_content = _extract_md_content(html)

        if not md_content:
            raise RuntimeError("MDContent is empty in price page")

        prices: dict[str, dict[str, Any]] = {}

        regular_section = self._extract_section(md_content, "在线推理（常规）")
        if regular_section:
            self._parse_price_table(regular_section, prices)

        search_section = self._extract_section(md_content, "联网内容插件")
        if search_section:
            self._parse_search_prices(search_section, prices)

        return prices

    def _extract_section(self, md_content: str, section_title: str) -> str:
        """从 MD 内容中提取指定章节的内容"""
        pattern = re.compile(r"##\s*" + re.escape(section_title), re.IGNORECASE)
        m = pattern.search(md_content)
        if not m:
            pattern = re.compile(r"###\s*" + re.escape(section_title), re.IGNORECASE)
            m = pattern.search(md_content)
        if not m:
            return ""

        start = m.end()
        next_section = re.search(r"\n(?:##|###|####)\s+", md_content[start:])
        if next_section:
            end = start + next_section.start()
        else:
            end = len(md_content)

        return md_content[start:end]

    def _parse_price_table(self, section: str, prices: dict[str, dict[str, Any]]) -> None:
        """解析价格表格"""
        lines = section.strip().split("\n")
        current_model: str | None = None

        for line in lines:
            line = line.strip()
            if not line.startswith("|"):
                continue

            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c]

            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue

            if any("模型名称" in c for c in cells):
                continue

            if len(cells) < 4:
                continue

            model_name = cells[0].replace("\\-", "-").replace("\\", "").strip()

            if model_name and re.match(r"[a-zA-Z]", model_name):
                current_model = model_name

            if not current_model:
                continue

            model_id = current_model.lower().strip()
            normalized_id = self._normalize_price_model_id(model_id)

            if len(cells) >= 8:
                input_price = _parse_price_value(cells[2])
                cache_hit_price = _parse_price_value(cells[5])
                output_price = _parse_price_value(cells[7])
            elif len(cells) >= 5:
                input_price = _parse_price_value(cells[2])
                cache_hit_price = None
                output_price = _parse_price_value(cells[4])
            else:
                continue

            if normalized_id not in prices:
                prices[normalized_id] = {}

            if input_price is not None and "input" not in prices[normalized_id]:
                prices[normalized_id]["input"] = input_price
            if output_price is not None and "output" not in prices[normalized_id]:
                prices[normalized_id]["output"] = output_price
            if cache_hit_price is not None and "cache_input" not in prices[normalized_id]:
                prices[normalized_id]["cache_input"] = cache_hit_price

    def _parse_search_prices(self, section: str, prices: dict[str, dict[str, Any]]) -> None:
        """解析联网搜索价格"""
        lines = section.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line.startswith("|"):
                continue

            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c]

            if len(cells) < 2:
                continue

            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue
            if "服务项" in cells[0]:
                continue

            if "联网资源" in cells[0]:
                price = _parse_price_value(cells[1])
                if price is not None:
                    for model_id in prices:
                        prices[model_id]["web_search"] = price

    @staticmethod
    def _normalize_price_model_id(model_id: str) -> str:
        return model_id.lower().strip()

    def _extract_model_specs_from_blocks(
        self, ordered_blocks: list[tuple[str, str]]
    ) -> dict[str, dict[str, Any]]:
        """从有序 Z-block 中提取每个模型的规格参数"""
        model_positions: dict[str, list[int]] = {}

        for idx, (key, text) in enumerate(ordered_blocks):
            for m in _MODEL_ID_PATTERN.finditer(text):
                mid = m.group(1)
                clean_id = re.sub(r"[^a-zA-Z0-9\-].*$", "", mid)
                if clean_id not in model_positions:
                    model_positions[clean_id] = []
                model_positions[clean_id].append(idx)

        results: dict[str, dict[str, Any]] = {}

        for model_id, positions in model_positions.items():
            best_data: dict[str, Any] = {}

            for block_idx in positions:
                search_blocks = ordered_blocks[block_idx : block_idx + 15]
                combined = "\n".join(t for _, t in search_blocks)

                ctx_m = re.search(r"上下文窗口[：:]\s*(\S+)", combined)
                max_in_m = re.search(r"最大输入[：:]\s*(\S+)", combined)
                max_out_m = re.search(r"最大回答[^：:\n]*[：:]\s*(\S+)", combined)
                thinking_m = re.search(r"最大思维链[：:]\s*(\S+)", combined)

                has_data = any([ctx_m, max_in_m, max_out_m, thinking_m])
                if has_data:
                    if ctx_m:
                        ctx = self.parse_token_count(ctx_m.group(1))
                        if ctx:
                            best_data["context_length"] = ctx
                    if max_in_m:
                        max_in = self.parse_token_count(max_in_m.group(1))
                        if max_in:
                            best_data["max_input"] = max_in
                    if max_out_m:
                        max_out = self.parse_token_count(max_out_m.group(1))
                        if max_out:
                            best_data["max_output"] = max_out
                    if thinking_m:
                        thinking = self.parse_token_count(thinking_m.group(1))
                        if thinking:
                            best_data["max_thinking"] = thinking
                    break

            results[model_id] = best_data

        return results

    def _extract_model_apis_from_text(
        self, full_text: str
    ) -> dict[str, list[str]]:
        """从全文中提取每个模型支持的 API 类型"""
        result: dict[str, list[str]] = {}

        matches = list(_MODEL_ID_PATTERN.finditer(full_text))
        seen: set[str] = set()

        for m in matches:
            model_id = m.group(1)
            clean_id = re.sub(r"[^a-zA-Z0-9\-].*$", "", model_id)

            if clean_id in seen:
                continue
            seen.add(clean_id)

            after_text = full_text[m.end() : m.end() + 600]
            apis: list[str] = []

            if "Chat API" in after_text[:400]:
                apis.append("chat")
            if "Responses API" in after_text[:400]:
                apis.append("responses")
            if "Batch API" in after_text[:400]:
                apis.append("batch")
            if "Context API" in after_text[:400]:
                apis.append("context")

            if apis:
                result[clean_id] = apis

        return result

    def _extract_generation_specs(
        self, model_id: str, ordered_blocks: list[tuple[str, str]]
    ) -> dict[str, Any] | None:
        """从 Z-block 中提取视频/图片生成模型的产物规格"""
        mid = model_id.lower()

        is_video = "seedance" in mid
        is_image = "seedream" in mid
        is_3d = "seed3d" in mid

        if not (is_video or is_image or is_3d):
            return None

        target_idx = None
        for idx, (key, text) in enumerate(ordered_blocks):
            if model_id in text or re.sub(r"[^a-zA-Z0-9\-].*$", "", model_id) in text:
                target_idx = idx
                break

        if target_idx is None:
            return None

        search_blocks = ordered_blocks[target_idx : target_idx + 20]
        combined = "\n".join(t for _, t in search_blocks)

        specs: dict[str, Any] = {}

        res_m = re.search(r"分辨率[：:]\s*([^\n*]+)", combined)
        if res_m:
            specs["resolution"] = res_m.group(1).strip()

        fps_m = re.search(r"帧率[：:]\s*([^\n*]+)", combined)
        if fps_m:
            specs["fps"] = fps_m.group(1).strip()

        dur_m = re.search(r"时长[：:]\s*([^\n*]+)", combined)
        if dur_m:
            specs["duration"] = dur_m.group(1).strip()

        fmt_m = re.search(r"视频格式[：:]\s*([^\n*]+)", combined)
        if fmt_m:
            specs["format"] = fmt_m.group(1).strip()

        rpm_m = re.search(r"最大 RPM[：:]\s*(\d+)", combined)
        if rpm_m:
            specs["max_rpm"] = int(rpm_m.group(1))

        conc_m = re.search(r"最大并发[：:]\s*(\d+)", combined)
        if conc_m:
            specs["max_concurrency"] = int(conc_m.group(1))

        tri_m = re.search(r"三角面模型[：:]\s*([^\n*]+)", combined)
        if tri_m:
            specs["triangle_model"] = tri_m.group(1).strip()

        return specs if specs else None

    @staticmethod
    def _id_to_display_name(model_id: str) -> str:
        """将规范化后的模型 ID 转换为显示名称"""
        name = model_id
        name = re.sub(r"-\d{6}$", "", name)
        name = name.replace("-", " ")
        name = name.title()
        name = re.sub(r"(\d)\s+(\d)\s+", r"\1.\2 ", name)
        name = re.sub(r"(\d)\s+(\d)$", r"\1.\2", name)
        return name

    @staticmethod
    def _infer_family_and_modality(model_id: str) -> tuple[str, dict[str, list[str]]]:
        """从规范化后的模型 ID 推断家族和模态"""
        mid = model_id.lower()

        if "seedance" in mid:
            if "2.0" in mid:
                return "Doubao Seedance 2.0", {"input": ["text", "image"], "output": ["video"]}
            return "Doubao Seedance", {"input": ["text", "image"], "output": ["video"]}

        if "seedream" in mid:
            return "Doubao Seedream", {"input": ["text", "image"], "output": ["image"]}

        if "seed3d" in mid:
            return "Doubao Seed3D", {"input": ["text", "image"], "output": ["3d_model"]}

        if "embedding" in mid:
            return "Doubao Embedding", {"input": ["text", "image"], "output": ["embedding"]}

        if "seed" in mid:
            if "2.0" in mid:
                if "pro" in mid:
                    return "Doubao Seed 2.0 Pro", {"input": ["text", "image"], "output": ["text"]}
                if "lite" in mid:
                    return "Doubao Seed 2.0 Lite", {"input": ["text"], "output": ["text"]}
                if "mini" in mid:
                    return "Doubao Seed 2.0 Mini", {"input": ["text"], "output": ["text"]}
                if "code" in mid:
                    return "Doubao Seed 2.0 Code", {"input": ["text"], "output": ["text"]}
                if "character" in mid:
                    return "Doubao Seed Character", {"input": ["text"], "output": ["text"]}
                return "Doubao Seed 2.0", {"input": ["text"], "output": ["text"]}
            if "1.8" in mid:
                return "Doubao Seed 1.8", {"input": ["text"], "output": ["text"]}
            if "1.6" in mid:
                if "flash" in mid:
                    return "Doubao Seed 1.6 Flash", {"input": ["text"], "output": ["text"]}
                if "vision" in mid:
                    return "Doubao Seed 1.6 Vision", {"input": ["text", "image"], "output": ["text"]}
                if "lite" in mid:
                    return "Doubao Seed 1.6 Lite", {"input": ["text"], "output": ["text"]}
                if "code" in mid:
                    return "Doubao Seed Code", {"input": ["text"], "output": ["text"]}
                if "character" in mid:
                    return "Doubao Seed Character", {"input": ["text"], "output": ["text"]}
                return "Doubao Seed 1.6", {"input": ["text"], "output": ["text"]}

        if "code" in mid:
            return "Doubao Code", {"input": ["text"], "output": ["text"]}

        if "character" in mid:
            return "Doubao Character", {"input": ["text"], "output": ["text"]}

        if "translation" in mid:
            return "Doubao Translation", {"input": ["text"], "output": ["text"]}

        if "vision" in mid:
            return "Doubao Vision", {"input": ["text", "image"], "output": ["text"]}

        if "1.5" in mid:
            if "pro" in mid:
                if "256k" in mid:
                    return "Doubao 1.5 Pro 256K", {"input": ["text"], "output": ["text"]}
                return "Doubao 1.5 Pro", {"input": ["text"], "output": ["text"]}
            if "lite" in mid:
                return "Doubao 1.5 Lite", {"input": ["text"], "output": ["text"]}
            if "vision" in mid:
                return "Doubao 1.5 Vision Pro", {"input": ["text", "image"], "output": ["text"]}
            return "Doubao 1.5", {"input": ["text"], "output": ["text"]}

        if "pro" in mid:
            return "Doubao Pro", {"input": ["text"], "output": ["text"]}
        if "lite" in mid:
            return "Doubao Lite", {"input": ["text"], "output": ["text"]}

        return "Doubao", {"input": ["text"], "output": ["text"]}
