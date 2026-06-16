#!/usr/bin/env python3
"""
数据源基类

所有数据源（API 平台、网页爬虫、本地文件）统一继承 BaseSource，
提供标准化的接口和输出格式。

数据源分两类：
  1. 聚合平台（如 OpenRouter、SiliconFlow）—— 模型来自多个厂商，
     fetch() 返回的模型分属不同 vendor_id
  2. 厂商专属（如百度千帆、火山引擎豆包）—— 模型全部属于同一厂商，
     fetch() 返回的模型统一归属该 vendor_id

输出格式统一为:
  {vendor_id: [model_dict, ...]}

model_dict 标准字段:
  必需:
    id:           str   模型 ID（小写短横线分隔，不含厂商前缀）
    name:         str   模型显示名
    vendor_id:    str   归属厂商 ID
  可选:
    model_aliases: list[str]  别名列表（全部小写）
    context_length: int       上下文窗口大小
    max_output_tokens: int    最大输出 token 数
    modalities:   dict        {"input": [...], "output": [...]}
    capabilities: dict        {"tool_call": True, "reasoning": True, ...}
    price:        dict        {"unit": "CNY/Mtok"|"USD/Mtok", "input": ..., "output": ...}
    knowledge_cutoff: str     知识截止日期
    open_weights: bool        是否开源权重
    family:       str         模型家族
    description:  str         描述
"""

import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


class BaseSource:
    """数据源基类"""

    # 子类需覆盖
    source_id: str = ""     # 数据源唯一标识（如 "openrouter", "siliconflow", "baidu"）
    name: str = ""          # 数据源显示名
    is_aggregator: bool = False  # 是否为聚合平台（模型来自多厂商）

    def fetch(self, alias_map: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
        """
        获取模型数据

        alias_map: alias -> vendor_id 的映射（用于将平台上的 org 前缀匹配到厂商）
        返回: {vendor_id: [model_dict, ...]}

        子类必须覆盖此方法
        """
        raise NotImplementedError

    # ─── 工具方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def http_get_json(
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> Any:
        """发送 GET 请求并返回 JSON 响应，支持自动重试"""
        last_error = None
        for attempt in range(max_retries):
            req = urllib.request.Request(url, headers=headers or {})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                last_error = RuntimeError(f"HTTP {e.code} for {url}: {body[:500]}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
            except urllib.error.URLError as e:
                last_error = RuntimeError(f"URL error for {url}: {e.reason}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))

        raise last_error or RuntimeError(f"Failed to fetch {url} after {max_retries} retries")

    def http_get_html(
        self,
        url: str,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> str:
        """获取网页 HTML 内容，支持自动重试"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        last_error = None
        for attempt in range(max_retries):
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as e:
                last_error = RuntimeError(f"HTTP {e.code} for {url}: {e.reason}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
            except urllib.error.URLError as e:
                last_error = RuntimeError(f"URL error for {url}: {e.reason}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))

        raise last_error or RuntimeError(f"Failed to fetch {url} after {max_retries} retries")

    @staticmethod
    def clean_html(text: str) -> str:
        """移除 HTML 标签"""
        return re.sub(r"<[^>]+>", "", text).strip()

    @staticmethod
    def normalize_id(raw_id: str) -> str:
        """将任意格式的 ID 转为小写短横线分隔式命名"""
        s = raw_id.replace("/", "-").replace("_", "-").replace(" ", "-")
        s = re.sub(r"-+", "-", s)
        s = s.strip("-")
        return s.lower()

    @staticmethod
    def strip_org_prefix(raw_id: str) -> str:
        """去掉 org/ 前缀"""
        if "/" in raw_id:
            return raw_id.split("/", 1)[1]
        return raw_id

    @staticmethod
    def parse_token_count(s: str) -> int | None:
        """解析 token 数量字符串，如 '128k' -> 131072"""
        if not s:
            return None
        s = s.strip().lower().replace(",", "").replace("，", "")
        multipliers = {"k": 1024, "m": 1048576, "w": 10000}
        for suffix, mult in multipliers.items():
            if s.endswith(suffix):
                try:
                    return int(float(s[:-1]) * mult)
                except ValueError:
                    return None
        try:
            return int(s)
        except ValueError:
            return None


# ─── 注册与调度 ──────────────────────────────────────────────────────────────

SOURCES: dict[str, type[BaseSource]] = {}


def register_source(cls: type[BaseSource]) -> type[BaseSource]:
    """注册数据源类"""
    SOURCES[cls.source_id] = cls
    return cls


def run_source(source_id: str, alias_map: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    """运行指定数据源"""
    cls = SOURCES.get(source_id)
    if not cls:
        raise ValueError(f"No source registered: {source_id}")
    return cls().fetch(alias_map)


def run_all_sources(
    source_ids: list[str] | None = None,
    alias_map: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """运行所有（或指定）数据源，合并结果"""
    alias_map = alias_map or {}
    result: dict[str, list[dict[str, Any]]] = {}
    target_ids = source_ids if source_ids is not None else sorted(SOURCES.keys())

    for sid in target_ids:
        cls = SOURCES.get(sid)
        if not cls:
            continue

        source = cls()
        print(f"  [{source.source_id}] {source.name}: fetching...", end=" ", flush=True)
        try:
            data = source.fetch(alias_map)
            model_count = sum(len(v) for v in data.values())
            for vid, models in data.items():
                result.setdefault(vid, []).extend(models)
            print(f"OK ({model_count} models)")
        except Exception as e:
            print(f"FAIL: {e}")

    return result


def fetch_prices(source_id: str) -> dict[str, dict[str, Any]]:
    """
    从数据源获取价格数据

    仅厂商专属数据源支持此方法。
    返回: {model_id: {input, output, cache_input, ...}}
    """
    cls = SOURCES.get(source_id)
    if not cls:
        raise ValueError(f"No source registered: {source_id}")
    source = cls()
    if not hasattr(source, "fetch_prices"):
        return {}
    return source.fetch_prices()


def run_all_price_sources(
    source_ids: list[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    运行所有支持价格获取的数据源

    返回: {vendor_id: {model_id: {input, output, ...}}}
    """
    result: dict[str, dict[str, dict[str, Any]]] = {}
    target_ids = source_ids if source_ids is not None else sorted(SOURCES.keys())

    for sid in target_ids:
        cls = SOURCES.get(sid)
        if not cls:
            continue

        source = cls()
        if not hasattr(source, "fetch_prices") or source.is_aggregator:
            continue

        print(f"  [{source.source_id}] {source.name}: fetching prices...", end=" ", flush=True)
        try:
            prices = source.fetch_prices()
            if prices:
                # 厂商专属数据源，vendor_id 就是 source.vendor_id
                vid = getattr(source, "vendor_id", source.source_id)
                result[vid] = prices
                print(f"OK ({len(prices)} models)")
            else:
                print("No prices found")
        except Exception as e:
            print(f"FAIL: {e}")

    return result
