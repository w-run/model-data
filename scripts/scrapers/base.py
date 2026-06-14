#!/usr/bin/env python3
"""
网页爬虫基类

为各厂商文档页面提供统一的爬虫接口，用于从官方文档页面抓取模型列表信息。
这是对 OpenRouter API 抓取的补充渠道，特别适用于没有上架聚合平台且没有公开 API 的厂商。
"""

import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


class BaseScraper:
    """网页爬虫基类"""

    # 子类需覆盖
    vendor_id: str = ""
    name: str = ""
    doc_url: str = ""
    price_url: str = ""  # 价格页面 URL

    def http_get_html(self, url: str, timeout: int = 60, max_retries: int = 3, retry_delay: float = 5.0) -> str:
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
    def parse_token_count(s: str) -> int | None:
        """解析 token 数量字符串，如 '128k' -> 131072, '1M' -> 1048576"""
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

    def scrape_models(self) -> list[dict[str, Any]]:
        """
        抓取模型列表，子类需覆盖此方法

        返回格式:
        [
            {
                "id": "model-id",
                "name": "Model Name",
                "vendor_id": self.vendor_id,
                "source": "scraper",
                "scraper": self.__class__.__name__,
                "fetched_at": "2026-...",
                # 可选字段:
                "context_length": 131072,
                "max_input": 124000,
                "max_output": 16384,
                "modalities": {"input": ["text"], "output": ["text"]},
                "capabilities": {"tool_call": True, ...},
                "family": "Model Family",
                "description": "...",
            },
            ...
        ]
        """
        raise NotImplementedError

    def scrape_prices(self) -> dict[str, dict[str, Any]]:
        """
        抓取模型价格信息，子类可选覆盖

        返回格式:
        {
            "model-id": {
                "input": 3.2,           # 元/百万token
                "output": 16.0,         # 元/百万token
                "cache_input": 0.64,    # 元/百万token（命中缓存时的输入价格）
                "cache_output": null,   # 元/百万token（命中缓存时的输出价格，如无则为null）
                "web_search": null,     # 元/次（联网搜索价格，如无则为null）
                "completion": null,     # 补齐价格
                "reasoning": null,      # 推理价格
            },
            ...
        }

        所有价格统一为"元/百万token"（或"元/次"对于web_search等非token计费项）。
        没有数据的价格项为 null。
        """
        return {}

    def run(self) -> list[dict[str, Any]]:
        """执行爬虫并返回标准化的模型列表"""
        now = datetime.now(timezone.utc).isoformat()
        models = self.scrape_models()
        for m in models:
            m["vendor_id"] = self.vendor_id
            m["source"] = "scraper"
            m["scraper"] = self.__class__.__name__
            m["fetched_at"] = now
        return models

    def run_prices(self) -> dict[str, dict[str, Any]]:
        """执行价格爬虫并返回价格数据"""
        if not self.price_url:
            return {}
        try:
            return self.scrape_prices()
        except Exception as e:
            print(f"    Price scraping failed for {self.name}: {e}", file=__import__("sys").stderr)
            return {}


# 注册所有爬虫
SCRAPERS: dict[str, type[BaseScraper]] = {}


def register_scraper(cls: type[BaseScraper]) -> type[BaseScraper]:
    """注册爬虫类"""
    SCRAPERS[cls.vendor_id] = cls
    return cls


def run_scraper(vendor_id: str) -> list[dict[str, Any]]:
    """运行指定厂商的爬虫"""
    cls = SCRAPERS.get(vendor_id)
    if not cls:
        raise ValueError(f"No scraper registered for vendor: {vendor_id}")
    return cls().run()


def run_all_scrapers(vendor_ids: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """运行所有（或指定）爬虫"""
    result: dict[str, list[dict[str, Any]]] = {}
    target_ids = vendor_ids or sorted(SCRAPERS.keys())

    for vid in target_ids:
        cls = SCRAPERS.get(vid)
        if not cls:
            continue

        scraper = cls()
        print(f"    {scraper.name}: scraping {scraper.doc_url}...", end=" ", flush=True)
        try:
            models = scraper.run()
            result[vid] = models
            print(f"OK ({len(models)} models)")
        except Exception as e:
            print(f"FAIL: {e}")
            result[vid] = []

    return result


def run_all_price_scrapers(vendor_ids: list[str] | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """运行所有（或指定）价格爬虫，返回 {vendor_id: {model_id: price_data}}"""
    result: dict[str, dict[str, dict[str, Any]]] = {}
    target_ids = vendor_ids or sorted(SCRAPERS.keys())

    for vid in target_ids:
        cls = SCRAPERS.get(vid)
        if not cls:
            continue

        scraper = cls()
        if not scraper.price_url:
            continue

        print(f"    {scraper.name}: scraping prices from {scraper.price_url}...", end=" ", flush=True)
        try:
            prices = scraper.run_prices()
            if prices:
                result[vid] = prices
                print(f"OK ({len(prices)} models with prices)")
            else:
                print("No prices found")
        except Exception as e:
            print(f"FAIL: {e}")

    return result
