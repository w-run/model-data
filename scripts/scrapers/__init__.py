"""
网页爬虫模块

为各厂商文档页面提供爬虫，从官方文档抓取模型列表信息和价格信息。
特别适用于没有上架聚合平台且没有公开 API 的厂商。

注意：硅基流动(SiliconFlow)是中转商/聚合商，其数据通过 build.py 中的
fetch_siliconflow() 函数获取，不走 scraper 通道。
"""

from .base import BaseScraper, SCRAPERS, register_scraper, run_scraper, run_all_scrapers, run_all_price_scrapers

# 导入各厂商爬虫模块（触发 @register_scraper 装饰器注册）
from .baidu_qianfan import BaiduQianfanScraper
from .volcengine_doubao import VolcengineDoubaoScraper

__all__ = [
    "BaseScraper",
    "SCRAPERS",
    "register_scraper",
    "run_scraper",
    "run_all_scrapers",
    "run_all_price_scrapers",
    "BaiduQianfanScraper",
    "VolcengineDoubaoScraper",
]
