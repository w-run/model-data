"""
数据源模块

所有数据源统一继承 BaseSource，提供标准化的接口：
  - fetch(alias_map) -> {vendor_id: [model_dict, ...]}
  - fetch_prices() -> {model_id: {input, output, ...}}  (厂商专属数据源)

数据源分两类：
  1. 聚合平台（is_aggregator=True）：如 OpenRouter、SiliconFlow
     模型来自多个厂商，fetch() 返回的模型分属不同 vendor_id
  2. 厂商专属（is_aggregator=False）：如百度千帆、火山引擎豆包
     模型全部属于同一厂商，fetch() 返回统一归属该 vendor_id
"""

from .base import BaseSource, SOURCES, register_source, run_source, run_all_sources, run_all_price_sources

# 导入各数据源模块（触发 @register_source 装饰器注册）
from .openrouter import OpenRouterSource
from .siliconflow import SiliconFlowSource
from .baidu_qianfan import BaiduQianfanSource
from .volcengine_doubao import VolcengineDoubaoSource

__all__ = [
    "BaseSource",
    "SOURCES",
    "register_source",
    "run_source",
    "run_all_sources",
    "run_all_price_sources",
    "OpenRouterSource",
    "SiliconFlowSource",
    "BaiduQianfanSource",
    "VolcengineDoubaoSource",
]
