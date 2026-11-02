#!/usr/bin/env python3
"""
计价币种处理

为什么必须保留原币种
--------------------
中国厂商按人民币定价，且价格是干净的整数：

    智谱 GLM-5.3    输入 8 元/M      输出 30 元/M
    DeepSeek 国内    输入 0.5 元/M    输出 2 元/M

一旦被第三方换算成美元，就会出现这种尾巴：

    models.dev 报 zhipuai glm-5.3  input = 1.19516 USD
    反推回人民币（×7.1）= 8.49 元   ← 原始整数被汇率抹掉了

再换算回去永远对不上原价，且随汇率漂移。所以：

  **有人民币渠道时优先用人民币原始值，不换算。**

一个容易忽略的点
----------------
同一个模型的「国际站」和「国内站」定价**不是简单汇率关系**：

    alibaba    qwen3-8b  0.18 USD
    alibaba-cn qwen3-8b  0.072 USD   ← 比值 2.5，不是 7.1

说明两边是独立定价策略，不能拿一边换算另一边的。

产出策略
--------
主字段 `price` 保持既有格式（`unit: "USD/Mtok"`），避免破坏已对接的消费方。
若拿到人民币原始价，额外给出：

    "price": {"unit": "CNY/Mtok", "input": 8, "output": 30}
    "price_usd": {"unit": "USD/Mtok", "input": 1.13, ...}   ← 换算值，仅供参考

即：主字段优先人民币，美元退居辅助。
"""

from __future__ import annotations

# 汇率仅用于「人民币 → 美元」的单向换算，且只作参考值。
# 不用于反向还原人民币价格 —— 那必然失真。
DEFAULT_USD_CNY = 7.1

# 国内运营商：这些渠道的定价视为人民币
CNY_CHANNELS = {
    "alibaba-cn", "dashscope", "zhipuai", "zhipu", "bigmodel",
    "moonshot-cn", "kimi-cn", "siliconflow-cn", "volcengine",
    "tencent", "hunyuan", "baidu", "qianfan", "minimax-cn",
    "stepfun", "baichuan", "sensetime", "sensenova", "modelscope",
    "infini", "lingyiwanwu", "01ai", "shanghai-ai-lab",
}

# 这些是国际站，明确用美元（哪怕厂商是中国公司）
INTL_CHANNELS = {
    "alibaba", "dashscope-intl", "moonshotai", "moonshot", "kimi",
    "zhipu", "minimax", "siliconflow", "stepfun-intl",
}


def is_cny_channel(name: str) -> bool:
    """判断一个 provider / 渠道名是否按人民币计价。"""
    n = str(name or "").strip().lower()
    if not n:
        return False
    if n in INTL_CHANNELS:
        return False
    if n in CNY_CHANNELS:
        return True
    # 后缀/包含判定：xxx-cn 视为国内站
    if n.endswith("-cn") or n.endswith("_cn"):
        return True
    return False


def channel_currency(provider: str) -> str:
    return "CNY" if is_cny_channel(provider) else "USD"


def cny_to_usd(cny: float, rate: float = DEFAULT_USD_CNY) -> float:
    """人民币 → 美元（仅作参考换算）。"""
    if not isinstance(cny, (int, float)) or cny <= 0:
        return cny  # type: ignore[return-value]
    return round(float(cny) / rate, 6)


def prefer_currency(has_cny: bool) -> str:
    """
    主字段该用哪种币种。

    有人民币原始价就用人民币 —— 那是厂商真实报价，没有汇率误差。
    """
    return "CNY" if has_cny else "USD"
