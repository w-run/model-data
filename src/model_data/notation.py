#!/usr/bin/env python3
"""
容量值的记法变体

同一容量有两种业界写法：

    2 的幂记法        十进制记法
    1_048_576    →   1_024_000      (1M)
      393_216    →     384_000      (384K)
      262_144    →     256_000      (256K)

为什么向下取整更稳妥
--------------------
1. 声明值本来就是理论上限。真实调用中 tokenizer 边界、system prompt、
   特殊 token 都会占位，永远用不满，所以取整不会损失实际可用空间。
2. 十进制记法是配置时真正会写的数字，人类可读性更好。
3. 只向下取整（不四舍五入）保证不会高报，消费方按它算永远不会超限。

什么情况下不该取整
------------------
如果消费方拿它做比例/整除运算，2.4% 的差异会被放大。所以默认仍输出
精确值，取整版作为独立文件提供给需要的一方。
"""

from __future__ import annotations


def round_k(value: int | None) -> int | None:
    """
    2^n 记法 → 同数量级的 10^n 记法（向下取整）。

        1048576 -> 1024000
         393216 ->  384000
          8192  ->    8000

    非整 K 的值原样返回，不强行对齐 —— 强行取整反而丢失信息。
    """
    if not isinstance(value, int) or value <= 0:
        return value
    if value % 1000 == 0:
        return value              # 已是十进制整 K
    if value % 1024 == 0:
        return value // 1024 * 1000
    return value                  # 非整 K（如 943718），保留精确值


def is_power_of_2_style(value: int | None) -> bool:
    """判断是否为 2 的幂记法的整 K。"""
    return bool(isinstance(value, int) and value > 0
                and value % 1024 == 0 and value % 1000 != 0)


# 输出至少要比窗口小一档，否则必然是错的（共享窗口下输出不能占满）
MIN_HEADROOM = 1024


def clamp_output(context: int | None, max_output: int | None) -> tuple:
    """
    保证 max_output < context。

    取整会把上下界拉到一起：
        1024000 / 1024000  → 取整后仍是相等
         262144 /  262000  → 取整成 256000 / 262000（反而变成 out > ctx）

    所以取整后必须重新钳一次。宁可把输出报小，也不能报一个物理上不存在的组合。
    """
    if not isinstance(context, int) or not isinstance(max_output, int):
        return context, max_output
    if max_output < context:
        return context, max_output
    # 输出占满窗口不可能：压到窗口的 3/4 并向下取整到整 K
    capped = (context * 3) // 4
    capped = capped // 1000 * 1000 or MIN_HEADROOM
    return context, min(capped, context - MIN_HEADROOM)


def apply_to_model(model: dict) -> dict:
    """
    生成取整版条目。

    保留精确值作为 `exact`，便于消费方回溯 —— 取整版不是替代品，
    是同一份数据的另一种读法。
    """
    out = dict(model)
    for key in ("context", "max_output"):
        original = model.get(key)
        rounded = round_k(original)
        if rounded != original:
            out[key] = rounded
            exact = dict(out.get("exact") or {})
            exact[key] = original
            out["exact"] = exact

    # 取整可能把 ctx/out 拉成相等甚至颠倒，必须重新钳
    ctx, outp = clamp_output(out.get("context"), out.get("max_output"))
    if outp != out.get("max_output"):
        exact = dict(out.get("exact") or {})
        exact.setdefault("max_output", model.get("max_output"))
        out["exact"] = exact
        out["max_output"] = outp

    # confidence 里记的 mimi_declared 也要跟着取整，否则两份文件不一致
    conf = dict(model.get("confidence") or {})
    for field in ("context", "max_output"):
        block = conf.get(field)
        if isinstance(block, dict) and "mimi_declared" in block:
            b = dict(block)
            b["mimi_declared"] = round_k(b["mimi_declared"])
            conf[field] = b
    out["confidence"] = conf
    return out
