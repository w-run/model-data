#!/usr/bin/env python3
"""
产出校验器 —— CI 里跑，不通过就 fail

校验的是「消费方拿到这份数据会不会被坑」，不是数据好看不好看。
"""

from __future__ import annotations

import math
from typing import Any

from .limits import is_placeholder


class Violation:
    __slots__ = ("level", "model_id", "rule", "detail")

    def __init__(self, level: str, model_id: str, rule: str, detail: str):
        self.level = level
        self.model_id = model_id
        self.rule = rule
        self.detail = detail

    def __str__(self) -> str:
        return f"[{self.level}] {self.model_id} :: {self.rule} :: {self.detail}"


def check_model(model: dict[str, Any]) -> list[Violation]:
    """单模型校验。error 会阻断 CI，warn 只报告。"""
    out: list[Violation] = []
    mid = str(model.get("id") or "?")

    limit = model.get("limit") or {}
    rec = model.get("recommended") or {}
    ctx = limit.get("context")
    mx = limit.get("max_output")
    in_budget = rec.get("input_budget")
    out_budget = rec.get("output_budget")
    semantics = limit.get("semantics") or "shared"

    # ── 能力上限自洽 ──
    if mx and ctx and mx > ctx and semantics == "shared":
        out.append(Violation("error", mid, "max_output>context",
                             f"max_output={mx} > context={ctx} (shared 语义不可能)"))

    # ── 占位值不得被当成真实能力 ──
    if model.get("limit", {}).get("confidence") == "first_party" or \
       model.get("limit", {}).get("confidence") == "consensus":
        if is_placeholder(ctx, mx):
            out.append(Violation("error", mid, "placeholder-as-truth",
                                 f"confidence={limit.get('confidence')} 但 max_output={mx} 是 floor(0.9*ctx) 占位"))

    # ── 推荐预算必须可满足 ──
    if None not in (in_budget, out_budget):
        if out_budget <= 0:
            out.append(Violation("error", mid, "output_budget<=0", f"output_budget={out_budget}"))
        if in_budget <= 0:
            out.append(Violation("error", mid, "input_budget<=0", f"input_budget={in_budget}"))

        if semantics == "shared" and ctx:
            if in_budget + out_budget > ctx:
                out.append(Violation(
                    "error", mid, "budget-overflow",
                    f"input_budget({in_budget}) + output_budget({out_budget}) "
                    f"= {in_budget + out_budget} > context({ctx})"))
        elif semantics == "input_only" and ctx:
            if in_budget > ctx:
                out.append(Violation("error", mid, "input-overflow",
                                     f"input_budget({in_budget}) > context({ctx})"))

        if mx and out_budget > mx:
            out.append(Violation("error", mid, "output-over-max",
                                 f"output_budget({out_budget}) > max_output({mx})"))

    # ── 输入预算过小 = 消费方必然踩坑（warn）──
    if ctx and in_budget and in_budget < ctx * 0.5:
        out.append(Violation("warn", mid, "input-starved",
                             f"input_budget={in_budget} 仅占 context 的 {in_budget / ctx:.0%}"))

    return out


def check_all(models: list[dict[str, Any]], *, max_warn_ratio: float = 0.25) -> list[Violation]:
    violations: list[Violation] = []
    for m in models:
        violations.extend(check_model(m))

    # 全局：占位值残留比例过高说明解析层没生效
    total = len(models)
    if total:
        placeholder_hits = [
            m for m in models
            if is_placeholder((m.get("limit") or {}).get("context"),
                              (m.get("limit") or {}).get("max_output"))
            and (m.get("limit") or {}).get("confidence") != "inferred"
        ]
        ratio = len(placeholder_hits) / total
        if ratio > max_warn_ratio:
            violations.append(Violation(
                "error", "<global>", "placeholder-leak",
                f"{len(placeholder_hits)}/{total} ({ratio:.0%}) 仍带着未标记的占位输出值"))

    return violations


def report(violations: list[Violation], *, strict: bool = True) -> tuple[bool, str]:
    errors = [v for v in violations if v.level == "error"]
    warns = [v for v in violations if v.level == "warn"]

    lines = [f"校验: {len(errors)} error / {len(warns)} warn"]
    for v in errors[:40]:
        lines.append(str(v))
    if len(errors) > 40:
        lines.append(f"... 还有 {len(errors) - 40} 条 error")
    for v in warns[:10]:
        lines.append(str(v))
    if len(warns) > 10:
        lines.append(f"... 还有 {len(warns) - 10} 条 warn")

    ok = not errors if strict else True
    return ok, "\n".join(lines)
