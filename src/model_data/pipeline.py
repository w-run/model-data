#!/usr/bin/env python3
"""
多阶段构建流水线

为什么分阶段
------------
原来是一条大流水线：改任何一处都要全量重跑（几分钟），而且中间结果不可见，
出问题只能靠加 print 猜。

现在每一阶段把结果落盘到 `.build/`：

    01-collect    采集各源 → 原始观测
    02-identity   身份对齐 → 族分组
    03-assemble   装配 → 模型条目
    04-finalize   取整/过滤 → 最终产出

好处
----
  - 改「过滤规则」只需重跑 04，秒级
  - 改「族分组」重跑 02/03，不必重新采集
  - 每阶段产物可单独审阅、单独写测试
  - 阶段失败不会污染下游

用法
----
    python -m model_data.pipeline --from 03        # 从 03 开始跑到底
    python -m model_data.pipeline --only 04        # 只跑 04
    python -m model_data.pipeline --dry 03         # 只验证 03，不落盘

中间产物格式统一为 JSON Lines，便于 grep / jq / 分段检查。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# 阶段定义：id → (名称, 说明)
STAGES: dict[str, tuple[str, str]] = {
    "01": ("collect", "采集各源，产出原始观测"),
    "02": ("identity", "身份对齐：canonical / 族分组"),
    "03": ("assemble", "按族装配模型条目"),
    "04": ("finalize", "取整、过滤、校验、写出"),
}

CACHE_DIRNAME = ".build"


@dataclass
class Context:
    """阶段间传递的上下文。"""

    build_dir: Path
    output_dir: Path
    offline: bool = False
    only_sources: str = ""
    dry: bool = False
    data: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = {}

    # ── 中间产物读写 ─────────────────────────────────────────────
    def path(self, stage: str) -> Path:
        return self.build_dir / f"{stage}-{STAGES[stage][0]}.json"

    def save(self, stage: str, payload: Any) -> None:
        p = self.path(stage)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        size = p.stat().st_size
        log(f"    ↳ 已保存 {p.name} ({size / 1024:.0f} KB)")

    def load(self, stage: str) -> Any:
        p = self.path(stage)
        if not p.exists():
            raise FileNotFoundError(f"缺少前置产物 {p.name}，请先跑阶段 {stage}")
        return json.loads(p.read_text(encoding="utf-8"))


def log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class Stage:
    id: str
    name: str
    run: Callable[[Context], None]
    # 该阶段依赖的前置阶段
    needs: tuple[str, ...] = ()


def resolve_stages(stage_only: str, start: str) -> list[str]:
    """
    决定要跑哪些阶段。

    stage_only 是 `--only 03,04` 传入的阶段列表；
    start 是 `--from 03` 传入的起始阶段。二者互斥，stage_only 优先。
    """
    order = sorted(STAGES)
    if stage_only:
        pick = {s.strip() for s in stage_only.split(",") if s.strip()}
        unknown = pick - set(order)
        if unknown:
            raise ValueError(f"未知阶段: {sorted(unknown)}（可选 {order}）")
        return [s for s in order if s in pick]
    if start:
        if start not in STAGES:
            raise ValueError(f"未知起始阶段: {start}")
        return [s for s in order if s >= start]
    return order


def run_pipeline(stages: list[Stage], ctx: Context, *, start: str = "",
                 stage_only: str = "") -> int:
    """按序执行阶段，失败即停。结尾打印各阶段耗时，便于定位瓶颈。"""
    selected = set(resolve_stages(stage_only, start))
    timings: dict[str, float] = {}
    total = len(stages)
    for i, st in enumerate(stages, 1):
        if st.id not in selected:
            continue
        t0 = time.time()
        log("")
        log(f"  [{i}/{total}] 阶段 {st.id} · {STAGES[st.id][0]}")
        log(f"        {STAGES[st.id][1]}")
        # 依赖检查：前置阶段被跳过时，必须有它的产物
        missing = [n for n in st.needs if n not in selected and not ctx.path(n).exists()]
        if missing:
            log(f"        ✗ 缺少前置产物: {missing}（请先跑阶段 {'/'.join(missing)}）")
            return 2
        try:
            st.run(ctx)
        except Exception as exc:  # noqa: BLE001
            log(f"        ✗ 失败: {exc.__class__.__name__}: {exc}")
            raise
        elapsed = time.time() - t0
        timings[st.id] = elapsed
        log(f"        ✓ 完成（{elapsed:.1f}s）")

    if len(timings) > 1:
        log("")
        log("  阶段耗时:")
        for sid, sec in sorted(timings.items()):
            log(f"    {sid} {STAGES[sid][0]:10s} {sec:6.1f}s")
    return 0
