#!/usr/bin/env python3
"""
models.dev API（https://models.dev/api.json）

与仓库版（models_dev_repo.py）分工：
  仓库  → 容量、价格、出处（人工核对，优先）
  API   → **阶梯价 tiers**（仓库里没有，821 条）
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

URL = "https://models.dev/api.json"
UA = "w-run-model-data/4.0 (+https://github.com/w-run/model-data)"
CACHE = Path(os.environ.get("MODEL_DATA_CACHE", Path.home() / ".cache" / "model-data"))


def fetch(offline: bool = False, timeout: int = 90) -> dict[str, Any]:
    cached = CACHE / "models_dev_api.json"
    if offline and cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception:
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        raise
