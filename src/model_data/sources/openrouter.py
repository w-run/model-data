#!/usr/bin/env python3
"""
OpenRouter 模型目录

只作为**补充覆盖**的低权重源：它的 max_completion_tokens 在 provider
未上报时用 floor(context*0.9) 填充，污染严重（实测 85/446），
collect.py 里会把这些值标成 invalid。
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

URL = "https://openrouter.ai/api/v1/models"
UA = "w-run-model-data/4.0 (+https://github.com/w-run/model-data)"
CACHE = Path(os.environ.get("MODEL_DATA_CACHE", Path.home() / ".cache" / "model-data"))


def fetch(offline: bool = False, timeout: int = 60) -> dict:
    cached = CACHE / "openrouter.json"
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
