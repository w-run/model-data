#!/usr/bin/env python3
"""LiteLLM 价格/能力表（社区维护，但多数条目带 source 出处链接）。"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/BerriAI/litellm/main/"
       "model_prices_and_context_window.json")
UA = "w-run-model-data/4.0 (+https://github.com/w-run/model-data)"
CACHE = Path(os.environ.get("MODEL_DATA_CACHE", Path.home() / ".cache" / "model-data"))


def fetch(offline: bool = False, timeout: int = 90) -> dict:
    cached = CACHE / "litellm.json"
    if offline and cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception:
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        raise
