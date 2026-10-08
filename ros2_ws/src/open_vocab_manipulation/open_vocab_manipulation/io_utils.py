from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple


def read_json(path: str) -> Tuple[Dict[str, Any], os.stat_result]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"JSON file not found: {p}")
    stat_before = p.stat()
    text = p.read_text(encoding="utf-8")
    stat_after = p.stat()
    if stat_before.st_mtime_ns != stat_after.st_mtime_ns or stat_before.st_size != stat_after.st_size:
        raise RuntimeError(f"JSON file changed while being read: {p}")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {p}")
    return data, stat_after


def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def append_jsonl(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def file_age_s(stat_result: os.stat_result) -> float:
    return max(0.0, time.time() - stat_result.st_mtime)
