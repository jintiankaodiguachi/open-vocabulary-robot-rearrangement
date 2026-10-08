from __future__ import annotations

import time
from typing import Any, Dict

from .io_utils import append_jsonl, atomic_write_json


class ExecutionJournal:
    def __init__(self, latest_path: str, history_path: str):
        self.latest_path = latest_path
        self.history_path = history_path

    def write(self, event: str, state: str, details: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            "timestamp_unix_s": time.time(),
            "event": event,
            "state": state,
            "details": details,
        }
        atomic_write_json(self.latest_path, row)
        append_jsonl(self.history_path, row)
        return row
