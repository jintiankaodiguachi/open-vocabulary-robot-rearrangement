from __future__ import annotations

import time
from pathlib import Path

from std_srvs.srv import Trigger

from .io_utils import read_json
from .models import LocalObservation


class LocalPerceptionClient:
    def __init__(self, node, callback_group, service_name: str, result_json_path: str):
        self.node = node
        self.service_name = service_name
        self.result_json_path = result_json_path
        self.client = node.create_client(Trigger, service_name, callback_group=callback_group)

    @staticmethod
    def _wait_future(future, timeout_s: float):
        deadline = time.monotonic() + float(timeout_s)
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError("Local perception service timed out")
            time.sleep(0.005)
        if future.exception() is not None:
            raise future.exception()
        return future.result()

    def trigger_and_read(self, timeout_s: float) -> LocalObservation:
        if not self.client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f"Local perception service unavailable: {self.service_name}")
        p = Path(self.result_json_path)
        old_mtime_ns = p.stat().st_mtime_ns if p.is_file() else -1
        call_start_ns = time.time_ns()
        resp = self._wait_future(self.client.call_async(Trigger.Request()), timeout_s)
        if resp is None or not resp.success:
            raise RuntimeError(
                f"Local perception trigger failed: {getattr(resp, 'message', 'no response')}"
            )
        data, stat_result = read_json(self.result_json_path)
        if stat_result.st_mtime_ns == old_mtime_ns:
            raise RuntimeError("Local perception succeeded but result JSON was not updated")
        # Allow coarse filesystem timestamp granularity while still rejecting obviously old data.
        if stat_result.st_mtime_ns + 2_000_000_000 < call_start_ns:
            raise RuntimeError("Local perception result JSON is older than this trigger")
        return LocalObservation.from_dict(data)
