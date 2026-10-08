from pathlib import Path
import requests


class GraspServerClient:
    def __init__(self, url: str, timeout_s: float = 20.0):
        self.url = str(url)
        self.timeout_s = float(timeout_s)

    def infer_npz(self, npz_path: str) -> dict:
        p = Path(npz_path)
        if not p.is_file():
            raise FileNotFoundError(f"NPZ not found: {p}")
        with p.open("rb") as f:
            resp = requests.post(self.url,
                                 files={"file": (p.name, f, "application/octet-stream")},
                                 timeout=self.timeout_s)
        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"Non-JSON HTTP {resp.status_code}: {resp.text[:500]}") from exc
        if resp.status_code != 200 or data.get("status") != "ok":
            raise RuntimeError(f"Grasp server failed: HTTP={resp.status_code}, body={data}")
        return data
