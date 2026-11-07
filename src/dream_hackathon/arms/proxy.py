"""ProxyArms: Arms client that forwards through a running arm_proxy daemon.

Implements the same Arms protocol as MakerModsArms but talks HTTP so multiple
processes can share the same CAN connections without conflict.

Start the daemon first:
    uv run python scripts/arm_proxy.py --config configs/station.yaml [--energize]

Then point clients at it:
    DREAM_ARM_PROXY=http://localhost:8082 uv run dhk-collect ...
"""

import json
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from dream_hackathon.arms import Grips, Joints


class ProxyArms:
    """Both arms proxied through arm_proxy.py (see the Arms protocol)."""

    def __init__(self, url: str = "http://localhost:8082") -> None:
        self._url = url.rstrip("/")
        health = self._get("/api/health")
        if not health.get("ok"):
            raise RuntimeError(f"arm proxy at {url} is not ready: {health}")
        self.energized: bool = bool(health["energized"])
        state = self._get("/api/state")
        self.start_joints: Joints = {s: np.array(j, dtype=np.float64) for s, j in state["start_joints"].items()}

    def read(self) -> tuple[Joints, Grips]:
        state = self._get("/api/state")
        joints: Joints = {s: np.array(j, dtype=np.float64) for s, j in state["joints"].items()}
        grips: Grips = {s: float(g) for s, g in state["grips"].items()}
        return joints, grips

    def command(self, joints: Joints, grips: Grips) -> None:
        if not self.energized:
            raise RuntimeError("proxy is not energized — restart arm_proxy.py with --energize")
        body = {
            "joints": {s: j.tolist() for s, j in joints.items()},
            "grips": {s: float(g) for s, g in grips.items()},
        }
        self._post("/api/command", body)

    def close(self) -> None:
        """No-op: the proxy daemon owns the CAN connections and keeps running."""

    # ------------------------------------------------------------------ helpers

    def _get(self, path: str) -> Any:
        with urllib.request.urlopen(f"{self._url}{path}", timeout=5) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, body: Any) -> Any:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self._url}{path}", data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
