"""Arm proxy: holds both CAN connections and fans them out over HTTP on port 8082.

One process owns the CAN buses; every other tool (motor_dashboard, dhk-collect,
custom scripts) connects through this proxy instead of opening the buses itself.

Usage (on the station):
    # Read-only — torque off, support the arms first:
    uv run python scripts/arm_proxy.py --config configs/station.yaml

    # Energized — full control, arms will move:
    uv run python scripts/arm_proxy.py --config configs/station.yaml --energize

Clients connect to it via:
    DREAM_ARM_PROXY=http://localhost:8082 uv run dhk-collect ...
    uv run python scripts/motor_dashboard.py --config configs/station.yaml --proxy http://localhost:8082

SSH tunnel: ssh -L 8082:localhost:8082 pah@<station>  then http://localhost:8082/api/health
"""

import argparse
import http.server
import json
import threading
import time

PORT = 8082

# Shared state: updated by poll thread, read by HTTP handler.
_state: dict = {}
_state_lock = threading.Lock()
# Serialize commands so only one client drives the arms at a time.
_cmd_lock = threading.Lock()
_arms = None  # MakerModsArms, set in main() before server starts.


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            with _state_lock:
                data = {
                    "ok": bool(_state),
                    "energized": _state.get("energized", False),
                    "sides": list(_state.get("joints", {}).keys()),
                }
            self._send(200, "application/json", json.dumps(data).encode())
        elif path == "/api/state":
            with _state_lock:
                self._send(200, "application/json", json.dumps(_state).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/api/command":
            self._send(404, "text/plain", b"not found")
            return
        with _state_lock:
            energized = _state.get("energized", False)
        if not energized:
            self._send(409, "application/json",
                       json.dumps({"ok": False, "error": "proxy not energized — restart with --energize"}).encode())
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(min(length, 65536)) or b"{}")
        try:
            import numpy as np
            joints = {side: np.array(j, dtype=np.float64) for side, j in body["joints"].items()}
            grips = {side: float(g) for side, g in body["grips"].items()}
            with _cmd_lock:
                _arms.command(joints, grips)
            self._send(200, "application/json", b'{"ok":true}')
        except Exception as e:
            self._send(500, "application/json", json.dumps({"ok": False, "error": str(e)[:300]}).encode())

    def _send(self, status: int, kind: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        pass


def _poll_loop(arms, energized: bool) -> None:
    import math

    start_joints, _ = arms.read()
    start_joints_list = {s: j.tolist() for s, j in start_joints.items()}

    while True:
        try:
            joints, grips = arms.read()
            # Also expose degrees for motor_dashboard: 6 arm joints + gripper fraction * 100.
            degrees: dict = {}
            for side, j in joints.items():
                degrees[side] = {
                    "shoulder_pan":   round(math.degrees(j[0]), 2),
                    "shoulder_lift":  round(math.degrees(j[1]), 2),
                    "elbow_flex":     round(math.degrees(j[2]), 2),
                    "wrist_flex":     round(math.degrees(j[3]), 2),
                    "wrist_yaw":      round(math.degrees(j[4]), 2),
                    "wrist_roll":     round(math.degrees(j[5]), 2),
                    "gripper":        round(grips[side] * 100, 2),  # 0–100 "percent open"
                }
            with _state_lock:
                _state.update({
                    "joints":       {s: j.tolist() for s, j in joints.items()},
                    "grips":        {s: float(g) for s, g in grips.items()},
                    "degrees":      degrees,
                    "start_joints": start_joints_list,
                    "energized":    energized,
                    "ts":           time.time(),
                })
        except Exception as e:
            print(f"poll error: {e}")
        time.sleep(0.05)  # 20 Hz


def main() -> None:
    global _arms

    parser = argparse.ArgumentParser(description="Arm proxy daemon on port 8082.")
    parser.add_argument("--config", default="configs/station.yaml")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--energize", action="store_true",
        help="Hold torque and allow commands (the arms will be driven — support them first).",
    )
    args = parser.parse_args()

    from dream_hackathon import config as cfg
    from dream_hackathon.arms import open_arms

    station = cfg.load(args.config)

    print(f"Opening arms (energize={args.energize}) ...")
    _arms = open_arms(station, energize=args.energize)
    print("Arms connected.")

    time.sleep(0.5)
    threading.Thread(target=_poll_loop, args=(_arms, args.energize), daemon=True).start()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\nProxy running. SSH tunnel: ssh -L {args.port}:localhost:{args.port} pah@<station>")
    print(f"Health check: http://localhost:{args.port}/api/health\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        _arms.close()


if __name__ == "__main__":
    main()
