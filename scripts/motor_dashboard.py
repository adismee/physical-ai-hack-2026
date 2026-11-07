"""Live motor status dashboard for both arms, served on port 8081.

Usage (on the station, torque off — support the arms):
    # Direct CAN connection (exclusive — conflicts with other CAN clients):
    uv run python scripts/motor_dashboard.py --config configs/station.yaml

    # Via arm proxy (non-exclusive — works alongside dhk-collect):
    uv run python scripts/motor_dashboard.py --proxy http://localhost:8082

On your laptop: ssh -L 8081:localhost:8081 pah@<station> then open http://localhost:8081
"""

import argparse
import http.server
import json
import pathlib
import subprocess
import threading
import time
import urllib.request

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")
PORT = 8081
LOG_PATH = pathlib.Path("~/motor_dashboard.log").expanduser()

# Shared state updated by the polling thread, read by the HTTP handler.
_state: dict = {"arms": {}}
_state_lock = threading.Lock()
_log_lock = threading.Lock()


def log(line: str) -> None:
    ts = time.strftime("%H:%M:%S")
    entry = f"{ts}  {line}\n"
    with _log_lock:
        with LOG_PATH.open("a") as f:
            f.write(entry)


def poll_arms(devices: dict) -> None:
    """Poll CAN buses directly (direct mode — exclusive CAN access)."""
    # Track last known status per (side, motor) to log only on changes.
    last: dict[tuple[str, str], bool] = {}
    while True:
        arms = {}
        for side, bus in devices.items():
            motors = {}
            for motor in MOTORS:
                try:
                    val = float(bus.read("Present_Position", motor))
                    motors[motor] = {"ok": True, "deg": round(val, 2)}
                    if last.get((side, motor)) is False:
                        log(f"RECOVERED  {side:5s}  {motor}  now ok at {val:.2f} deg")
                    last[(side, motor)] = True
                except Exception as e:
                    err = str(e).split("\n")[0][:120]
                    motors[motor] = {"ok": False, "error": err}
                    if last.get((side, motor)) is not False:
                        log(f"ERROR      {side:5s}  {motor}  {err}")
                    last[(side, motor)] = False
            arms[side] = motors
        with _state_lock:
            _state["arms"] = arms
        time.sleep(0.1)


def poll_proxy(proxy_url: str) -> None:
    """Poll an arm_proxy daemon (proxy mode — non-exclusive, works alongside dhk-collect)."""
    last: dict[tuple[str, str], bool] = {}
    url = proxy_url.rstrip("/") + "/api/state"
    while True:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read())
            degrees = data.get("degrees", {})
            arms = {}
            for side, motor_degs in degrees.items():
                motors = {}
                for motor in MOTORS:
                    if motor in motor_degs:
                        val = motor_degs[motor]
                        motors[motor] = {"ok": True, "deg": round(val, 2)}
                        if last.get((side, motor)) is False:
                            log(f"RECOVERED  {side:5s}  {motor}  now ok at {val:.2f}")
                        last[(side, motor)] = True
                    else:
                        motors[motor] = {"ok": False, "error": "not in proxy state"}
                        if last.get((side, motor)) is not False:
                            log(f"ERROR      {side:5s}  {motor}  not in proxy state")
                        last[(side, motor)] = False
                arms[side] = motors
            with _state_lock:
                _state["arms"] = arms
        except Exception as e:
            err = str(e).split("\n")[0][:120]
            log(f"PROXY ERROR  {err}")
        time.sleep(0.1)


# Commands allowed via /api/shell — no freeform execution.
_ALLOWED_COMMANDS = {
    "ps":        ["ps", "aux"],
    "lsof-arms": ["lsof", "/dev/ttyACM0", "/dev/ttyACM1"],
    "lsof-serial": ["lsof", "-c", "python"],
    "log":       ["cat", str(LOG_PATH)],
    "dmesg-usb": ["dmesg", "--time-format=reltime", "--facility=kern", "--level=err,warn"],
    "ls-dev":    ["ls", "-la", "/dev/ttyACM0", "/dev/ttyACM1"],
    "ls-serial": ["ls", "-la", "/dev/serial/by-id/"],
}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif path == "/api/status":
            with _state_lock:
                data = json.dumps(_state).encode()
            self._send(200, "application/json", data)
        elif path == "/api/log":
            text = LOG_PATH.read_text() if LOG_PATH.exists() else "(no log yet)"
            self._send(200, "text/plain; charset=utf-8", text.encode())
        elif path == "/api/commands":
            self._send(200, "application/json", json.dumps(list(_ALLOWED_COMMANDS)).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/api/shell":
            self._send(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(min(length, 4096)) or b"{}")
        cmd_key = str(body.get("cmd", ""))
        command = _ALLOWED_COMMANDS.get(cmd_key)
        if command is None:
            self._send(400, "application/json", json.dumps(
                {"ok": False, "output": f"unknown command {cmd_key!r}. allowed: {list(_ALLOWED_COMMANDS)}"}
            ).encode())
            return
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            output = result.stdout + (result.stderr if result.returncode != 0 else "")
        except Exception as e:
            output = str(e)
        self._send(200, "application/json", json.dumps({"ok": True, "output": output}).encode())

    def _send(self, status: int, kind: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Motor status dashboard on port 8081.")
    parser.add_argument("--config", default="configs/station.yaml")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--proxy", metavar="URL",
        help="Read from an arm_proxy daemon (e.g. http://localhost:8082) instead of opening CAN directly. "
             "Use this when dhk-collect is running so both can share the arms.",
    )
    args = parser.parse_args()

    if args.proxy:
        # Proxy mode: no CAN connection, reads from arm_proxy daemon.
        log(f"starting — proxy {args.proxy}")
        print(f"Proxy mode: reading from {args.proxy}")
        threading.Thread(target=poll_proxy, args=(args.proxy,), daemon=True).start()
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        print(f"\nDashboard: on your laptop, ssh -L {args.port}:localhost:{args.port} pah@<station>")
        print(f"Then open http://localhost:{args.port}\n")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    # Direct mode: open CAN buses exclusively.
    from dream_hackathon import config as cfg

    station = cfg.load(args.config)
    mm = station.makermods
    if mm is None:
        print("This script is for Metal/Maker arm stations only.")
        return

    if station.arm == "metal":
        from lerobot.robots.metal_follower import MetalFollower as Follower
        from lerobot.robots.metal_follower import MetalFollowerConfig as FollowerConfig
    else:
        from lerobot.robots.maker_follower import MakerFollower as Follower
        from lerobot.robots.maker_follower import MakerFollowerConfig as FollowerConfig

    log(f"starting — config {args.config}")
    buses: dict[str, object] = {}
    for side, arm_cfg in station.arms.items():
        print(f"Connecting to {side} arm on {arm_cfg.port} ...")
        try:
            device = Follower(FollowerConfig(port=arm_cfg.port, id=f"motordash_{side}", can_interface=mm.can_interface))
            device.bus.connect(handshake=False)
            device.bus.disable_torque()
            buses[side] = device.bus
            print(f"  {side}: connected (torque off)")
            log(f"connected  {side}  {arm_cfg.port}")
        except Exception as e:
            print(f"  {side}: FAILED — {e}")
            log(f"FAILED     {side}  {arm_cfg.port}  {e}")

    if not buses:
        print("No arms connected.")
        return

    time.sleep(1.0)  # let buses finish initializing before polling
    threading.Thread(target=poll_arms, args=(buses,), daemon=True).start()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\nDashboard: on your laptop, ssh -L {args.port}:localhost:{args.port} pah@<station>")
    print(f"Then open http://localhost:{args.port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for side, bus in buses.items():
            try:
                bus.disconnect(False)  # type: ignore[attr-defined]
            except Exception:
                pass


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Motor status</title>
<style>
  :root { --bg: #161615; --panel: #1f1f1d; --text: #ecece8; --muted: #9b9b95;
          --line: #33332f; --ok: #5cc28a; --err: #ff5a4d; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 20px; background: var(--bg); color: var(--text);
         font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  h1 { font-size: 16px; margin: 0 0 20px; color: var(--muted); }
  .arms { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .arm { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
  .arm h2 { margin: 0 0 12px; font-size: 15px; text-transform: uppercase; letter-spacing: .05em; }
  table { width: 100%; border-collapse: collapse; table-layout: fixed; }
  td, th { padding: 5px 8px; border-bottom: 1px solid var(--line); text-align: left; overflow: hidden; white-space: nowrap; }
  th { color: var(--muted); font-weight: 500; font-size: 12px; }
  .col-motor { width: 55%; }
  .col-pos   { width: 25%; }
  .col-state { width: 20%; }
  .ok  { color: var(--ok); font-weight: 600; }
  .err { color: var(--err); font-weight: 600; }
  .muted { color: var(--muted); }
  #log { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
  #log h2 { margin: 0 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
  #loglines { max-height: 200px; overflow-y: auto; font-size: 12px; }
  #loglines div { padding: 2px 0; border-bottom: 1px solid var(--line); }
  #loglines .err { color: var(--err); font-weight: normal; }
  #loglines .rec { color: var(--ok); font-weight: normal; }
  #updated { color: var(--muted); font-size: 12px; margin-top: 12px; }
</style>
</head>
<body>
<h1>Motor status — both arms</h1>
<div class="arms" id="arms"></div>
<div id="log"><h2>Log</h2><div id="loglines"></div></div>
<div id="updated"></div>
<script>
const MOTORS = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_yaw","wrist_roll","gripper"];
const MAX_LOG = 100;
let logLines = [];
let prevState = {};

async function poll() {
  try {
    const data = await (await fetch("/api/status")).json();

    // Update arm tables (fixed layout, no reflow)
    for (const [side, motors] of Object.entries(data.arms || {})) {
      let tbody = document.getElementById("tbody-" + side);
      if (!tbody) {
        const container = document.getElementById("arms");
        const div = document.createElement("div");
        div.className = "arm";
        div.innerHTML = `<h2>${side}</h2><table><thead><tr>
          <th class="col-motor">motor</th>
          <th class="col-pos">position</th>
          <th class="col-state">status</th>
        </tr></thead><tbody id="tbody-${side}"></tbody></table>`;
        container.appendChild(div);
        tbody = document.getElementById("tbody-" + side);
        // pre-create rows
        MOTORS.forEach(m => {
          const tr = document.createElement("tr");
          tr.id = `row-${side}-${m}`;
          tr.innerHTML = `<td class="muted">${m}</td><td id="pos-${side}-${m}">—</td><td id="state-${side}-${m}">—</td>`;
          tbody.appendChild(tr);
        });
      }
      MOTORS.forEach(m => {
        const info = motors[m];
        if (!info) return;
        const posEl = document.getElementById(`pos-${side}-${m}`);
        const stateEl = document.getElementById(`state-${side}-${m}`);
        const key = `${side}-${m}`;
        if (info.ok) {
          posEl.textContent = info.deg.toFixed(2) + "°";
          posEl.className = "";
          stateEl.textContent = "ok";
          stateEl.className = "ok";
          if (prevState[key] === false) addLog("rec", `RECOVERED  ${side}  ${m}`);
          prevState[key] = true;
        } else {
          posEl.textContent = "—";
          posEl.className = "err";
          stateEl.textContent = "error";
          stateEl.className = "err";
          if (prevState[key] !== false) addLog("err", `ERROR  ${side}  ${m}  ${info.error}`);
          prevState[key] = false;
        }
      });
    }

    document.getElementById("updated").textContent = "updated " + new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById("updated").textContent = "disconnected";
  }
  setTimeout(poll, 200);
}

function addLog(cls, text) {
  const ts = new Date().toLocaleTimeString();
  logLines.unshift({cls, text: `${ts}  ${text}`});
  if (logLines.length > MAX_LOG) logLines.pop();
  const container = document.getElementById("loglines");
  container.innerHTML = "";
  logLines.forEach(l => {
    const d = document.createElement("div");
    d.className = l.cls;
    d.textContent = l.text;
    container.appendChild(d);
  });
}

poll();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
