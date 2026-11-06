"""A browser dashboard for a station with no screen, reached through SSH.

`dhk-collect` serves it on the station's loopback address only, never on the network. On your laptop, forward the
port when you connect, then open http://localhost:8080:

```
ssh -L 8080:localhost:8080 <user>@<station>
```

The page shows the three cameras as they are recorded, every device's state with the fix for one that is not ready
(the Quest, each camera, each arm's link), what the session is waiting for, whether an episode is recording, each
arm's clutch and tool position, warnings, the session's events, and the dataset's episodes with their videos to play
back. It only reads: nothing on it drives the arms or changes the dataset. `scripts/check_cameras.py` serves the same page with the
cameras alone.

@see Dashboard
"""

import collections
import contextlib
import http.server
import json
import pathlib
import re
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from dream_hackathon import cameras, dataset, devices

DEFAULT_PORT = 8080
## How many of the session's events the page lists.
_EVENTS_KEPT = 50
_CAMERA_PATH = re.compile(r"^/api/camera/([a-z_]+)\.jpg$")
_VIDEO_PATH = re.compile(r"^/api/video/(\d+)/([a-z_]+)\.mp4$")


class Dashboard:
    """Serves the page on `127.0.0.1:<port>` from a background thread.

    Attributes:
    - port: The port it listens on.
    """

    def __init__(
        self,
        port: int,
        frame: Callable[[str], npt.NDArray[np.uint8]],
        video_keys: dict[str, str],
        dataset_root: pathlib.Path | None,
        title: str,
    ) -> None:
        """Starts serving.

        @param port The port; 0 picks a free one.
        @param frame Returns one camera role's newest frame, at the recording size, or raises CameraError.
        @param video_keys Each camera role's dataset video key, to find the episodes' videos.
        @param dataset_root The dataset whose episodes the page lists, or None.
        @param title The page's heading, e.g. the station's name.
        @throws OSError The port is taken.
        """
        self._frame = frame
        self._video_keys = video_keys
        self._dataset_root = dataset_root
        self._title = title
        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] = {}
        self._devices: list[dict[str, Any]] = []
        self._phase = ""
        self._events: collections.deque[str] = collections.deque(maxlen=_EVENTS_KEPT)
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _handler_for(self))
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, name="dashboard", daemon=True).start()

    def publish(self, snapshot: dict[str, Any]) -> None:
        """Replaces the session's state the page shows; called from the control loop."""
        with self._lock:
            self._snapshot = snapshot

    def devices(self, statuses: list[devices.Status]) -> None:
        """Replaces the devices' states the page lists."""
        with self._lock:
            self._devices = [status.to_json() for status in statuses]

    def phase(self, text: str) -> None:
        """Sets the banner over the page: what the session waits for (`waiting`, `ready`, `opening`), or `` once it runs."""
        with self._lock:
            self._phase = text

    def event(self, line: str) -> None:
        """Adds one line to the page's session log."""
        with self._lock:
            self._events.appendleft(f"{time.strftime('%H:%M:%S')}  {line}")

    def how_to_open(self) -> str:
        """The line telling a participant how to reach the page from their laptop."""
        host = socket.gethostname()
        return (
            f"dashboard: on your laptop, connect with `ssh -L {self.port}:localhost:{self.port} <user>@{host}` "
            f"and open http://localhost:{self.port}"
        )

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "title": self._title,
                "cameras": list(self._video_keys),
                "snapshot": self._snapshot,
                "devices": self._devices,
                "phase": self._phase,
                "events": list(self._events),
            }

    def _episodes(self) -> list[dict[str, Any]]:
        if self._dataset_root is None:
            return []
        fps = 30
        info = self._dataset_root / "meta" / "info.json"
        if info.exists():
            fps = int(json.loads(info.read_text())["fps"])
        return [
            {"index": e.index, "frames": e.frames, "seconds": round(e.frames / fps, 1), "tasks": list(e.tasks)}
            for e in dataset.list_episodes(self._dataset_root)
        ]

    def _video(self, index: int, role: str) -> pathlib.Path | None:
        """An episode's video file, or None when there is no such episode or camera."""
        if self._dataset_root is None or role not in self._video_keys:
            return None
        if index not in {episode.index for episode in dataset.list_episodes(self._dataset_root)}:
            return None
        relative = dataset.VIDEO_PATH.format(
            episode_chunk=index // dataset.CHUNKS_SIZE, video_key=self._video_keys[role], episode_index=index
        )
        path = self._dataset_root / relative
        return path if path.is_file() else None


def _handler_for(board: Dashboard) -> type[http.server.BaseHTTPRequestHandler]:
    """The request handler class serving one dashboard."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - the name http.server calls
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif path == "/api/status":
                self._send_json(board._status())
            elif path == "/api/episodes":
                self._send_json(board._episodes())
            elif match := _CAMERA_PATH.match(path):
                self._camera(match.group(1))
            elif match := _VIDEO_PATH.match(path):
                video = board._video(int(match.group(1)), match.group(2))
                if video is None:
                    self._send(404, "text/plain", b"no such episode video")
                else:
                    self._send_file(video)
            else:
                self._send(404, "text/plain", b"not found")

        def _camera(self, role: str) -> None:
            if role not in board._video_keys:
                self._send(404, "text/plain", b"no such camera")
                return
            try:
                frame = board._frame(role)
            except cameras.CameraError as error:
                self._send(503, "text/plain", str(error).encode())
                return
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                self._send(500, "text/plain", b"could not encode the frame")
                return
            self._send(200, "image/jpeg", jpeg.tobytes())

        def _send_file(self, path: pathlib.Path) -> None:
            """Sends a video, honouring a byte range so the browser can seek in it."""
            data = path.read_bytes()
            match = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
            if match is None or not (match.group(1) or match.group(2)):
                self._send(200, "video/mp4", data, {"Accept-Ranges": "bytes"})
                return
            if match.group(1):
                start = int(match.group(1))
                end = min(int(match.group(2) or len(data) - 1), len(data) - 1)
            else:  # the last N bytes
                start, end = max(0, len(data) - int(match.group(2))), len(data) - 1
            if start > end:
                self._send(416, "text/plain", b"range not satisfiable", {"Content-Range": f"bytes */{len(data)}"})
                return
            headers = {"Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{len(data)}"}
            self._send(206, "video/mp4", data[start : end + 1], headers)

        def _send_json(self, value: Any) -> None:
            self._send(200, "application/json", json.dumps(value).encode())

        def _send(self, status: int, kind: str, body: bytes, headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            # The browser may have moved on (a newer frame, a closed tab).
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server's signature
            pass  # the terminal is the session's; requests are not worth a line each

    return Handler


## The page: no external scripts, fonts or images, so it works on a station with no internet.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Station dashboard</title>
<style>
  :root { --bg: #f6f6f4; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --line: #e2e2dd;
          --rec: #d23a2f; --ok: #2f7d4f; --warn: #b26b00; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #161615; --panel: #1f1f1d; --text: #ecece8; --muted: #9b9b95; --line: #33332f;
            --rec: #ff5a4d; --ok: #5cc28a; --warn: #f0a83a; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 16px; background: var(--bg); color: var(--text);
         font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-bottom: 12px; }
  h1 { font-size: 18px; margin: 0; flex: 1 1 auto; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 0 0 8px; }
  .badge { padding: 4px 12px; border-radius: 999px; font-weight: 600; border: 1px solid var(--line); }
  .badge.rec { background: var(--rec); color: #fff; border-color: var(--rec); }
  .badge.live { color: var(--ok); }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
  .cams { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-bottom: 12px; }
  .cam img { width: 100%; aspect-ratio: 16 / 9; object-fit: cover; background: #000; border-radius: 6px; display: block; }
  .cam .label { display: flex; justify-content: space-between; margin-top: 6px; color: var(--muted); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
  .arms { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .arm .state { font-weight: 600; }
  .arm .state.on { color: var(--ok); }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
  #warnings div { color: var(--warn); font-weight: 600; margin-top: 6px; }
  ul { list-style: none; padding: 0; margin: 0; max-height: 220px; overflow: auto; }
  li { padding: 3px 0; border-bottom: 1px solid var(--line); }
  table { width: 100%; border-collapse: collapse; }
  td, th { text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 500; }
  .tablewrap { max-height: 260px; overflow: auto; }
  button.link { background: none; border: none; color: inherit; text-decoration: underline; cursor: pointer; padding: 0 4px; }
  video { width: 100%; margin-top: 8px; border-radius: 6px; background: #000; display: none; }
  .muted { color: var(--muted); }
  #banner { display: none; margin-bottom: 12px; padding: 12px 14px; border-radius: 10px; font-weight: 600;
            border: 1px solid var(--line); background: var(--panel); }
  #banner.waiting { display: block; border-color: var(--warn); color: var(--warn); }
  #banner.ready { display: block; border-color: var(--ok); color: var(--ok); font-size: 16px; }
  #banner.opening { display: block; }
  #devices { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px; margin-bottom: 12px; }
  .device { padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); }
  .device .name { font-weight: 600; }
  .device.ok .name::before { content: "\\25CF  "; color: var(--ok); }
  .device.bad { border-color: var(--rec); }
  .device.bad .name::before { content: "\\25CF  "; color: var(--rec); }
  .device .problem { color: var(--rec); margin-top: 2px; }
</style>
</head>
<body>
<header>
  <h1 id="title">Station</h1>
  <span id="mode" class="badge">connecting</span>
  <span id="rec" class="badge">idle</span>
  <label class="muted"><input type="checkbox" id="sound"> sound cues</label>
</header>
<div id="banner"></div>
<section id="devices"></section>
<section class="cams" id="cams"></section>
<section class="grid">
  <div class="panel">
    <h2>Session</h2>
    <div id="session" class="muted">no session yet</div>
    <div class="arms" id="arms"></div>
    <div id="warnings"></div>
  </div>
  <div class="panel">
    <h2>Events</h2>
    <ul id="events"></ul>
  </div>
  <div class="panel">
    <h2>Episodes</h2>
    <div class="tablewrap"><table id="episodes"></table></div>
    <video id="player" controls></video>
  </div>
</section>
<script>
const $ = (id) => document.getElementById(id);
let cameras = [], audio = null, wasRecording = null, lastCount = null;

function beep(freq, times) {
  if (!audio || !$("sound").checked) return;
  for (let i = 0; i < times; i++) {
    const osc = audio.createOscillator(), gain = audio.createGain(), t = audio.currentTime + i * 0.18;
    osc.frequency.value = freq; gain.gain.setValueAtTime(0.2, t); gain.gain.exponentialRampToValueAtTime(0.001, t + 0.15);
    osc.connect(gain).connect(audio.destination); osc.start(t); osc.stop(t + 0.16);
  }
}
$("sound").addEventListener("change", () => { if (!audio) audio = new AudioContext(); });

function buildCameras(roles) {
  cameras = roles;
  $("cams").innerHTML = roles.map((r) =>
    `<div class="panel cam"><img id="img-${r}" alt="${r}"><div class="label"><span>${r}</span><span id="err-${r}"></span></div></div>`
  ).join("");
  roles.forEach(refreshCamera);
}
function refreshCamera(role) {
  const img = $("img-" + role), err = $("err-" + role);
  const next = () => setTimeout(() => refreshCamera(role), 120);
  const probe = new Image();
  probe.onload = () => { img.src = probe.src; err.textContent = ""; next(); };
  probe.onerror = () => { err.textContent = "no frame"; setTimeout(() => refreshCamera(role), 1000); };
  probe.src = `/api/camera/${role}.jpg?t=${Date.now()}`;
}

function fmt(v) { return (v >= 0 ? "+" : "") + v.toFixed(3); }
const BANNERS = {
  waiting: "Waiting for the devices below: fix what each red one says. The arms are not opened yet.",
  ready: "Everything is ready: press A on the right controller to open the arms.",
  opening: "Opening the arms...",
};
function renderDevices(status) {
  const phase = status.phase || "";
  $("banner").className = phase;
  $("banner").textContent = BANNERS[phase] || "";
  $("devices").innerHTML = "";
  (status.devices || []).forEach((d) => {
    const div = document.createElement("div");
    div.className = "device " + (d.ok ? "ok" : "bad");
    const name = document.createElement("div"); name.className = "name"; name.textContent = d.name + (d.ok ? ": ready" : "");
    div.appendChild(name);
    if (!d.ok) { const p = document.createElement("div"); p.className = "problem"; p.textContent = d.problem; div.appendChild(p); }
    $("devices").appendChild(div);
  });
}
function render(status) {
  $("title").textContent = status.title;
  renderDevices(status);
  if (cameras.length !== status.cameras.length) buildCameras(status.cameras);
  const s = status.snapshot;
  $("mode").textContent = s.mode || (status.phase ? "starting" : "camera check");
  $("mode").className = "badge" + (s.mode === "live" ? " live" : "");
  const recording = !!s.recording;
  $("rec").textContent = recording ? `REC  ${s.frames} frames, ${s.seconds.toFixed(1)} s` : (s.homing ? "homing" : "idle");
  $("rec").className = "badge" + (recording ? " rec" : "");
  if (wasRecording === false && recording) beep(880, 1);                     // recording started
  if (lastCount !== null && s.episodes > lastCount) beep(1320, 2);           // saved
  else if (wasRecording && !recording && s.episodes === lastCount) beep(330, 1);  // discarded
  wasRecording = recording;
  if (s.mode) lastCount = s.episodes;
  if (s.mode) {
    $("session").innerHTML = `Task: <b></b><br>Dataset: <span class="mono"></span> &middot; ${s.episodes} episodes`;
    $("session").querySelector("b").textContent = s.task;
    $("session").querySelector(".mono").textContent = s.dataset;
    $("arms").innerHTML = ["left", "right"].map((side) => {
      const a = s.arms[side];
      return `<div class="arm"><div class="state ${a.engaged ? "on" : ""}">${side}: ${a.engaged ? "driving" : "holding"}</div>
        <div class="mono">tool ${a.tool_m.map(fmt).join(" ")} m</div><div class="muted">gripper ${(a.grip * 100).toFixed(0)}% open</div></div>`;
    }).join("");
  }
  const warnings = (s.warnings || []).slice();
  if (s.loop_hz && s.control_hz && s.loop_hz < 0.9 * s.control_hz) warnings.push(`loop running slow: ${s.loop_hz.toFixed(0)} of ${s.control_hz} Hz`);
  $("warnings").innerHTML = "";
  warnings.forEach((w) => { const d = document.createElement("div"); d.textContent = w; $("warnings").appendChild(d); });
  $("events").innerHTML = "";
  status.events.forEach((e) => { const li = document.createElement("li"); li.className = "mono"; li.textContent = e; $("events").appendChild(li); });
}

function renderEpisodes(episodes) {
  const head = "<tr><th>#</th><th>length</th><th>task</th><th>play</th></tr>";
  $("episodes").innerHTML = head + episodes.slice().reverse().map((e) =>
    `<tr><td>${e.index}</td><td>${e.seconds} s</td><td class="task"></td><td>` +
    cameras.map((r) => `<button class="link" data-ep="${e.index}" data-role="${r}">${r}</button>`).join("") + "</td></tr>"
  ).join("");
  const rows = $("episodes").querySelectorAll("td.task");
  episodes.slice().reverse().forEach((e, i) => { rows[i].textContent = e.tasks.join("; "); });
  $("episodes").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    const player = $("player");
    player.src = `/api/video/${b.dataset.ep}/${b.dataset.role}.mp4`;
    player.style.display = "block"; player.play();
  }));
}

async function poll() {
  try { render(await (await fetch("/api/status")).json()); }
  catch (e) { $("mode").textContent = "disconnected"; $("mode").className = "badge"; }
  setTimeout(poll, 300);
}
async function pollEpisodes() {
  try { renderEpisodes(await (await fetch("/api/episodes")).json()); } catch (e) {}
  setTimeout(pollEpisodes, 3000);
}
poll(); pollEpisodes();
</script>
</body>
</html>
"""
