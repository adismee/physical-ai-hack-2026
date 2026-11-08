"""Mirror teleoperation with recording: move the left arm by hand, the right arm follows.

Writes a LeRobot v2.1 dataset in the same format as dhk-collect, compatible with
tools/add_eef.py and tools/finetune.py.

Usage (THE RIGHT ARM WILL MOVE — support it before starting):
    uv run python scripts/mirror_collect.py \\
        --config configs/station.yaml \\
        --dataset ~/hackathon_data/my-task \\
        --task "pick up the red cube"

    # Gravity compensation / weightless mode (Metal arms only, requires pinocchio):
    uv run python scripts/mirror_collect.py \\
        --config configs/station.yaml \\
        --dataset ~/hackathon_data/my-task \\
        --task "pick up the red cube" \\
        --gravity

Controls (type then Enter, or use the browser dashboard on http://localhost:8080):
    Enter   Start a new episode (idle) / save the current episode (recording)
    d       Discard the current episode
    h       Return follower to home pose
    q       Quit (saves active episode if one is running)

The left arm is opened torque-off (or gravity-compensated with --gravity) — move it freely.
The right arm is opened energized and tracks the left arm in real time.
"""

import argparse
import dataclasses
import http.server
import json
import math
import pathlib
import queue
import re
import sys
import threading
import time

import cv2
import numpy as np

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
GRIPPER = "gripper"

_CAMERA_PATH = re.compile(r"^/api/camera/([a-z_]+)\.jpg$")


# ------------------------------------------------------------------ arm helpers

def _read_leader_degs(device, gravity: bool) -> dict[str, float]:
    if gravity:
        # MetalLeader: get_action() returns the current joint positions as the teleop command.
        action = device.get_action()
        return {m: float(action[f"{m}.pos"]) for m in (*MOTORS, GRIPPER)}
    return {m: float(device.bus.read("Present_Position", m)) for m in (*MOTORS, GRIPPER)}


def _read_follower_degs(device) -> dict[str, float]:
    obs = device.get_observation()
    return {m: float(obs[f"{m}.pos"]) for m in (*MOTORS, GRIPPER)}


def _degs_to_joints_grip(degs: dict[str, float], arm_cfg) -> tuple[np.ndarray, float]:
    joints = np.radians([degs[m] for m in MOTORS])
    assert arm_cfg.gripper_open_deg is not None and arm_cfg.gripper_closed_deg is not None
    span = arm_cfg.gripper_open_deg - arm_cfg.gripper_closed_deg
    grip = float(min(1.0, max(0.0, (degs[GRIPPER] - arm_cfg.gripper_closed_deg) / span)))
    return joints, grip


def _gripper_deg(arm_cfg, grip: float) -> float:
    # Allow grip slightly below 0 so the follower can push past the calibrated closed position.
    return arm_cfg.gripper_closed_deg + min(1.0, max(-1.0, grip)) * (arm_cfg.gripper_open_deg - arm_cfg.gripper_closed_deg)


def _step_toward(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    """Moves every joint from `current` toward `target` by at most `max_step`."""
    return current + np.clip(target - current, -max_step, max_step)


def _command_follower(device, arm_cfg, joints_rad: np.ndarray, grip: float, grip_extra: float = 0.0) -> None:
    action = {f"{m}.pos": math.degrees(v) for m, v in zip(MOTORS, joints_rad)}
    # grip_extra shifts the follower gripper more closed than the leader (closes the gap at the hard stop).
    action[f"{GRIPPER}.pos"] = _gripper_deg(arm_cfg, grip - grip_extra)
    device.send_action(action)


# ------------------------------------------------------------------ input thread

_cmd_q: queue.Queue[str] = queue.Queue()


def _input_loop() -> None:
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            _cmd_q.put(line.strip().lower())
        except Exception:
            break


# ------------------------------------------------------------------ dashboard

@dataclasses.dataclass
class _DashState:
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    recording: bool = False
    homing: bool = False
    syncing: bool = False
    frames: int = 0
    total_episodes: int = 0
    task: str = ""
    leader_grip: float = 0.0
    shoulder_pan_deg: float = 0.0
    camera_roles: list = dataclasses.field(default_factory=list)


_PAGE = """\
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Mirror Collect</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: monospace; background: #111; color: #eee; margin: 0; padding: 1rem; }
  h1 { margin: 0 0 1rem; font-size: 1.1rem; color: #aaa; letter-spacing: 0.05em; }
  #cameras { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .cam-wrap { display: flex; flex-direction: column; }
  .cam-wrap img { width: 320px; height: 180px; object-fit: cover; background: #222; border: 1px solid #333; display: block; }
  .cam-label { color: #666; font-size: 0.75rem; margin-top: 0.25rem; text-align: center; }
  #status-bar { background: #1a1a1a; border: 1px solid #2a2a2a; padding: 0.6rem 1rem; border-radius: 4px; margin-bottom: 1rem; display: flex; gap: 2rem; align-items: center; flex-wrap: wrap; font-size: 0.9rem; }
  .rec  { color: #f55; font-weight: bold; }
  .homing { color: #fa0; font-weight: bold; }
  .idle { color: #666; }
  #controls { display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
  button { font-family: monospace; font-size: 0.9rem; padding: 0.45rem 1rem; border: 1px solid #444; background: #1e1e1e; color: #ccc; cursor: pointer; border-radius: 3px; }
  button:hover:not(:disabled) { background: #2a2a2a; border-color: #666; }
  button:disabled { opacity: 0.3; cursor: default; }
  #btn-primary { background: #0d2a0d; border-color: #2d6a2d; color: #8f8; }
  #btn-primary:hover:not(:disabled) { background: #133813; }
  #btn-discard { background: #2a0d0d; border-color: #6a2d2d; color: #f88; }
  #btn-discard:hover:not(:disabled) { background: #381313; }
  #task-row { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
  #task-input { font-family: monospace; font-size: 0.9rem; flex: 1; background: #1a1a1a; border: 1px solid #3a3a3a; color: #eee; padding: 0.4rem 0.6rem; border-radius: 3px; min-width: 0; }
  #task-input:focus { outline: none; border-color: #555; }
</style>
</head>
<body>
<h1>Mirror Collect Dashboard</h1>
<div id="cameras"></div>
<div id="status-bar">
  <span id="mode" class="idle">&#9675; idle</span>
  <span id="frames">0 frames</span>
  <span id="epcnt">0 episodes</span>
  <span id="pan">pan=&#8212;</span>
  <span id="grip">grip=&#8212;</span>
</div>
<div id="controls">
  <button id="btn-primary" onclick="primaryAction()">Start Episode</button>
  <button id="btn-discard" onclick="doPost('/api/episode/discard')" disabled>Discard</button>
  <button id="btn-home" onclick="doPost('/api/home')">&#8629; Go to Home</button>
</div>
<div id="task-row">
  <input id="task-input" type="text" placeholder="Task description (set before starting episode)">
  <button onclick="setTask()">Set Task</button>
</div>
<script>
const HDR = {'X-Mirror-Dashboard': '1'};
let _recording = false, _homing = false, _cameras = [];

function doPost(url, body) {
  const opts = {method: 'POST', headers: {...HDR}};
  if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  fetch(url, opts).catch(() => {});
}

function primaryAction() {
  if (_recording) doPost('/api/episode/save');
  else doPost('/api/episode/start');
}

function setTask() {
  const t = document.getElementById('task-input').value.trim();
  if (t) doPost('/api/task', {task: t});
}

function updateCameras(cameras) {
  if (JSON.stringify(cameras) === JSON.stringify(_cameras)) return;
  _cameras = cameras;
  const grid = document.getElementById('cameras');
  grid.innerHTML = '';
  cameras.forEach(role => {
    const wrap = document.createElement('div');
    wrap.className = 'cam-wrap';
    const img = document.createElement('img');
    img.id = 'cam-' + role;
    const lbl = document.createElement('div');
    lbl.className = 'cam-label';
    lbl.textContent = role;
    wrap.appendChild(img);
    wrap.appendChild(lbl);
    grid.appendChild(wrap);
  });
}

function refreshCameras() {
  _cameras.forEach(role => {
    const img = document.getElementById('cam-' + role);
    if (img) img.src = '/api/camera/' + role + '.jpg?t=' + Date.now();
  });
}

function pollStatus() {
  fetch('/api/status').then(r => r.json()).then(s => {
    _recording = s.recording;
    _homing = s.homing;
    updateCameras(s.cameras || []);

    const mode = document.getElementById('mode');
    if (s.homing)         { mode.className = 'homing'; mode.innerHTML = '&#8629; HOMING'; }
    else if (s.recording) { mode.className = 'rec';    mode.innerHTML = '&#9679; REC'; }
    else if (s.syncing)   { mode.className = 'homing'; mode.innerHTML = '&#8644; SYNCING TO LEADER'; }
    else                  { mode.className = 'idle';   mode.innerHTML = '&#9675; idle'; }

    document.getElementById('frames').textContent = s.frames + ' frames';
    document.getElementById('epcnt').textContent  = s.total_episodes + ' episodes';
    document.getElementById('pan').textContent    = 'pan=' + (s.shoulder_pan_deg != null ? s.shoulder_pan_deg.toFixed(1) + '\\u00b0' : '\\u2014');
    document.getElementById('grip').textContent   = 'grip=' + (s.leader_grip != null ? s.leader_grip.toFixed(2) : '\\u2014');

    const busy = s.recording || s.homing;
    document.getElementById('btn-primary').disabled  = s.homing || (s.syncing && !s.recording);
    document.getElementById('btn-primary').textContent = s.recording ? 'Save Episode' : 'Start Episode';
    document.getElementById('btn-discard').disabled  = !s.recording;
    document.getElementById('btn-home').disabled     = busy;

    if (s.task) {
      const inp = document.getElementById('task-input');
      if (!inp.value) inp.placeholder = s.task;
    }
  }).catch(() => {});
}

setInterval(pollStatus, 300);
setInterval(refreshCameras, 200);
pollStatus();
</script>
</body>
</html>
"""


def _make_handler(dash_state: _DashState, cmd_q: "queue.Queue[str]", cams: dict) -> type:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, "text/html; charset=utf-8", _PAGE.encode())
            elif path == "/api/status":
                with dash_state.lock:
                    snap = {
                        "recording": dash_state.recording,
                        "homing": dash_state.homing,
                        "syncing": dash_state.syncing,
                        "frames": dash_state.frames,
                        "total_episodes": dash_state.total_episodes,
                        "task": dash_state.task,
                        "leader_grip": dash_state.leader_grip,
                        "shoulder_pan_deg": dash_state.shoulder_pan_deg,
                        "cameras": list(dash_state.camera_roles),
                    }
                self._send(200, "application/json", json.dumps(snap).encode())
            elif m := _CAMERA_PATH.match(path):
                role = m.group(1)
                if role not in cams:
                    self._send(404, "text/plain", b"no such camera")
                    return
                try:
                    frame = cams[role].latest()
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        raise RuntimeError("encode failed")
                    self._send(200, "image/jpeg", buf.tobytes(), extra={"Cache-Control": "no-store"})
                except Exception:
                    self._send(503, "text/plain", b"camera unavailable")
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            if not self.headers.get("X-Mirror-Dashboard"):
                self._send(403, "text/plain", b"forbidden")
                return
            path = self.path.split("?", 1)[0]
            if path in ("/api/episode/start", "/api/episode/save"):
                cmd_q.put("")
            elif path == "/api/episode/discard":
                cmd_q.put("d")
            elif path == "/api/home":
                cmd_q.put("h")
            elif path == "/api/task":
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length))
                    with dash_state.lock:
                        dash_state.task = str(body.get("task", ""))[:512]
                except Exception:
                    self._send(400, "text/plain", b"bad request")
                    return
            else:
                self._send(404, "text/plain", b"not found")
                return
            self._send(200, "application/json", b'{"ok":true}')

        def _send(self, code: int, ctype: str, body: bytes, *, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass  # silence per-request logs

    return Handler


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror teleoperation + recording. THE RIGHT ARM WILL MOVE.")
    parser.add_argument("--config", default="configs/station.yaml")
    parser.add_argument("--dataset", required=True, help="Dataset folder path, e.g. ~/hackathon_data/my-task")
    parser.add_argument("--task", default="", help="Task description recorded with every episode")
    parser.add_argument("--leader", default="left", choices=["left", "right"], help="Which arm to move by hand")
    parser.add_argument("--follower", default="right", choices=["left", "right"], help="Which arm tracks the leader")
    parser.add_argument(
        "--gravity", action="store_true",
        help="Gravity compensation / weightless mode on the leader arm (Metal arms only; requires pinocchio).",
    )
    parser.add_argument(
        "--gravity-kd", type=float, default=0.0, metavar="KD",
        help="Velocity damping on the leader arm in gravity mode (default 0 = no resistance when moving).",
    )
    parser.add_argument(
        "--grip-extra", type=float, default=0.15, metavar="EXTRA",
        help="How much extra the follower gripper closes past the leader's calibrated zero (default 0.15).",
    )
    parser.add_argument(
        "--no-gripper-hold", action="store_true",
        help="Skip the metal gripper holding-torque controller (use if the gripper motor is unresponsive).",
    )
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port (default 8080).")
    args = parser.parse_args()

    if args.leader == args.follower:
        print("Error: leader and follower must be different arms.")
        return

    from dream_hackathon import cameras as cam_mod
    from dream_hackathon import config as cfg
    from dream_hackathon import dataset as ds
    from dream_hackathon import layouts

    station = cfg.load(args.config)
    mm = station.makermods
    if mm is None:
        print("This script is for Metal/Maker arm stations only.")
        return

    if station.arm == "metal":
        from lerobot.robots.metal_follower import MetalFollower as Follower
        from lerobot.robots.metal_follower import MetalFollowerConfig as FollowerConfig
        from makermodslab import metal_gripper
        from makermodslab.arms import METAL as FAMILY
    else:
        from lerobot.robots.maker_follower import MakerFollower as Follower
        from lerobot.robots.maker_follower import MakerFollowerConfig as FollowerConfig
        from makermodslab.arms import MAKER as FAMILY
        metal_gripper = None  # type: ignore[assignment]

    # ---- open cameras
    print("Opening cameras ...")
    cams = cam_mod.open_cameras(station.cameras, width=station.record_width, height=station.record_height)
    vk = layouts.video_keys(station.family)
    video_key_map = {"overhead": vk["overhead"], "wrist_l": vk["wrist_l"], "wrist_r": vk["wrist_r"]}

    # ---- dashboard state + server
    dash_state = _DashState(task=args.task or "mirror teleoperation")
    with dash_state.lock:
        dash_state.camera_roles = list(cams.keys())

    try:
        _server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", args.port),
            _make_handler(dash_state, _cmd_q, cams),
        )
        _server.daemon_threads = True
        threading.Thread(target=_server.serve_forever, name="dashboard", daemon=True).start()
        print(f"dashboard: http://localhost:{args.port}  (remote: ssh -L {args.port}:localhost:{args.port} <user>@<station>)")
    except OSError as e:
        print(f"Warning: could not start dashboard on port {args.port} ({e}). Continuing without it.")

    # ---- open leader arm (torque off or gravity-compensated)
    leader_cfg = station.arms[args.leader]
    leader_gravity = False
    if args.gravity:
        if station.arm != "metal":
            print("--gravity is only supported for Metal arms; falling back to torque-off.")
        else:
            try:
                from lerobot.teleoperators.metal_leader import MetalLeader, MetalLeaderConfig
                print(f"Opening {args.leader} arm (gravity compensation — weightless mode) ...")
                kd_kwargs: dict = {}
                try:
                    # Reduce velocity damping so the arm moves freely; not all versions expose kd.
                    MetalLeaderConfig(port=leader_cfg.port, id="_probe", can_interface=mm.can_interface, kd=0.0)
                    kd_kwargs = {"kd": args.gravity_kd}
                except TypeError:
                    pass
                leader_dev = MetalLeader(MetalLeaderConfig(
                    port=leader_cfg.port, id="mirror_leader", can_interface=mm.can_interface,
                    hold_kp_on_disconnect=0.0, **kd_kwargs,
                ))
                leader_dev.connect()
                leader_gravity = True
                print(f"  {args.leader}: gravity compensation active — arm is weightless")
            except Exception as e:
                print(f"--gravity failed ({e}); falling back to torque-off.")

    if not leader_gravity:
        print(f"Opening {args.leader} arm (torque off — leader) ...")
        leader_dev = Follower(FollowerConfig(
            port=leader_cfg.port, id="mirror_leader", can_interface=mm.can_interface
        ))
        leader_dev.bus.connect(handshake=False)
        leader_dev.bus.disable_torque()
        print(f"  {args.leader}: connected (torque off — move it by hand)")

    # ---- open follower arm (energized — THE ARM WILL MOVE)
    print(f"Opening {args.follower} arm (energized — follower, THE ARM WILL MOVE) ...")
    follower_cfg = station.arms[args.follower]
    follower_dev = Follower(FollowerConfig(
        port=follower_cfg.port, id=f"mirror_follower", can_interface=mm.can_interface
    ))
    if metal_gripper is not None and FAMILY.supports_gripper_effort_control and not args.no_gripper_hold:
        metal_gripper.install_metal_gripper(follower_dev, "")
    FAMILY.prepare_teleoperation(follower_dev)
    FAMILY.prepare_leader_following(follower_dev)
    follower_dev.connect(calibrate=False)
    rest_poses = list(FAMILY.capture_rest_poses(follower_dev))
    # Capture follower start joints (home pose) right after connect.
    _home_degs = _read_follower_degs(follower_dev)
    home_joints = np.radians([_home_degs[m] for m in MOTORS])
    home_grip = 1.0  # open gripper at home
    print(f"  {args.follower}: connected (energized)")

    # ---- easing (syncing) to the leader
    # The follower never jumps to the leader's pose: at startup, and again after homing, it first eases there at
    # SYNC_SPEED (the loop's "syncing" phase), and only then mirrors the leader live.
    HOME_SPEED = 0.5  # rad/s, the "Go to Home" move
    SYNC_SPEED = 0.5  # rad/s, easing to the leader's pose
    SYNC_GRIP_SPEED = 0.5  # gripper fraction per second while easing
    SYNC_TOLERANCE_RAD = math.radians(3.0)  # mirroring starts once every joint is this close
    SYNC_GRIP_TOLERANCE = 0.05
    # What the follower was last commanded: every move steps from here, never from a (possibly failed) reading.
    cmd_joints = home_joints.copy()
    cmd_grip = _degs_to_joints_grip(_home_degs, follower_cfg)[1]
    last_leader: tuple[np.ndarray, float] | None = None
    syncing = True
    print("Easing the follower to the leader's pose (slowly) before mirroring ...")

    # ---- dataset writer
    state_names = []
    for side in layouts.SIDE_ORDER[station.family]:
        state_names += [f"{side}_joint_{i}" for i in range(layouts.JOINTS)] + [f"{side}_gripper"]
    root = pathlib.Path(args.dataset).expanduser()
    writer = ds.DatasetWriter(
        root=root,
        fps=station.fps,
        robot_type=station.arm,
        state_names=state_names,
        video_keys=list(video_key_map.values()),
        frame_width=station.record_width,
        frame_height=station.record_height,
    )
    print(f"\nDataset: {root}  (episodes so far: {writer.total_episodes})")

    # ---- start input thread
    threading.Thread(target=_input_loop, daemon=True).start()

    # ---- control loop
    dt = 1.0 / station.control_hz
    record_every = station.control_hz // station.fps
    episode = None
    tick = 0

    print("\nReady. Controls (type then Enter, or use the dashboard):")
    print("  Enter — start episode / save episode")
    print("  d     — discard episode")
    print("  h     — go to home pose")
    print("  q     — quit\n")

    try:
        while True:
            t0 = time.monotonic()

            # Snapshot homing flag once per tick (avoid holding lock during arm I/O)
            with dash_state.lock:
                _homing_now = dash_state.homing

            # Read arms
            try:
                leader_degs = _read_leader_degs(leader_dev, leader_gravity)
                leader_joints, leader_grip = _degs_to_joints_grip(leader_degs, leader_cfg)
                leader_ok = True
                last_leader = (leader_joints, leader_grip)
            except Exception as e:
                # Never chase a made-up pose: the follower holds where it was commanded until the leader reads again.
                print(f"leader read error: {e} (follower holds)")
                leader_ok = False
                leader_joints, leader_grip = last_leader if last_leader is not None else (cmd_joints.copy(), cmd_grip)

            try:
                follower_degs = _read_follower_degs(follower_dev)
                follower_joints, follower_grip = _degs_to_joints_grip(follower_degs, follower_cfg)
            except Exception as e:
                print(f"follower read error: {e}")
                follower_joints = np.zeros(6)
                follower_grip = 0.0

            # Command follower: homing first, then easing to the leader, then mirroring
            send = True
            if _homing_now:
                cmd_joints = _step_toward(cmd_joints, home_joints, HOME_SPEED * dt)
                cmd_grip = float(_step_toward(np.array([cmd_grip]), np.array([home_grip]), SYNC_GRIP_SPEED * dt)[0])
                if np.array_equal(cmd_joints, home_joints):
                    with dash_state.lock:
                        dash_state.homing = False
                    syncing = True
                    print("↩ Home reached. Easing back to the leader's pose before mirroring ...")
            elif not leader_ok:
                send = False  # hold: the follower stays where it was last commanded
            elif syncing:
                cmd_joints = _step_toward(cmd_joints, leader_joints, SYNC_SPEED * dt)
                cmd_grip = float(_step_toward(np.array([cmd_grip]), np.array([leader_grip]), SYNC_GRIP_SPEED * dt)[0])
                gap = float(np.max(np.abs(leader_joints - cmd_joints)))
                if gap <= SYNC_TOLERANCE_RAD and abs(leader_grip - cmd_grip) <= SYNC_GRIP_TOLERANCE:
                    syncing = False
                    print("⇄ Follower reached the leader: mirroring live.")
            else:
                cmd_joints, cmd_grip = leader_joints.copy(), leader_grip
            if send:
                try:
                    _command_follower(follower_dev, follower_cfg, cmd_joints, cmd_grip, args.grip_extra)
                except Exception as e:
                    print(f"command error: {e}")

            # Record frame
            if episode is not None and tick % record_every == 0:
                images = {video_key_map[role]: cams[role].latest() for role in ("overhead", "wrist_l", "wrist_r")}
                joints_map = {args.leader: leader_joints, args.follower: leader_joints}
                grips_map = {args.leader: leader_grip, args.follower: leader_grip}
                measured_joints = {args.leader: leader_joints, args.follower: follower_joints}
                measured_grips = {args.leader: leader_grip, args.follower: follower_grip}
                vec_state = layouts.to_vector(station.arm, measured_joints, measured_grips)
                vec_action = layouts.to_vector(station.arm, joints_map, grips_map)
                episode.add_frame(state=vec_state, action=vec_action, images=images)

            tick += 1

            # Update dashboard state
            with dash_state.lock:
                dash_state.recording = episode is not None
                dash_state.syncing = syncing
                dash_state.frames = episode.frames if episode is not None else 0
                dash_state.total_episodes = writer.total_episodes
                dash_state.leader_grip = leader_grip
                dash_state.shoulder_pan_deg = math.degrees(leader_joints[0])

            # Handle keyboard / dashboard commands
            try:
                cmd = _cmd_q.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd is not None:
                if cmd == "" or cmd == "s":        # Enter / s = start or save
                    with dash_state.lock:
                        _is_homing = dash_state.homing
                        task_text = dash_state.task or "mirror teleoperation"
                    if _is_homing:
                        print("(cannot start while homing — wait for home to complete)")
                    elif syncing and episode is None:
                        print("(cannot start while the follower eases to the leader — wait for 'mirroring live')")
                    elif episode is None:
                        episode = writer.start_episode(task_text)
                        print(f"● Recording episode {writer.total_episodes} ...")
                    else:
                        idx = episode.save()
                        print(f"✓ Saved episode {idx}  ({writer.total_episodes} total)")
                        episode = None
                elif cmd == "d":                    # discard
                    if episode is not None:
                        episode.discard()
                        print("✗ Discarded.")
                        episode = None
                    else:
                        print("(no episode recording)")
                elif cmd == "h":                    # go to home
                    if episode is not None:
                        print("(cannot home while recording — save or discard first)")
                    else:
                        with dash_state.lock:
                            dash_state.homing = True
                        print("↩ Returning to home pose ...")
                elif cmd == "q":                    # quit
                    if episode is not None:
                        idx = episode.save()
                        print(f"✓ Saved episode {idx} before quitting.")
                    break

            # Print status every ~2s
            if tick % (station.control_hz * 2) == 0:
                if episode is not None:
                    status = "● REC"
                elif _homing_now:
                    status = "↩ HOME"
                else:
                    status = "⇄ SYNC" if syncing else "○ idle"
                pan = math.degrees(leader_joints[0])
                print(f"{status}  shoulder_pan={pan:+.1f}°  grip={leader_grip:.2f}  episodes={writer.total_episodes}")

            # Sleep to maintain control rate
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        if episode is not None:
            episode.discard()
            print("\nCtrl+C — discarded active episode.")
        print("Stopping.")
    finally:
        print("Returning follower to start pose ...")
        for ok, reason in FAMILY.return_to_rest(rest_poses):
            if not ok:
                print(f"  warning: {reason}")
        for problem in FAMILY.release_torque(follower_dev, f"{args.follower} follower"):
            print(f"  warning: {problem}")
        follower_dev.disconnect()
        if leader_gravity:
            try:
                from makermodslab.maker_rest_pose import stop_gravity_compensation
                stop_gravity_compensation(leader_dev)
            except Exception:
                pass
            leader_dev.disconnect()
        else:
            leader_dev.bus.disconnect(False)
        print("Done.")


if __name__ == "__main__":
    main()
