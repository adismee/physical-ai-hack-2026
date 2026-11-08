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

Controls (type then Enter):
    Enter   Start a new episode (idle) / save the current episode (recording)
    d       Discard the current episode
    q       Quit (saves active episode if one is running)

The left arm is opened torque-off (or gravity-compensated with --gravity) — move it freely.
The right arm is opened energized and tracks the left arm in real time.
"""

import argparse
import math
import pathlib
import queue
import sys
import threading
import time

import numpy as np

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
GRIPPER = "gripper"


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

    # ---- move follower to home
    print("Moving follower to home pose ...")
    HOME_SPEED = 0.5  # rad/s
    dt_home = 1.0 / station.control_hz
    while True:
        try:
            curr_degs = _read_follower_degs(follower_dev)
            curr_joints = np.radians([curr_degs[m] for m in MOTORS])
        except Exception:
            break
        delta = home_joints - curr_joints
        dist = float(np.max(np.abs(delta)))
        if dist < 1e-3:
            break
        step = np.clip(delta, -HOME_SPEED * dt_home, HOME_SPEED * dt_home)
        _command_follower(follower_dev, follower_cfg, curr_joints + step, home_grip, args.grip_extra)
        time.sleep(dt_home)
    print("  follower at home.")

    time.sleep(0.5)

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

    print("\nReady. Controls (type then Enter):")
    print("  Enter — start episode / save episode")
    print("  d     — discard episode")
    print("  q     — quit\n")

    try:
        while True:
            t0 = time.monotonic()

            # Read arms
            try:
                leader_degs = _read_leader_degs(leader_dev, leader_gravity)
                leader_joints, leader_grip = _degs_to_joints_grip(leader_degs, leader_cfg)
            except Exception as e:
                print(f"leader read error: {e}")
                leader_joints = np.zeros(6)
                leader_grip = 0.0

            try:
                follower_degs = _read_follower_degs(follower_dev)
                follower_joints, follower_grip = _degs_to_joints_grip(follower_degs, follower_cfg)
            except Exception as e:
                print(f"follower read error: {e}")
                follower_joints = np.zeros(6)
                follower_grip = 0.0

            # Command follower to track leader
            try:
                _command_follower(follower_dev, follower_cfg, leader_joints, leader_grip, args.grip_extra)
            except Exception as e:
                print(f"command error: {e}")

            # Record frame
            if episode is not None and tick % record_every == 0:
                images = {video_key_map[role]: cams[role].latest() for role in ("overhead", "wrist_l", "wrist_r")}
                joints_map = {args.leader: leader_joints, args.follower: leader_joints}
                grips_map = {args.leader: leader_grip, args.follower: leader_grip}
                measured_joints = {args.leader: leader_joints, args.follower: follower_joints}
                measured_grips = {args.leader: leader_grip, args.follower: follower_grip}
                state = layouts.to_vector(station.arm, measured_joints, measured_grips)
                action = layouts.to_vector(station.arm, joints_map, grips_map)
                episode.add_frame(state=state, action=action, images=images)

            tick += 1

            # Handle keyboard commands
            try:
                cmd = _cmd_q.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd is not None:
                if cmd == "" or cmd == "s":        # Enter / s = start or save
                    if episode is None:
                        episode = writer.start_episode(args.task or "mirror teleoperation")
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
                elif cmd == "q":                    # quit
                    if episode is not None:
                        idx = episode.save()
                        print(f"✓ Saved episode {idx} before quitting.")
                    break

            # Print status every ~2s
            if tick % (station.control_hz * 2) == 0:
                status = "● REC" if episode is not None else "○ idle"
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
