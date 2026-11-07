"""Diagnostic script: connect to each arm independently and print joints that move.

Usage (on the station, with torque off — arms must be supported):
    uv run python scripts/check_joints.py --config configs/station.yaml

Move a joint by hand: its arm, name and position print. Gripper errors are shown per arm
so you can tell which arm has the bad motor.
"""

import argparse
import time

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")
MOVE_THRESHOLD_DEG = 1.0  # degrees of change before we print


def read_arm(bus, side: str, prev: dict[str, float | None]) -> dict[str, float | None]:
    current: dict[str, float | None] = {}
    for motor in MOTORS:
        try:
            val = float(bus.read("Present_Position", motor))
            current[motor] = val
            prev_val = prev.get(motor)
            if prev_val is not None and abs(val - prev_val) > MOVE_THRESHOLD_DEG:
                print(f"  {side:5s}  {motor:15s}  {val:+8.2f} deg  (Δ {val - prev_val:+.2f})")
        except Exception as e:
            if prev.get(motor) != "error":
                print(f"  {side:5s}  {motor:15s}  ERROR: {e}")
            current[motor] = "error"  # type: ignore[assignment]
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Print joints that move, per arm.")
    parser.add_argument("--config", default="configs/station.yaml")
    args = parser.parse_args()

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

    arms: dict[str, object] = {}
    for side, arm_cfg in station.arms.items():
        print(f"Connecting to {side} arm on {arm_cfg.port} ...")
        try:
            device = Follower(FollowerConfig(port=arm_cfg.port, id=f"check_{side}", can_interface=mm.can_interface))
            device.bus.connect(handshake=False)
            device.bus.disable_torque()
            arms[side] = device
            print(f"  {side} arm: connected (torque off — support the arm)")
        except Exception as e:
            print(f"  {side} arm: FAILED to connect: {e}")

    if not arms:
        print("No arms connected.")
        return

    print(f"\nReading {', '.join(arms)} arm(s). Move a joint by hand to see it here. Ctrl+C to stop.\n")
    prev: dict[str, dict[str, float | None]] = {side: {} for side in arms}

    try:
        while True:
            for side, device in arms.items():
                prev[side] = read_arm(device.bus, side, prev[side])  # type: ignore[attr-defined]
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for side, device in arms.items():
            try:
                device.bus.disconnect(False)  # type: ignore[attr-defined]
                print(f"{side} arm disconnected.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
