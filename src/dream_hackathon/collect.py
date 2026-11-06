"""Records Quest-teleoperated demonstrations into a LeRobot v2.1 dataset.

```
uv run dhk-collect --config configs/yam.yaml --task "put the red cube in the bowl" --dataset-dir ~/hackathon_data/cubes
uv run dhk-collect --config configs/yam.yaml --dry-run     # read only: shows the targets, moves nothing
uv run dhk-collect --config configs/yam.yaml --sim         # YAM: i2rt's simulated arms, blank camera frames
```

Hold a controller's grip to drive that arm, pull its trigger to close its gripper. A starts an episode and, pressed
again, saves it; B discards it; the right joystick click sends both arms back to where they started (not while
recording). Ctrl+C ends the session: an episode still recording is discarded, and the arms return to where they
started before the vendor's shutdown releases them.

The loop runs at the config's `control_hz`; every `control_hz / fps`-th tick is recorded: the measured joints and
grippers as `observation.state`, the commanded ones as `action`, and each camera's newest frame.

@see Collector, main()
"""

import argparse
import dataclasses
import pathlib
import sys
import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from dream_hackathon import arms, cameras, config, dataset, ik, layouts, quest, teleop

## Homing and the end-of-session return move every joint at most this fast.
HOME_SPEED_RAD_S = 0.5
## Homing is done when every joint is this close to its start.
HOME_TOLERANCE_RAD = 1e-3
## A live arm whose measured joints fall this far behind the command is re-synced and its clutch released.
MAX_TRACKING_ERROR_RAD = 0.35
## How often the status line is printed.
STATUS_EVERY_S = 1.0

## Frames by camera role, at the recording size.
FrameSource = Callable[[], dict[str, npt.NDArray[np.uint8]]]


@dataclasses.dataclass
class _Command:
    """What the loop last commanded: per side, six joints (radians) and the gripper (0 closed to 1 open)."""

    joints: arms.Joints
    grips: arms.Grips


class Collector:
    """The teleop-and-record loop over injected devices, one tick at a time (tick()).

    Attributes:
    - homing: Whether the arms are on their way back to where they started.
    - episode: The episode being recorded, or None.
    """

    def __init__(
        self,
        station: config.StationConfig,
        task: str,
        robot: arms.Arms,
        headset: "quest.Quest",
        frames: FrameSource,
        iks: dict[str, ik.ArmIK],
        writer: dataset.DatasetWriter,
        log: Callable[[str], None] = print,
    ) -> None:
        """Holds the devices; the command starts where the arms are.

        @param station The station config.
        @param task The task text recorded with each episode.
        @param robot Both arms.
        @param headset The Quest (anything with read() returning a quest.QuestState).
        @param frames Returns each camera's newest frame, at the recording size.
        @param iks Per side, the teleop IK.
        @param writer The dataset episodes are saved to.
        @param log Where events are reported.
        """
        self._station = station
        self._task = task
        self._robot = robot
        self._headset = headset
        self._frames = frames
        self._iks = iks
        self._writer = writer
        self._log = log
        self._dt_s = 1.0 / station.control_hz
        self._record_every = station.control_hz // station.fps
        self._video_keys = layouts.video_keys(station.family)
        self._ticks = 0
        joints, grips = robot.read()
        self._command = _Command({s: q.copy() for s, q in joints.items()}, dict(grips))
        self._clutches = {
            side: teleop.Clutch(
                quest_to_base=arm.quest_to_base,
                workspace_min_m=arm.workspace_min_m,
                workspace_max_m=arm.workspace_max_m,
                translation_scale=station.teleop.translation_scale,
            )
            for side, arm in station.arms.items()
        }
        self.homing = False
        self.episode: dataset.Episode | None = None

    def tick(self) -> None:
        """One control tick: read, react to the buttons, move the command, send it, record every n-th tick."""
        measured, measured_grips = self._robot.read()
        state = self._headset.read()
        self._handle_buttons(state.pressed)
        if self.homing:
            self._step_home()
        else:
            self._follow(state, measured)
        if self._robot.energized:
            self._resync_lagging(measured)
            self._robot.command(self._command.joints, self._command.grips)
        if self.episode is not None and self._ticks % self._record_every == 0:
            images = self._frames()
            self.episode.add_frame(
                layouts.to_vector(self._station.arm, measured, measured_grips),
                layouts.to_vector(self._station.arm, self._command.joints, self._command.grips),
                {self._video_keys[role]: frame for role, frame in images.items()},
            )
        self._ticks += 1

    def status(self) -> str:
        """One line: what is recording, and per arm whether it is engaged and where its tool target is."""
        parts = [f"REC {self.episode.frames} frames" if self.episode is not None else "idle"]
        if self.homing:
            parts.append("homing")
        for side in ("left", "right"):
            xyz = self._iks[side].fk(self._command.joints[side])[:3, 3]
            engaged = "ON " if self._clutches[side].engaged else "off"
            parts.append(f"{side} {engaged} tool [{xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f}] m")
        return " | ".join(parts)

    def go_home(self) -> None:
        """Brings the command back to where the arms started, at HOME_SPEED_RAD_S, ticking until it is there."""
        self.homing = True
        while self.homing:
            started_s = time.perf_counter()
            self._step_home()
            if self._robot.energized:
                self._robot.command(self._command.joints, self._command.grips)
            time.sleep(max(0.0, self._dt_s - (time.perf_counter() - started_s)))

    def discard(self) -> None:
        """Drops the episode being recorded, if any."""
        if self.episode is not None:
            self.episode.discard()
            self.episode = None
            self._log("episode discarded")

    def _handle_buttons(self, pressed: frozenset[str]) -> None:
        if "A" in pressed:
            self._toggle_recording()
        if "B" in pressed:
            self.discard()
        if "RJ" in pressed:
            if self.episode is not None:
                self._log("homing is off while recording: save (A) or discard (B) first")
            else:
                self.homing = True
                for clutch in self._clutches.values():
                    clutch.engaged = False

    def _toggle_recording(self) -> None:
        if self.episode is None:
            if not self._robot.energized:
                self._log("a dry run does not record")
                return
            self.episode = self._writer.start_episode(self._task)
            self._log(f"recording episode {self._writer.total_episodes}: {self._task!r}")
            return
        episode, self.episode = self.episode, None
        try:
            index = episode.save()
        except dataset.DatasetError as error:
            self._log(f"not saved: {error}")
            return
        self._log(f"saved episode {index} ({episode.frames} frames); {self._writer.total_episodes} in the dataset")

    def _follow(self, state: "quest.QuestState", measured: arms.Joints) -> None:
        """Moves each arm's command toward its clutch's target; a released clutch holds it.

        In a dry run the arms never move, so a released clutch puts the command where the real arm is: move an arm by
        hand and the status line's tool position follows it, which checks the IK model against the real arm.
        """
        for side, clutch in self._clutches.items():
            solver = self._iks[side]
            joints = self._command.joints[side]
            target = clutch.update(state.grip[side], state.poses[side], solver.fk(joints))
            if target is None:
                if not self._robot.energized:
                    self._command.joints[side] = measured[side].copy()
                continue
            self._command.joints[side] = solver.step(target, joints, self._dt_s)
            self._command.grips[side] = 1.0 - state.trigger[side]

    def _step_home(self) -> None:
        max_step = HOME_SPEED_RAD_S * self._dt_s
        done = True
        for side, start in self._robot.start_joints.items():
            joints = teleop.step_toward(self._command.joints[side], start, max_step)
            self._command.joints[side] = joints
            done = done and bool(np.all(np.abs(joints - start) <= HOME_TOLERANCE_RAD))
        if done:
            self.homing = False

    def _resync_lagging(self, measured: arms.Joints) -> None:
        """Re-syncs an arm that cannot follow (blocked, or stalled) to where it is, and releases its clutch."""
        for side, joints in measured.items():
            error = float(np.max(np.abs(self._command.joints[side] - joints)))
            if error > MAX_TRACKING_ERROR_RAD:
                self._log(f"{side} arm is {np.degrees(error):.0f} deg behind its target: holding where it is")
                self._command.joints[side] = joints.copy()
                self._clutches[side].engaged = False


def run(station: config.StationConfig, task: str, *, dry_run: bool, sim: bool, quest_ip: str | None) -> None:
    """Opens the devices, runs the loop until Ctrl+C, then shuts everything down safely."""
    iks = arms.open_ik(station)  # first: a missing tool center point stops here, before any device opens
    writer = dataset.DatasetWriter(
        station.dataset_dir,
        station.fps,
        station.arm,
        _state_names(station),
        list(layouts.video_keys(station.family).values()),
        station.record_width,
        station.record_height,
    )
    headset = quest.Quest(quest_ip)
    open_cameras: dict[str, cameras.Camera] = {}
    blank = np.zeros((station.record_height, station.record_width, 3), dtype=np.uint8)

    def blank_frames() -> dict[str, npt.NDArray[np.uint8]]:
        return {role: blank for role in layouts.CAMERA_ROLES}

    def camera_frames() -> dict[str, npt.NDArray[np.uint8]]:
        return {role: camera.latest() for role, camera in open_cameras.items()}

    try:
        if not sim:
            open_cameras = cameras.open_cameras(station.cameras, station.record_width, station.record_height)
        robot = arms.open_arms(station, energize=not dry_run, sim=sim)
        collector = None
        try:
            frames = blank_frames if sim else camera_frames
            collector = Collector(station, task, robot, headset, frames, iks, writer)
            _loop(collector, 1.0 / station.control_hz)
        finally:
            if collector is not None:
                collector.discard()
                if robot.energized:
                    print("\nreturning the arms to where they started...")
                    collector.go_home()
            robot.close()
    finally:
        for camera in open_cameras.values():
            camera.close()
        headset.close()


def _loop(collector: Collector, period_s: float) -> None:
    """Ticks at a steady rate until Ctrl+C; a late tick starts the next one at once rather than catching up."""
    print(
        "ready: hold a grip to drive an arm, A to record, B to discard, right joystick click to home, Ctrl+C to quit"
    )
    next_s = time.perf_counter()
    status_s = next_s
    try:
        while True:
            collector.tick()
            now = time.perf_counter()
            if now >= status_s:
                print(collector.status(), flush=True)
                status_s = now + STATUS_EVERY_S
            next_s = max(next_s + period_s, now)
            time.sleep(max(0.0, next_s - time.perf_counter()))
    except KeyboardInterrupt:
        pass


def _state_names(station: config.StationConfig) -> list[str]:
    """One name per value of the recorded vectors, e.g. `right_joint_0` ... `right_gripper`."""
    names = []
    for side in layouts.SIDE_ORDER[station.family]:
        names += [f"{side}_joint_{i}" for i in range(layouts.JOINTS)] + [f"{side}_gripper"]
    return names


def main(argv: list[str] | None = None) -> None:
    """The `dhk-collect` command."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="the station config, e.g. configs/yam.yaml")
    parser.add_argument("--task", help="the task text recorded with each episode (default: the config's)")
    parser.add_argument("--dataset-dir", help="the dataset folder episodes are added to (default: the config's)")
    parser.add_argument("--dry-run", action="store_true", help="read the arms and the Quest; command nothing")
    parser.add_argument("--sim", action="store_true", help="YAM: i2rt's simulated arms and blank camera frames")
    parser.add_argument("--quest-ip", help="the Quest's address for adb over Wi-Fi (default: USB)")
    args = parser.parse_args(argv)
    try:
        station = config.load(args.config)
    except config.ConfigError as error:
        sys.exit(f"config: {error}")
    if args.dataset_dir:
        station = dataclasses.replace(station, dataset_dir=pathlib.Path(args.dataset_dir).expanduser())
    run(station, args.task or station.task, dry_run=args.dry_run, sim=args.sim, quest_ip=args.quest_ip)


if __name__ == "__main__":
    main()
