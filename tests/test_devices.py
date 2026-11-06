"""The device checks, the session's wait for its devices, and a camera that reconnects. No hardware needed."""

import pathlib
import time

import numpy as np
import pytest
from conftest import filled_template

from dream_hackathon import cameras, collect, config, devices, quest

ADB_HEADER = "List of devices attached\n"


def _link(sys_net: pathlib.Path, name: str, flags: str = "0x40081", operstate: str = "up") -> None:
    link = sys_net / name
    link.mkdir(parents=True)
    (link / "flags").write_text(flags + "\n")
    (link / "operstate").write_text(operstate + "\n")


# --- the arms' links


def test_a_can_link_that_is_up_is_ready(tmp_path):
    _link(tmp_path, "can_follower_l")
    assert devices.can_link_problem("can_follower_l", tmp_path) is None


def test_a_can_link_left_down_after_a_reboot_gets_its_command(tmp_path):
    _link(tmp_path, "can_follower_l", flags="0x80", operstate="down")
    problem = devices.can_link_problem("can_follower_l", tmp_path)
    assert "sudo ip link set can_follower_l up type can bitrate 1000000" in problem


def test_a_can_link_up_without_carrier_points_at_the_arm(tmp_path):
    _link(tmp_path, "can_follower_r", operstate="down")
    assert "powered" in devices.can_link_problem("can_follower_r", tmp_path)


def test_unnamed_can_links_point_at_the_naming_step(tmp_path):
    _link(tmp_path, "can0", flags="0x80", operstate="down")
    _link(tmp_path, "can1", flags="0x80", operstate="down")
    problem = devices.can_link_problem("can_follower_l", tmp_path)
    assert "can0, can1" in problem and "CAN names" in problem


def test_a_serial_adapter_that_is_not_plugged_in(tmp_path):
    assert "not there" in devices.serial_port_problem(str(tmp_path / "ttyACM0"))
    port = tmp_path / "ttyACM1"
    port.write_text("")
    assert devices.serial_port_problem(str(port)) is None


# --- the Quest


def test_the_quest_states_adb_reports():
    assert "not connected" in devices.quest_problem(ADB_HEADER + "\n")
    assert "Allow USB debugging" in devices.quest_problem(ADB_HEADER + "1WMHH0000000AB\tunauthorized\n")
    assert "udev rule" in devices.quest_problem(ADB_HEADER + "1WMHH0000000AB\tno permissions (missing udev rules?)\n")
    assert devices.quest_problem(ADB_HEADER + "1WMHH0000000AB\tdevice\n") is None


# --- the device set, with fakes


class FakeHeadset:
    """A Quest whose data age and presses the test sets."""

    def __init__(self) -> None:
        self.age_s = 0.0
        self.press: set[str] = set()
        self.closed = False

    def data_age_s(self) -> float:
        return self.age_s

    def read(self) -> quest.QuestState:
        pressed, self.press = frozenset(self.press), set()
        poses = {"left": np.eye(4), "right": np.eye(4)}
        return quest.QuestState(poses, {"left": 0.0, "right": 0.0}, {"left": 0.0, "right": 0.0}, pressed)

    def close(self) -> None:
        self.closed = True


class FakeCamera:
    def __init__(self, cfg: config.CameraConfig) -> None:
        self.role = cfg.role
        self.lost: str | None = None

    def problem(self, after_s: float) -> str | None:
        return self.lost

    def close(self) -> None:
        pass


class Station:
    """A YAM station's device set over fakes: which cameras fail to open, what adb says, the CAN links."""

    def __init__(self, tmp_path: pathlib.Path) -> None:
        self.config = config.load(filled_template("yam", tmp_path))
        self.sys_net = tmp_path / "net"
        for name in ("can_follower_l", "can_follower_r"):
            _link(self.sys_net, name)
        self.adb = ADB_HEADER + "1WMHH0000000AB\tdevice\n"
        self.broken_cameras: set[str] = set()
        self.headset = FakeHeadset()
        self.found = devices.DeviceSet(
            self.config,
            open_quest=lambda ip: self.headset,
            open_camera=self._open_camera,
            adb_devices=lambda: self.adb,
            sys_net=self.sys_net,
        )

    def _open_camera(self, cfg: config.CameraConfig, width: int, height: int) -> FakeCamera:
        if cfg.role in self.broken_cameras:
            raise cameras.CameraError(f"camera '{cfg.role}' gave no frame")
        return FakeCamera(cfg)

    def problems(self) -> dict[str, str | None]:
        return {status.name: status.problem for status in self.found.statuses()}


def test_device_set_opens_what_is_ready_and_names_what_is_not(tmp_path):
    station = Station(tmp_path)
    station.adb = ADB_HEADER + "\n"
    station.broken_cameras = {"wrist_l"}
    station.found.try_open()
    problems = station.problems()
    assert set(problems) == {
        "Quest",
        "camera overhead",
        "camera wrist_l",
        "camera wrist_r",
        "left arm link",
        "right arm link",
    }
    assert "not connected" in problems["Quest"] and station.found.headset is None
    assert "gave no frame" in problems["camera wrist_l"] and "USB cable" in problems["camera wrist_l"]
    assert problems["camera overhead"] is None and problems["left arm link"] is None

    station.adb = ADB_HEADER + "1WMHH0000000AB\tdevice\n"
    station.broken_cameras.clear()
    station.found.try_open()
    assert all(problem is None for problem in station.problems().values())


def test_device_set_sees_problems_during_a_session(tmp_path):
    station = Station(tmp_path)
    station.found.try_open()
    station.headset.age_s = 5.0
    station.found.cameras["wrist_r"].lost = "no frame for 3 s"
    (station.sys_net / "can_follower_l" / "flags").write_text("0x80\n")
    problems = station.problems()
    assert "put the headset on" in problems["Quest"]
    assert problems["camera wrist_r"] == "no frame for 3 s"
    assert "sudo ip link set can_follower_l up" in problems["left arm link"]


# --- the wait for the devices, then A


class Clock:
    """A fake clock that a fake sleep moves on; each sleep also runs the test's script for that moment."""

    def __init__(self, script) -> None:
        self.now = 0.0
        self.script = script

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.script(self.now)
        if self.now > 60:
            raise AssertionError("the wait never ended")


def test_the_arms_open_only_once_everything_is_ready_and_a_is_pressed(tmp_path):
    station = Station(tmp_path)
    station.broken_cameras = {"wrist_l"}
    opened, logs = [], []

    def script(now: float) -> None:
        if now > 3:
            station.broken_cameras.clear()  # the camera is plugged in
        if now > 8:
            station.headset.press.add("A")

    def open_arms(cfg, *, energize, sim):
        opened.append(clock.now)
        return "the arms"

    clock = Clock(script)
    robot = collect.wait_and_open_arms(
        station.config,
        station.found,
        None,
        logs.append,
        dry_run=True,
        sim=False,
        open_arms=open_arms,
        clock=clock,
        sleep=clock.sleep,
    )
    assert robot == "the arms" and len(opened) == 1 and opened[0] > 8
    assert any(line.startswith("waiting for: camera wrist_l") for line in logs)
    assert any("press A on the right controller" in line for line in logs)
    assert any(line == "camera wrist_l: ready" for line in logs)


def test_arms_that_fail_to_open_are_reported_and_a_tries_again(tmp_path):
    station = Station(tmp_path)
    attempts, logs = [], []

    def script(now: float) -> None:
        if len(attempts) < 2:
            station.headset.press.add("A")

    def open_arms(cfg, *, energize, sim):
        attempts.append(clock.now)
        if len(attempts) == 1:
            raise RuntimeError("motor 3 does not answer")
        return "the arms"

    clock = Clock(script)
    robot = collect.wait_and_open_arms(
        station.config,
        station.found,
        None,
        logs.append,
        dry_run=False,
        sim=False,
        open_arms=open_arms,
        clock=clock,
        sleep=clock.sleep,
    )
    assert robot == "the arms" and len(attempts) == 2
    assert any("did not open: motor 3 does not answer" in line for line in logs)


def test_the_report_logs_only_what_changed():
    reported, logs = {}, []
    statuses = [devices.Status("Quest"), devices.Status("camera overhead", "no frame")]
    collect._report(statuses, reported, logs.append)
    collect._report(statuses, reported, logs.append)
    assert logs == ["Quest: ready", "camera overhead: no frame"]
    collect._report([devices.Status("Quest"), devices.Status("camera overhead")], reported, logs.append)
    assert logs[-1] == "camera overhead: ready" and len(logs) == 3


# --- a camera that comes back


class ScriptedReader:
    """A camera reader that gives frames, then fails as an unplugged one does, until the test plugs it back."""

    plugged = True
    opened = 0

    def __init__(self, cfg) -> None:
        if not ScriptedReader.plugged:
            raise cameras.CameraError("cannot open the camera")
        ScriptedReader.opened += 1

    def __call__(self):
        if not ScriptedReader.plugged:
            raise RuntimeError("Frame didn't arrive within 1000")
        time.sleep(0.005)
        return np.full((720, 1280, 3), 100, np.uint8)

    def release(self) -> None:
        pass


def test_a_camera_reconnects_by_itself_once_it_is_back(monkeypatch):
    monkeypatch.setattr(cameras, "_open_reader", ScriptedReader)
    monkeypatch.setattr(cameras, "REOPEN_EVERY_S", 0.05)
    cfg = config.CameraConfig(
        role="wrist_l", kind="uvc", device="/dev/video0", serial=None, width=1280, height=720, fps=30
    )
    camera = cameras.Camera(cfg, 320, 180)
    try:
        assert camera.latest().shape == (180, 320, 3) and camera.problem(0.5) is None
        ScriptedReader.plugged = False
        time.sleep(0.7)
        with pytest.raises(cameras.CameraError):
            camera.latest()
        assert "USB cable" in camera.problem(0.5)
        ScriptedReader.plugged = True
        time.sleep(0.3)
        assert camera.latest().shape == (180, 320, 3) and camera.problem(0.5) is None
        assert ScriptedReader.opened >= 2
    finally:
        ScriptedReader.plugged = True
        camera.close()


class FakeRealSense:
    """Enough of pyrealsense2 to see whether a camera's pipeline is started."""

    def __init__(self, connected: list[str]) -> None:
        self.connected = connected
        self.started: list[str] = []
        self.camera_info = type("camera_info", (), {"serial_number": "serial"})
        self.stream = type("stream", (), {"color": "color"})
        self.format = type("format", (), {"bgr8": "bgr8"})
        fake = self

        class Device:
            def __init__(self, serial: str) -> None:
                self._serial = serial

            def get_info(self, what):
                return self._serial

        class Context:
            def query_devices(self):
                return [Device(serial) for serial in fake.connected]

        class Config:
            def enable_device(self, serial):
                self.serial = serial

            def enable_stream(self, *args):
                pass

        class Pipeline:
            def start(self, config):
                fake.started.append(config.serial)

        self.context, self.config, self.pipeline = Context, Config, Pipeline


def test_a_realsense_that_is_not_connected_is_never_started(monkeypatch):
    # Starting one that is not connected blocks for 15 s holding the GIL, which would freeze the whole session.
    fake = FakeRealSense(connected=["111"])
    monkeypatch.setitem(__import__("sys").modules, "pyrealsense2", fake)
    cfg = config.CameraConfig(
        role="wrist_r", kind="realsense", device=None, serial="222", width=1280, height=720, fps=30
    )
    with pytest.raises(cameras.CameraError, match="RealSense 222 is not connected"):
        cameras._RealSenseReader(cfg)
    assert fake.started == []
    fake.connected.append("222")
    cameras._RealSenseReader(cfg)
    assert fake.started == ["222"]


# --- the dashboard's fix: bring a CAN link up


class FakeRun:
    """subprocess.run for `sudo -n ip link set ...`: records the command and answers as the test says."""

    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.commands: list[list[str]] = []
        self.returncode, self.stderr = returncode, stderr

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        return __import__("subprocess").CompletedProcess(command, self.returncode, "", self.stderr)


def test_a_down_can_link_offers_its_fix_and_the_fix_brings_it_up(tmp_path):
    station = Station(tmp_path)
    (station.sys_net / "can_follower_r" / "flags").write_text("0x80\n")
    station.found.try_open()
    by_name = {status.name: status for status in station.found.statuses()}
    assert by_name["right arm link"].fix == "can-up:can_follower_r"
    assert by_name["right arm link"].fix_label == "Bring can_follower_r up"
    assert by_name["left arm link"].fix is None  # up already: nothing to offer

    run = FakeRun()
    ok, message = devices.apply_fix(station.config, "can-up:can_follower_r", station.sys_net, run)
    assert ok and message == "brought can_follower_r up"
    assert run.commands[0][:2] == ["sudo", "-n"]
    assert run.commands[0][3:] == ["link", "set", "can_follower_r", "up", "type", "can", "bitrate", "1000000"]


def test_the_fix_refuses_anything_but_bringing_a_down_arm_link_up(tmp_path):
    station = Station(tmp_path)
    run = FakeRun()
    assert devices.apply_fix(station.config, "can-up:eth0", station.sys_net, run) == (
        False,
        "'eth0' is not one of this station's arm links",
    )
    ok, message = devices.apply_fix(station.config, "can-up:can_follower_l", station.sys_net, run)
    assert ok and "nothing to bring up" in message  # already up: never touched
    assert not devices.apply_fix(station.config, "can-down:can_follower_l", station.sys_net, run)[0]
    assert run.commands == []


def test_a_station_without_the_sudo_rule_gets_the_command_to_type(tmp_path):
    station = Station(tmp_path)
    (station.sys_net / "can_follower_l" / "flags").write_text("0x80\n")
    for refusal in ("sudo: a password is required", "sudo: interactive authentication is required"):  # sudo, sudo-rs
        run = FakeRun(returncode=1, stderr=refusal + "\n")
        ok, message = devices.apply_fix(station.config, "can-up:can_follower_l", station.sys_net, run)
        assert not ok and "sudo ip link set can_follower_l up type can bitrate 1000000" in message
        assert "allow_can_bringup.sh" in message
