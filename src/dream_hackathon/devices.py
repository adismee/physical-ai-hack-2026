"""What each of the station's devices is doing, and the fix when one is not ready.

`dhk-collect` starts its dashboard before it touches a device, then opens the Quest and the cameras as they become
ready (DeviceSet.try_open()), and shows every device's state on the page and in the terminal (DeviceSet.statuses()).
The checks here open nothing: the Quest is read through `adb devices`, the arms' links through `/sys/class/net` (CAN)
or `/dev` (a serial adapter), so they can run every second, during a session too.

@see Status, DeviceSet
"""

import dataclasses
import os
import pathlib
from collections.abc import Callable
from typing import Any

from dream_hackathon import cameras, config, quest

## Where Linux lists the network links, the CAN ones among them.
SYS_NET = pathlib.Path("/sys/class/net")
## The `flags` bit of a link that has been brought up (IFF_UP).
_IFF_UP = 0x1
## Controller data older than this means the Quest has stopped sending (asleep, or its link is gone).
QUEST_LOST_AFTER_S = 2.0
## A camera with no new frame for this long is shown as a problem (a shorter gap is a hiccup).
CAMERA_LOST_AFTER_S = 1.0


@dataclasses.dataclass(frozen=True)
class Status:
    """One device's state.

    Attributes:
    - name: What it is, e.g. `Quest`, `camera wrist_l`, `left arm link`.
    - problem: What is wrong and what to do about it, or None when it is ready.
    """

    name: str
    problem: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.problem is None, "problem": self.problem or ""}


def can_link_problem(port: str, sys_net: pathlib.Path = SYS_NET) -> str | None:
    """What keeps an arm's CAN link from being used, with the command that fixes it, or None when it is ready.

    The links come up down after every reboot, and are named by the setup's udev rule (docs/hardware_setup.md).
    """
    link = sys_net / port
    if not link.exists():
        others = sorted(path.name for path in sys_net.glob("can*")) if sys_net.exists() else []
        return (
            f"there is no CAN link named {port} (this computer has {', '.join(others) or 'none'}): if it lists "
            'can0 and can1, apply the setup\'s CAN names (docs/hardware_setup.md, "YAM arms"), or call the organizers'
        )
    if not int((link / "flags").read_text().strip(), 16) & _IFF_UP:
        return (
            f"CAN link {port} is down, as the links are after every reboot: "
            f"sudo ip link set {port} up type can bitrate 1000000"
        )
    if (link / "operstate").read_text().strip() not in ("up", "unknown"):
        return f"CAN link {port} is up but hears no arm: check that the arm is powered and its CAN cable is plugged in"
    return None


def serial_port_problem(port: str) -> str | None:
    """What keeps a serial CAN adapter (`/dev/serial/...`, `/dev/ttyACM*`) from being used, or None."""
    if not os.path.exists(port):
        return f"{port} is not there: check that the arm's CAN adapter is plugged in (or call the organizers)"
    if not os.access(port, os.R_OK | os.W_OK):
        return f"this account cannot open {port}: sudo usermod -aG dialout $USER, then log in again"
    return None


def arm_link_problem(station: config.StationConfig, side: str, sys_net: pathlib.Path = SYS_NET) -> str | None:
    """What keeps one arm's link (its CAN interface or its serial adapter) from being used, or None."""
    port = station.arms[side].port
    over_socketcan = station.family == "yam" or (
        station.makermods is not None and station.makermods.can_interface == "socketcan"
    )
    return can_link_problem(port, sys_net) if over_socketcan else serial_port_problem(port)


def quest_problem(adb_devices_output: str) -> str | None:
    """What keeps the Quest from being used over USB, from what `adb devices` prints, or None when it is ready."""
    problem = quest.adb_problem(adb_devices_output)
    if problem is not None:
        return problem
    if not quest.usb_devices(adb_devices_output):
        return (
            "not connected: plug the headset into the station with a USB-C data cable, wake it, and check that its "
            "Developer Mode is on"
        )
    return None


class DeviceSet:
    """The Quest and the cameras, opened as they become ready, and every device's state.

    Attributes:
    - headset: The Quest once it is open, else None.
    - cameras: Per role, the cameras open so far.
    """

    def __init__(
        self,
        station: config.StationConfig,
        *,
        sim: bool = False,
        quest_ip: str | None = None,
        open_quest: Callable[[str | None], Any] = quest.Quest,
        open_camera: Callable[[config.CameraConfig, int, int], Any] = cameras.Camera,
        adb_devices: Callable[[], str] = quest.adb_devices,
        sys_net: pathlib.Path = SYS_NET,
    ) -> None:
        """Opens nothing yet.

        @param station The station config.
        @param sim A simulated YAM: no cameras or arm links to check.
        @param quest_ip The Quest's address for adb over Wi-Fi, or None for USB.
        @param open_quest, open_camera, adb_devices, sys_net The real ones by default; tests pass fakes.
        """
        self._station = station
        self._sim = sim
        self._quest_ip = quest_ip
        self._open_quest = open_quest
        self._open_camera = open_camera
        self._adb_devices = adb_devices
        self._sys_net = sys_net
        self.headset: Any = None
        self.cameras: dict[str, Any] = {}
        self._quest_error: str | None = "not checked yet"
        self._camera_errors: dict[str, str] = {}

    def try_open(self) -> None:
        """Opens the Quest and each camera that is not open yet; one that fails keeps its error for statuses().

        It can take a few seconds (a camera waits up to two for its first frame), so call it outside the control loop.
        """
        if self.headset is None:
            problem = None if self._quest_ip else quest_problem(self._adb_devices())
            if problem is None:
                try:
                    self.headset = self._open_quest(self._quest_ip)
                except SystemExit as error:  # oculus_reader exits when it finds no headset
                    problem = f"could not start the Quest's app (exit {error.code}): replug the headset and wake it"
                except Exception as error:  # noqa: BLE001 - shown on the page, and tried again
                    problem = str(error) or type(error).__name__
            self._quest_error = problem
        if self._sim:
            return
        for cfg in self._station.cameras:
            if cfg.role in self.cameras:
                continue
            try:
                self.cameras[cfg.role] = self._open_camera(
                    cfg, self._station.record_width, self._station.record_height
                )
            except cameras.CameraError as error:
                self._camera_errors[cfg.role] = f"{error}: check its USB cable"
            else:
                self._camera_errors.pop(cfg.role, None)

    def statuses(self) -> list[Status]:
        """Every device's state now; opens nothing, so it can run every second, during a session too."""
        found = [Status("Quest", self._quest_status())]
        if self._sim:
            return found
        for cfg in self._station.cameras:
            camera = self.cameras.get(cfg.role)
            problem = camera.problem(CAMERA_LOST_AFTER_S) if camera is not None else self._camera_errors.get(cfg.role)
            found.append(Status(f"camera {cfg.role}", problem or (None if camera is not None else "not open yet")))
        for side in ("left", "right"):
            found.append(Status(f"{side} arm link", arm_link_problem(self._station, side, self._sys_net)))
        return found

    def close(self) -> None:
        for camera in self.cameras.values():
            camera.close()
        if self.headset is not None:
            self.headset.close()

    def _quest_status(self) -> str | None:
        if self.headset is None:
            return self._quest_error
        if not self._quest_ip:
            problem = quest_problem(self._adb_devices())
            if problem is not None:
                return f"{problem}; then restart dhk-collect (Ctrl+C, and run it again)"
        age_s = self.headset.data_age_s()
        if age_s > QUEST_LOST_AFTER_S:
            waited = "no controller data yet" if age_s == float("inf") else f"no controller data for {age_s:.0f} s"
            return f"{waited}: put the headset on (it sleeps when taken off) and hold the controllers in view"
        return None
