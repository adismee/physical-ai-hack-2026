"""Both YAM arms through i2rt's SDK (github.com/i2rt-robotics/i2rt).

i2rt runs each arm's motor loop on its own thread: this module only reads its observations and hands it joint
targets (`command_joint_pos`), in i2rt's units, which are the recordings' (joints in radians, the gripper 0 closed
to 1 open). The teleop IK runs on i2rt's own MuJoCo model of the arm and gripper.

Starting: i2rt holds each arm where it is (`zero_gravity_mode=False`), and finds a linear gripper's travel by
sweeping it with a small torque, so keep fingers clear of the jaws. A dry run opens the arms in i2rt's gravity
compensation instead: they float, and can be moved by hand.

@see YamArms
"""

import pathlib

import mujoco
import numpy as np

from dream_hackathon import config, ik
from dream_hackathon.arms import ArmsError, Grips, Joints

## The six arm joints in i2rt's model, base to wrist.
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
## Where Linux lists the network links, the CAN ones among them.
SYS_NET = pathlib.Path("/sys/class/net")
## The `flags` bit of a link that has been brought up (IFF_UP).
_IFF_UP = 0x1


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


class YamArms:
    """Both YAM arms (see Arms)."""

    def __init__(self, station: config.StationConfig, *, energize: bool, sim: bool = False) -> None:
        """Opens both arms on their CAN channels (or in i2rt's simulation).

        @param station The station config; its `yam` section names the gripper and the IK's site.
        @param energize Hold and drive the arms; False floats them in gravity compensation and only reads.
        @param sim Run i2rt's MuJoCo simulation instead of the real arms.
        """
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        assert station.yam is not None
        if not sim:
            problems = [
                problem for side in ("left", "right") if (problem := can_link_problem(station.arms[side].port))
            ]
            if problems:
                raise ArmsError("the arms cannot be opened yet:\n  " + "\n  ".join(problems))
        gripper = GripperType.from_string_name(station.yam.gripper)
        self.energized = energize
        self._robots = {}
        try:
            for side in ("left", "right"):
                self._robots[side] = get_yam_robot(
                    channel=station.arms[side].port, gripper_type=gripper, zero_gravity_mode=not energize, sim=sim
                )
        except BaseException:
            self.close()
            raise
        self.start_joints, _ = self.read()

    def read(self) -> tuple[Joints, Grips]:
        joints: Joints = {}
        grips: Grips = {}
        for side, robot in self._robots.items():
            observation = robot.get_observations()
            joints[side] = np.asarray(observation["joint_pos"], dtype=np.float64)[: len(JOINT_NAMES)]
            grips[side] = float(np.asarray(observation["gripper_pos"]).reshape(-1)[0])
        return joints, grips

    def command(self, joints: Joints, grips: Grips) -> None:
        if not self.energized:
            raise RuntimeError("a dry run never commands the arms")
        for side, robot in self._robots.items():
            robot.command_joint_pos(np.append(joints[side], np.clip(grips[side], 0.0, 1.0)))

    def close(self) -> None:
        """Stops i2rt's motor loops; the motors switch off, so the arms must be resting (at start_joints)."""
        for robot in self._robots.values():
            robot.close()
        self._robots = {}


def teleop_ik(station: config.StationConfig) -> dict[str, ik.ArmIK]:
    """Per side, the teleop IK on i2rt's MuJoCo model of the arm and its gripper, driving the config's site."""
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    assert station.yam is not None
    gripper = GripperType.from_string_name(station.yam.gripper)
    model = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(ArmType.YAM, gripper))
    teleop = station.teleop
    return {
        side: ik.ArmIK(
            model, station.yam.tcp_site, JOINT_NAMES, teleop.max_joint_speed_rad_s, teleop.joint_limit_margin_rad
        )
        for side in station.arms
    }
