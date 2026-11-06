"""The teleop IK: reaches a reachable target, never steps a joint faster than the cap, and honours joint signs."""

import numpy as np
import pytest
from conftest import FLANGE, JOINT_NAMES, arm_urdf

from dream_hackathon import ik

GOAL = np.array([0.2, -0.6, 0.9, 0.3, 0.2, -0.4])


def _solve(solver: ik.ArmIK, goal: np.ndarray, start: np.ndarray, ticks: int = 300) -> np.ndarray:
    target = solver.fk(goal)
    joints = start
    for _ in range(ticks):
        joints = solver.step(target, joints, 1 / 60)
    return joints


def test_urdf_model_reaches_a_target_and_caps_the_step() -> None:
    model = ik.urdf_model(arm_urdf(), FLANGE, (0.0, 0.0, 0.08))
    solver = ik.ArmIK(model, ik.TCP_SITE, JOINT_NAMES, max_joint_speed_rad_s=2.0, joint_limit_margin_rad=0.02)
    joints = _solve(solver, GOAL, GOAL + 0.2)
    np.testing.assert_allclose(solver.fk(joints)[:3, 3], solver.fk(GOAL)[:3, 3], atol=1e-4)
    far = GOAL + 1.0
    assert np.max(np.abs(solver.step(solver.fk(GOAL), far, 1 / 60) - far)) <= 2.0 / 60 + 1e-9


def test_joint_signs_flip_the_model_joints() -> None:
    model = ik.urdf_model(arm_urdf(), FLANGE, (0.0, 0.0, 0.08))
    plain = ik.ArmIK(model, ik.TCP_SITE, JOINT_NAMES, 2.0, 0.02)
    signs = [1, 1, 1, 1, -1, 1]
    flipped = ik.ArmIK(model, ik.TCP_SITE, JOINT_NAMES, 2.0, 0.02, joint_signs=signs)
    np.testing.assert_allclose(flipped.fk(GOAL * signs), plain.fk(GOAL), atol=1e-9)


def test_the_tcp_site_sits_on_the_flange_at_the_offset() -> None:
    near = ik.ArmIK(ik.urdf_model(arm_urdf(), FLANGE, (0.0, 0.0, 0.0)), ik.TCP_SITE, JOINT_NAMES, 2.0, 0.02)
    far = ik.ArmIK(ik.urdf_model(arm_urdf(), FLANGE, (0.0, 0.0, 0.1)), ik.TCP_SITE, JOINT_NAMES, 2.0, 0.02)
    flange = near.fk(GOAL)
    np.testing.assert_allclose(far.fk(GOAL)[:3, 3], flange[:3, 3] + flange[:3, 2] * 0.1, atol=1e-9)


def test_yam_model_from_i2rt() -> None:
    utils = pytest.importorskip("i2rt.robots.utils", reason="needs i2rt (the yam extra)")
    import mujoco

    path = utils.combine_arm_and_gripper_xml(utils.ArmType.YAM, utils.GripperType.from_string_name("linear_4310"))
    solver = ik.ArmIK(mujoco.MjModel.from_xml_path(path), "grasp_site", [f"joint{i}" for i in range(1, 7)], 2.0, 0.05)
    goal = np.array([0.1, 1.0, 1.0, 0.2, 0.1, 0.3])
    joints = _solve(solver, goal, goal + 0.15)
    np.testing.assert_allclose(solver.fk(joints)[:3, 3], solver.fk(goal)[:3, 3], atol=1e-4)
