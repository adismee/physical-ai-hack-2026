"""The clutch: the tool follows the hand's motion since engaging, in the arm's base frame, and holds when released."""

import numpy as np

from dream_hackathon import teleop

QUEST_TO_BASE = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _pose(xyz, rotation=None):
    pose = np.eye(4)
    pose[:3, 3] = xyz
    if rotation is not None:
        pose[:3, :3] = rotation
    return pose


def _clutch():
    return teleop.Clutch(QUEST_TO_BASE, np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 1.0]), translation_scale=1.0)


def test_engages_with_hysteresis_and_holds_when_released() -> None:
    clutch, tool = _clutch(), _pose([0.3, 0.0, 0.2])
    assert clutch.update(0.5, _pose([0, 0, 0]), tool) is None  # below the engage level
    assert clutch.update(0.7, _pose([0, 0, 0]), tool) is not None
    assert clutch.update(0.5, _pose([0, 0, 0]), tool) is not None  # still engaged above the release level
    assert clutch.update(0.3, _pose([0, 0, 0]), tool) is None
    assert clutch.update(0.9, None, tool) is None  # a lost controller never drives


def test_hand_forward_moves_the_tool_forward_in_the_base_frame() -> None:
    clutch, tool = _clutch(), _pose([0.3, 0.0, 0.2])
    clutch.update(1.0, _pose([0.0, 1.5, 0.0]), tool)
    # Quest -z is forward, +x is right, +y is up: forward 10 cm, right 5 cm, up 2 cm.
    target = clutch.update(1.0, _pose([0.05, 1.52, -0.10]), tool)
    np.testing.assert_allclose(target[:3, 3], [0.4, -0.05, 0.22], atol=1e-9)
    np.testing.assert_allclose(target[:3, :3], np.eye(3), atol=1e-9)


def test_hand_turn_turns_the_tool_about_the_same_base_axis() -> None:
    clutch, tool = _clutch(), _pose([0.3, 0.0, 0.2])
    clutch.update(1.0, _pose([0, 0, 0]), tool)
    angle = 0.3  # about the Quest's +y (up), which is the base's +z
    c, s = np.cos(angle), np.sin(angle)
    quest_yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    target = clutch.update(1.0, _pose([0, 0, 0], quest_yaw), tool)
    np.testing.assert_allclose(target[:3, :3], [[c, -s, 0], [s, c, 0], [0, 0, 1]], atol=1e-9)


def test_workspace_clamps_but_never_pulls_a_tool_that_started_outside() -> None:
    clutch = _clutch()
    start = _pose([0.3, 0.0, -0.2])  # below the box's floor
    clutch.update(1.0, _pose([0, 0, 0]), start)
    target = clutch.update(1.0, _pose([0, -0.1, 0]), start)  # hand down another 10 cm
    np.testing.assert_allclose(target[:3, 3], [0.3, 0.0, -0.2])
    target = clutch.update(1.0, _pose([0, 0, -5.0]), start)  # hand far forward: stops at the box
    assert target[0, 3] == 1.0


def test_step_toward_bounds_every_joint() -> None:
    np.testing.assert_allclose(teleop.step_toward(np.zeros(3), np.array([1.0, -0.01, 0.0]), 0.1), [0.1, -0.01, 0.0])
