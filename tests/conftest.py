"""Shared test helpers: a small six-joint arm URDF and synthetic recordings."""

import pathlib

import numpy as np
import pytest

from dream_hackathon import dataset, layouts

## A six-joint arm with a flange link, enough for the IK and collector tests (no meshes).
ARM_URDF = """<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link"><inertial><mass value="1"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  {links}
</robot>
"""
_JOINT = """
  <link name="link{i}"><inertial><mass value="0.5"/>
    <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><mesh filename="package://does_not_exist/link{i}.stl"/></geometry></visual></link>
  <joint name="j{i}" type="revolute">
    <parent link="{parent}"/><child link="link{i}"/>
    <origin xyz="{xyz}" rpy="0 0 0"/><axis xyz="{axis}"/>
    <limit lower="-2.8" upper="2.8" effort="10" velocity="5"/>
  </joint>"""
_CHAIN = [
    ("0 0 0.1", "0 0 1"),
    ("0 0 0.05", "0 1 0"),
    ("0.25 0 0", "0 1 0"),
    ("0.2 0 0", "0 1 0"),
    ("0.05 0 0", "0 0 1"),
    ("0.05 0 0", "1 0 0"),
]
JOINT_NAMES = [f"j{i}" for i in range(1, 7)]
FLANGE = "link6"


def arm_urdf() -> str:
    links = "".join(
        _JOINT.format(i=i, parent="base_link" if i == 1 else f"link{i - 1}", xyz=xyz, axis=axis)
        for i, (xyz, axis) in enumerate(_CHAIN, start=1)
    )
    return ARM_URDF.format(links=links)


def write_episodes(root: pathlib.Path, family: str, episodes: int, frames: int, tasks: list[str]) -> None:
    """Writes synthetic episodes in a family's collected layout (14-wide, its cameras)."""
    keys = list(layouts.video_keys(family).values())
    writer = dataset.DatasetWriter(root, 30, family, [f"v{i}" for i in range(layouts.WIDTH)], keys, 320, 180)
    rng = np.random.default_rng(0)
    for e in range(episodes):
        episode = writer.start_episode(tasks[e % len(tasks)])
        for t in range(frames):
            image = np.full((180, 320, 3), (t * 7) % 255, np.uint8)
            vector = (0.1 * rng.standard_normal(layouts.WIDTH)).astype(np.float32)
            vector[6] = vector[13] = 40.0 + t * 0.1  # a plausible gripper value in every layout's unit
            episode.add_frame(vector, vector + 0.01, {key: image for key in keys})
        episode.save()


@pytest.fixture
def urdf_path(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "arm.urdf"
    path.write_text(arm_urdf())
    return path
