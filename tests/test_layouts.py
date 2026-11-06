"""The recorded layouts and the Dream embodiments match what `dreamcli robot run` sends each arm."""

import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest
import yaml

from dream_hackathon import layouts

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("family", ["yam", "makermods"])
def test_spans_tile_each_vector(family: str) -> None:
    config = layouts.embodiment(family)
    for part in ("state", "action"):
        spans = sorted(config[part].values())
        assert spans[0][0] == 0
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert set(layouts.video_keys(family)) == set(layouts.CAMERA_ROLES)
    assert sorted(layouts.video_keys(family).values()) == sorted(config["video"].values())


def test_vectors_follow_each_familys_order_and_gripper_unit() -> None:
    joints = {"left": np.full(6, 1.0), "right": np.full(6, 2.0)}
    grips = {"left": 0.5, "right": 1.0}
    assert layouts.to_vector("yam", joints, grips).tolist() == [2.0] * 6 + [1.0] + [1.0] * 6 + [0.5]
    assert layouts.to_vector("metal", joints, grips).tolist() == [1.0] * 6 + [50.0] + [2.0] * 6 + [100.0]
    assert layouts.to_vector("maker", joints, grips).tolist() == [1.0] * 6 + [40.0] + [2.0] * 6 + [80.0]


def test_layouts_match_what_dreamcli_sends() -> None:
    adapter = pytest.importorskip("robots.adapter", reason="needs the dreamlabs library (the dream group)")
    yam, makermods = layouts.embodiment("yam"), layouts.embodiment("makermods")
    for part in ("state", "action", "video"):
        assert yam[part] == adapter.YAM_EMBODIMENT[part]
    for part in ("state", "action"):
        assert makermods[part] == adapter.MAKERMODS_EMBODIMENT[part]
    # dreamcli's MakerMods wire sends each camera under the name `dreamcli setup` gives it; a fine-tune takes only
    # the roles it was registered with, so they must be those names, with the same video keys.
    setup_session = pytest.importorskip("robots.setup_session")
    assert tuple(makermods["video"]) == setup_session.MAKERMODS_ROLES
    for role, key in makermods["video"].items():
        assert adapter.MAKERMODS_EMBODIMENT["video"][role] == key
    assert set(yam["video"]) == set(adapter.YAM_VIDEO_ROLES.values())
    assert layouts.GRIP_SCALE["metal"] == pytest.importorskip("robots.metal").GRIP_SCALE_MM
    assert layouts.GRIP_SCALE["maker"] == pytest.importorskip("robots.makers").GRIPPER_MAX_MM


@pytest.mark.parametrize("arm", ["metal", "maker"])
def test_the_templates_carry_dreams_own_tool_point(arm: str) -> None:
    # dreamcli drives a Metal or Maker arm about this point and add_eef records the training poses about it, so the
    # kit's teleoperation uses the same one; scripts/use_dream_tcp.py writes it into a station's config.
    specs = pytest.importorskip("dream_ik.specs")
    spec = {"metal": specs.METAL_ARM, "maker": specs.MAKERS_ARM}[arm]
    raw = yaml.safe_load((CONFIGS / f"{arm}.yaml").read_text())
    assert raw["makermods"]["flange_link"] == spec.flange
    expected = [round(float(value), 7) for value in spec.tools["tcp"].offset]
    assert raw["arms"]["left"]["tcp_xyz_m"] == raw["arms"]["right"]["tcp_xyz_m"] == expected


def test_use_dream_tcp_writes_the_point_into_a_station_config(tmp_path) -> None:
    pytest.importorskip("dream_ik.specs")
    station = tmp_path / "station.yaml"
    station.write_text(re.sub(r"tcp_xyz_m: .*", "tcp_xyz_m: null", (CONFIGS / "maker.yaml").read_text()))
    script = CONFIGS.parent / "scripts" / "use_dream_tcp.py"
    done = subprocess.run([sys.executable, str(script), str(station)], capture_output=True, text=True, check=True)
    assert "Dream's maker tool point" in done.stdout
    raw = yaml.safe_load(station.read_text())
    assert raw["arms"]["left"]["tcp_xyz_m"] == raw["arms"]["right"]["tcp_xyz_m"] == [0.0, 0.0, -0.1696387]
    yam = tmp_path / "yam.yaml"
    yam.write_text((CONFIGS / "yam.yaml").read_text())
    refused = subprocess.run([sys.executable, str(script), str(yam)], capture_output=True, text=True)
    assert refused.returncode != 0 and "metal or maker" in refused.stderr
