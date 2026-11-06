"""The recorded layouts and the Dream embodiments match what `dreamcli robot run` sends each arm."""

import numpy as np
import pytest

from dream_hackathon import layouts


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
