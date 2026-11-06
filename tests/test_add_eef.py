"""tools/add_eef.py: Dream's tool poses added in the MakerMods layout, accepted by Dream's dataset checks."""

import importlib.util
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq
import pytest
from conftest import write_episodes

pytest.importorskip("dream_ik", reason="needs Dream's IK (the dream group: uv sync --group dream)")
import dream  # noqa: E402
from dream_ik.decode import mat_to_rot6d  # noqa: E402
from dream_ik.solver import ChunkIK  # noqa: E402
from dream_ik.specs import SERVE  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "add_eef", pathlib.Path(__file__).resolve().parents[1] / "tools" / "add_eef.py"
)
add_eef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(add_eef)


@pytest.mark.parametrize(("arm", "kinematics", "scale"), [("metal", "metal", 100.0), ("maker", "makers", 80.0)])
def test_adds_dreams_tool_poses(tmp_path: pathlib.Path, arm: str, kinematics: str, scale: float) -> None:
    source = tmp_path / "task"
    write_episodes(source, "makermods", episodes=2, frames=50, tasks=["fold the towel"])
    target = tmp_path / "task-eef"
    add_eef.convert(source, target, arm)

    embodiment = json.loads(add_eef.EMBODIMENT.read_text())
    report = dream.datasets.validate_dataset(target, embodiment)
    assert report.ok, report.error

    collected = pq.read_table(source / "data/chunk-000/episode_000000.parquet")
    converted = pq.read_table(target / "data/chunk-000/episode_000000.parquet")
    raw = np.asarray(collected.column("observation.state").to_pylist())
    state = np.asarray(converted.column("observation.state").to_pylist())
    spans = embodiment["state"]
    solver = ChunkIK(kinematics, SERVE.tool, 30.0)
    for i, side in enumerate(("left", "right")):
        joints = raw[:, i * 7 : i * 7 + 6]
        poses = np.asarray(solver.fk(joints, side))
        # The same tool pose dream.client_ik adds from the joints at run time.
        np.testing.assert_allclose(state[:, slice(*spans[f"{side}_eef_xyz"])], poses[:, :3, 3], atol=1e-5)
        np.testing.assert_allclose(
            state[:, slice(*spans[f"{side}_eef_rot6"])], mat_to_rot6d(poses[:, :3, :3]), atol=1e-5
        )
        np.testing.assert_allclose(state[:, slice(*spans[f"{side}_arm"])], joints, atol=1e-6)
        np.testing.assert_allclose(
            state[:, slice(*spans[f"{side}_grip"])], np.clip(raw[:, i * 7 + 6 : i * 7 + 7] / scale, 0, 1), atol=1e-6
        )
    # Videos are shared, not rewritten; the collected dataset is left exactly as it was.
    video = "videos/chunk-000/observation.images.scene/episode_000000.mp4"
    assert (target / video).stat().st_ino == (source / video).stat().st_ino
    assert json.loads((source / "meta" / "info.json").read_text())["features"]["action"]["shape"] == [14]
    stats = [json.loads(line) for line in (source / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    assert len(stats[0]["stats"]["action"]["mean"]) == 14


def test_replace_rebuilds_only_a_dataset(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "task"
    write_episodes(source, "makermods", episodes=1, frames=50, tasks=["fold the towel"])
    target = tmp_path / "task-eef"
    add_eef.convert(source, target, "metal")
    with pytest.raises(SystemExit, match="--replace"):
        add_eef.convert(source, target, "metal")
    write_episodes(source, "makermods", episodes=1, frames=50, tasks=["fold the towel"])
    add_eef.convert(source, target, "metal", replace=True)
    assert json.loads((target / "meta" / "info.json").read_text())["total_episodes"] == 2
    other = tmp_path / "notes"
    other.mkdir()
    with pytest.raises(SystemExit, match="not a dataset"):
        add_eef.convert(source, other, "metal", replace=True)
