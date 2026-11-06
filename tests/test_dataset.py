"""The LeRobot v2.1 writer: what it writes passes Dream's own dataset checks, run locally (nothing is sent)."""

import json
import pathlib

import dream
import numpy as np
import pyarrow.parquet as pq
import pytest
from conftest import write_episodes

from dream_hackathon import dataset, layouts

MAKERMODS_JOINTS = {
    "state": {"left_arm": [0, 6], "left_gripper": [6, 7], "right_arm": [7, 13], "right_gripper": [13, 14]},
    "action": {"left_arm": [0, 6], "left_gripper": [6, 7], "right_arm": [7, 13], "right_gripper": [13, 14]},
    "video": layouts.embodiment("makermods")["video"],
    "head_camera": "scene",
}


def test_yam_dataset_passes_dreams_checks(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "yam-task"
    write_episodes(root, "yam", episodes=2, frames=60, tasks=["pick up the cube", "stack the cups"])
    report = dream.datasets.validate_dataset(root, layouts.embodiment("yam"))
    assert report.ok, report.error
    assert (report.episodes, report.frames, report.fps) == (2, 120, 30)


def test_makermods_collected_dataset_passes_dreams_checks(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "metal-task"
    write_episodes(root, "makermods", episodes=2, frames=50, tasks=["fold the towel"])
    report = dream.datasets.validate_dataset(root, MAKERMODS_JOINTS)
    assert report.ok, report.error


def test_a_new_session_appends_and_reuses_task_indices(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=1, frames=55, tasks=["pick up the cube"])
    write_episodes(root, "yam", episodes=2, frames=55, tasks=["pick up the cube", "stack the cups"])
    info = json.loads((root / "meta" / "info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["total_tasks"]) == (3, 165, 2)
    tasks = [json.loads(line) for line in (root / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert tasks == [{"task_index": 0, "task": "pick up the cube"}, {"task_index": 1, "task": "stack the cups"}]
    assert dream.datasets.validate_dataset(root, layouts.embodiment("yam")).ok


def test_a_too_short_episode_is_refused_and_leaves_nothing(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    keys = list(layouts.video_keys("yam").values())
    writer = dataset.DatasetWriter(root, 30, "yam", [f"v{i}" for i in range(14)], keys, 320, 180)
    episode = writer.start_episode("pick up the cube")
    for _ in range(dataset.min_frames(30) - 1):
        episode.add_frame(np.zeros(14), np.zeros(14), {key: np.zeros((180, 320, 3), np.uint8) for key in keys})
    with pytest.raises(dataset.DatasetError, match="too short"):
        episode.save()
    assert not root.exists()


def test_a_dataset_recorded_another_way_is_refused(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=1, frames=50, tasks=["pick up the cube"])
    keys = list(layouts.video_keys("yam").values())
    with pytest.raises(dataset.DatasetError, match="fps"):
        dataset.DatasetWriter(root, 15, "yam", [f"v{i}" for i in range(14)], keys, 320, 180)
    with pytest.raises(dataset.DatasetError, match="cameras"):
        dataset.DatasetWriter(root, 30, "yam", [f"v{i}" for i in range(14)], keys[:2], 320, 180)


def test_dropping_episodes_renumbers_the_rest_and_dream_still_accepts_it(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=5, frames=50, tasks=["a", "b", "c", "d", "e"])
    left = dataset.drop_episodes(root, {1, 3})
    assert left == 3
    episodes = dataset.list_episodes(root)
    assert [(e.index, e.tasks) for e in episodes] == [(0, ("a",)), (1, ("c",)), (2, ("e",))]
    first_frame = 0
    for episode in episodes:
        table = pq.read_table(root / f"data/chunk-000/episode_{episode.index:06d}.parquet")
        assert set(table.column("episode_index").to_pylist()) == {episode.index}
        assert table.column("index").to_pylist() == list(range(first_frame, first_frame + 50))
        first_frame += 50
    for key in layouts.video_keys("yam").values():
        assert sorted(p.name for p in (root / "videos/chunk-000" / key).iterdir()) == [
            f"episode_{i:06d}.mp4" for i in range(3)
        ]
    info = json.loads((root / "meta" / "info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["total_videos"]) == (3, 150, 9)
    report = dream.datasets.validate_dataset(root, layouts.embodiment("yam"))
    assert report.ok, report.error
    # A new session continues the numbering after the drop.
    write_episodes(root, "yam", episodes=1, frames=50, tasks=["f"])
    assert [e.index for e in dataset.list_episodes(root)] == [0, 1, 2, 3]
    assert dream.datasets.validate_dataset(root, layouts.embodiment("yam")).ok


def test_dropping_an_unknown_episode_changes_nothing(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=2, frames=50, tasks=["a"])
    with pytest.raises(dataset.DatasetError, match="no episode 5"):
        dataset.drop_episodes(root, {0, 5})
    assert len(dataset.list_episodes(root)) == 2
