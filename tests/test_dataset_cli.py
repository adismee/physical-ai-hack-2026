"""dhk-dataset: lists, drops after asking, and refuses while a collector records into the same folder."""

import pathlib

import pytest
from conftest import write_episodes

from dream_hackathon import dataset, dataset_cli, station_lock


def test_list_and_drop(tmp_path: pathlib.Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(station_lock, "LOCK_PATH", tmp_path / "arms.lock")
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=3, frames=60, tasks=["pick up the cube"])
    dataset_cli.main(["list", str(root)])
    assert "3 episodes" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    dataset_cli.main(["drop", str(root), "1"])
    assert len(dataset.list_episodes(root)) == 3
    dataset_cli.main(["drop", str(root), "1", "--yes"])
    assert len(dataset.list_episodes(root)) == 2


def test_refuses_while_a_collector_records_into_it(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setattr(station_lock, "LOCK_PATH", tmp_path / "arms.lock")
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=2, frames=60, tasks=["pick up the cube"])
    lock = station_lock.acquire("dhk-collect", root.resolve())
    with pytest.raises(SystemExit, match="recording into"):
        dataset_cli.main(["drop", str(root), "0", "--yes"])
    lock.close()
    dataset_cli.main(["drop", str(root), "0", "--yes"])
    assert len(dataset.list_episodes(root)) == 1
