"""The station lock: one holder at a time, shared with dreamcli's, and a refusal that says who holds it."""

import pathlib

import pytest

from dream_hackathon import station_lock


def test_one_holder_at_a_time_and_released_on_close(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "arms.lock"
    lock = station_lock.acquire("dhk-collect", tmp_path / "cubes", path=path)
    held = station_lock.holder(path)
    assert held["holder"] == "dhk-collect" and held["dataset"] == str(tmp_path / "cubes")
    with pytest.raises(station_lock.StationBusyError, match=r"in use by dhk-collect \(pid \d+\).*recording into"):
        station_lock.acquire("scripts/check_arms.py", path=path)
    lock.close()
    assert station_lock.holder(path) is None
    station_lock.acquire("scripts/check_arms.py", path=path).close()


def test_reads_the_line_dreamcli_writes(tmp_path: pathlib.Path) -> None:
    found = station_lock._parse("4242 dreamcli robot run\n")
    assert found == {"pid": "4242", "holder": "dreamcli robot run"}
    assert station_lock.describe(found).startswith("the station's arms are in use by dreamcli robot run (pid 4242)")


def test_our_first_line_is_dreamclis_format(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "arms.lock"
    lock = station_lock.acquire("dhk-collect", path=path)
    pid, _, holder = path.read_text().splitlines()[0].partition(" ")
    assert pid.isdigit() and holder == "dhk-collect"
    lock.close()
