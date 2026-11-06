"""The dashboard serves its page, the state, the cameras and the episodes' videos, and nothing else."""

import json
import pathlib
import urllib.error
import urllib.request

import numpy as np
import pytest
from conftest import write_episodes

from dream_hackathon import cameras, dashboard, layouts


def _get(board: dashboard.Dashboard, path: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(f"http://127.0.0.1:{board.port}{path}", headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


@pytest.fixture
def board(tmp_path: pathlib.Path):
    root = tmp_path / "task"
    write_episodes(root, "yam", episodes=2, frames=50, tasks=["pick up the cube"])
    frame = np.full((180, 320, 3), 128, np.uint8)
    stalled = {"wrist_r"}

    def frame_of(role: str):
        if role in stalled:
            raise cameras.CameraError(f"camera '{role}' stalled")
        return frame

    board = dashboard.Dashboard(0, frame_of, layouts.video_keys("yam"), root, title="test station")
    board.stalled = stalled
    yield board
    board.close()


def test_serves_the_page_and_the_state(board: dashboard.Dashboard) -> None:
    status, page = _get(board, "/")
    assert status == 200 and b"<title>Station dashboard</title>" in page
    board.publish({"mode": "live", "recording": True})
    board.event("saved episode 0")
    status, body = _get(board, "/api/status")
    state = json.loads(body)
    assert state["snapshot"]["recording"] and state["events"][0].endswith("saved episode 0")
    assert state["cameras"] == list(layouts.CAMERA_ROLES)


def test_cameras_as_jpeg_and_a_stall_as_503(board: dashboard.Dashboard) -> None:
    status, body = _get(board, "/api/camera/wrist_r.jpg")
    assert status == 503 and b"stalled" in body
    assert _get(board, "/api/camera/overhead.jpg")[0] == 200  # one stalled camera leaves the others on the page
    board.stalled.clear()
    status, body = _get(board, "/api/camera/wrist_r.jpg")
    assert status == 200 and body[:2] == b"\xff\xd8"  # a JPEG
    assert _get(board, "/api/camera/kitchen.jpg")[0] == 404


def test_episodes_and_their_videos_with_ranges(board: dashboard.Dashboard) -> None:
    episodes = json.loads(_get(board, "/api/episodes")[1])
    assert [e["index"] for e in episodes] == [0, 1] and episodes[0]["frames"] == 50
    status, video = _get(board, "/api/video/1/wrist_l.mp4")
    assert status == 200 and video[4:8] == b"ftyp"
    status, part = _get(board, "/api/video/1/wrist_l.mp4", {"Range": "bytes=0-99"})
    assert status == 206 and part == video[:100]
    for path in (
        "/api/video/7/wrist_l.mp4",
        "/api/video/0/kitchen.mp4",
        "/api/video/0/../../meta.mp4",
        "/meta/info.json",
    ):
        assert _get(board, path)[0] == 404, path


def test_the_page_gets_each_devices_state_and_the_phase(board: dashboard.Dashboard) -> None:
    from dream_hackathon import devices

    board.devices([devices.Status("Quest"), devices.Status("camera wrist_l", "no frame for 3 s; check its USB cable")])
    board.phase("waiting")
    state = json.loads(_get(board, "/api/status")[1])
    assert state["phase"] == "waiting"
    assert state["devices"] == [
        {"name": "Quest", "ok": True, "problem": ""},
        {"name": "camera wrist_l", "ok": False, "problem": "no frame for 3 s; check its USB cable"},
    ]
    page = _get(board, "/")[1].decode()
    assert 'id="devices"' in page and "press A on the right controller" in page
