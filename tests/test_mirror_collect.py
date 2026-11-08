"""mirror_collect's dashboard: who may talk to it, what a click does, and how homing and saving fail safely."""

import dataclasses
import http.server
import importlib.util
import json
import pathlib
import queue
import socket
import threading
from collections.abc import Iterator
from types import ModuleType

import numpy as np
import pytest

from dream_hackathon import dataset

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "mirror_collect.py"
HOME = np.zeros(6)


@pytest.fixture(scope="module")
def mc() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mirror_collect", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Camera:
    def latest(self) -> np.ndarray:
        return np.full((36, 64, 3), 128, np.uint8)


@dataclasses.dataclass
class _Server:
    port: int
    state: object
    commands: "queue.Queue[str]"


@pytest.fixture
def server(mc: ModuleType) -> Iterator[_Server]:
    state = mc._DashState(task="pick it up")
    commands: queue.Queue[str] = queue.Queue()
    handler = mc._make_handler(state, commands, {"overhead": _Camera()})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield _Server(httpd.server_address[1], state, commands)
    httpd.shutdown()
    httpd.server_close()


def _raw(port: int, method: str, path: str, headers: dict[str, str], body: bytes = b"") -> tuple[int, bytes]:
    """Sends one request with exactly these headers (http.client would add its own Host and Content-Length)."""
    head = "\r\n".join([f"{method} {path} HTTP/1.0", *(f"{k}: {v}" for k, v in headers.items())])
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(head.encode() + b"\r\n\r\n" + body)
        data = b""
        while chunk := sock.recv(65536):
            data += chunk
    status_line, _, payload = data.partition(b"\r\n\r\n")
    return int(status_line.split(b" ", 2)[1]), payload


def _ok(port: int) -> dict[str, str]:
    return {"Host": f"127.0.0.1:{port}", "X-Mirror-Dashboard": "1"}


def _post_task(server: _Server, body: bytes, **headers: str) -> tuple[int, bytes]:
    sent = {**_ok(server.port), "Content-Length": str(len(body)), **headers}
    return _raw(server.port, "POST", "/api/task", sent, body)


# ------------------------------------------------------------------ who may talk to the dashboard


@pytest.mark.parametrize("host", ["localhost", "localhost:8090", "LOCALHOST:9000", "127.0.0.1:8090", "[::1]:8090"])
def test_loopback_hosts_are_allowed(mc: ModuleType, host: str) -> None:
    assert mc._host_allowed(host)


@pytest.mark.parametrize(
    "host",
    [None, "", "evil.example", "evil.example:8090", "localhost.evil.example", "127.0.0.1.evil.example:80",
     "localhost:abc", "localhost:123456", "[::1", "0.0.0.0:8090", "localhost:8090/x"],
)  # fmt: skip
def test_other_hosts_are_refused(mc: ModuleType, host: str | None) -> None:
    assert not mc._host_allowed(host)


def test_a_rebound_host_gets_nothing(server: _Server) -> None:
    rebound = {"Host": "evil.example:8080", "X-Mirror-Dashboard": "1"}
    for method, path in (
        ("GET", "/"),
        ("GET", "/api/status"),
        ("GET", "/api/camera/overhead.jpg"),
        ("POST", "/api/home"),
        ("POST", "/api/episode/start"),
    ):
        assert _raw(server.port, method, path, rebound)[0] == 403, (method, path)
    assert server.commands.empty()


def test_the_page_and_the_cameras_load_for_the_loopback_host(server: _Server) -> None:
    assert _raw(server.port, "GET", "/", {"Host": f"localhost:{server.port}"})[0] == 200
    assert _raw(server.port, "GET", "/api/camera/overhead.jpg", {"Host": f"[::1]:{server.port}"})[0] == 200


def test_a_cross_origin_post_is_refused(server: _Server) -> None:
    assert _raw(server.port, "POST", "/api/home", {**_ok(server.port), "Origin": "http://evil.example"})[0] == 403
    assert server.commands.empty()
    same_origin = {**_ok(server.port), "Origin": f"http://127.0.0.1:{server.port}"}
    assert _raw(server.port, "POST", "/api/home", same_origin)[0] == 200


def test_a_post_without_the_dashboard_header_is_refused(server: _Server) -> None:
    assert _raw(server.port, "POST", "/api/home", {"Host": f"127.0.0.1:{server.port}"})[0] == 403
    assert server.commands.empty()


# ------------------------------------------------------------------ what a click does


def test_start_and_save_are_separate_commands(server: _Server) -> None:
    for path in ("/api/episode/start", "/api/episode/save", "/api/episode/discard", "/api/home"):
        assert _raw(server.port, "POST", path, _ok(server.port))[0] == 200
    assert [server.commands.get_nowait() for _ in range(4)] == ["start", "save", "d", "h"]


@pytest.mark.parametrize(
    ("cmd", "recording", "action"),
    [
        ("", False, "start"),  # Enter and `s` on the keyboard toggle
        ("", True, "save"),
        ("s", False, "start"),
        ("s", True, "save"),
        ("start", False, "start"),
        ("start", True, "ignore"),  # a second click on Start must not save the episode it just began
        ("save", True, "save"),
        ("save", False, "ignore"),  # a second click on Save must not begin a new one
    ],
)
def test_what_a_start_or_save_command_does(mc: ModuleType, cmd: str, recording: bool, action: str) -> None:
    assert mc._episode_action(cmd, recording) == action


# ------------------------------------------------------------------ the task text


def test_the_task_is_cleaned(server: _Server) -> None:
    status, _ = _post_task(server, json.dumps({"task": "  pick\nup\tthe cube  "}).encode())
    assert status == 200 and server.state.task == "pick up the cube"


def test_a_task_is_cut_to_dreams_limit(server: _Server, mc: ModuleType) -> None:
    assert _post_task(server, json.dumps({"task": "x" * 600}).encode())[0] == 200
    assert server.state.task == "x" * mc.TASK_MAX_CHARS


def test_a_blank_task_falls_back_to_the_default(mc: ModuleType) -> None:
    assert mc._task_or_default("   ") == mc.DEFAULT_TASK
    assert mc._task_or_default("") == mc.DEFAULT_TASK
    assert mc._task_or_default("  go  ") == "go"


@pytest.mark.parametrize(("length", "status"), [("99999999", 413), ("abc", 400), ("-5", 400), ("5.0", 400), ("", 400)])
def test_a_bad_length_is_refused_without_reading_the_body(server: _Server, length: str, status: int) -> None:
    headers = {**_ok(server.port), "Content-Length": length}
    assert _raw(server.port, "POST", "/api/task", headers)[0] == status
    assert server.state.task == "pick it up"


def test_a_missing_length_is_refused(server: _Server) -> None:
    assert _raw(server.port, "POST", "/api/task", _ok(server.port))[0] == 400


@pytest.mark.parametrize("body", [b"{nope", b"[1]", b"\xff\xfe", b""])
def test_a_body_that_is_not_a_json_object_is_a_400(server: _Server, body: bytes) -> None:
    assert _post_task(server, body)[0] == 400
    assert server.state.task == "pick it up"


# ------------------------------------------------------------------ homing and saving fail safely


def test_homing_ends_when_every_joint_is_at_home(mc: ModuleType) -> None:
    assert mc._home_step(HOME + 5e-4, HOME, 0.1, 0.0) == ("reached", None)


def test_homing_moves_in_capped_steps(mc: ModuleType) -> None:
    outcome, command = mc._home_step(np.array([1.0, -1.0, 0.05, 0, 0, 0]), HOME, 0.1, 1.0)
    assert outcome == "move"
    assert np.allclose(command, [0.9, -0.9, 0.0, 0, 0, 0])


def test_homing_gives_up_after_the_timeout(mc: ModuleType) -> None:
    far = np.full(6, 1.0)
    assert mc._home_step(far, HOME, 0.1, mc.HOME_TIMEOUT_S - 1)[0] == "move"
    assert mc._home_step(far, HOME, 0.1, mc.HOME_TIMEOUT_S + 1) == ("timeout", None)


def test_homing_holds_while_the_follower_cannot_be_read(mc: ModuleType) -> None:
    """A failed read must not be replaced by a placeholder pose: the arm would be ramped from there."""
    assert mc._home_step(None, HOME, 0.1, 1.0) == ("hold", None)
    assert mc._home_step(None, HOME, 0.1, mc.HOME_TIMEOUT_S + 1) == ("timeout", None)


class _Episode:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    def save(self) -> int:
        if self._error:
            raise self._error
        return 4


def test_a_saved_episode_returns_its_index(mc: ModuleType) -> None:
    assert mc._save_episode(_Episode()) == 4


def test_a_short_episode_is_reported_and_the_session_goes_on(mc: ModuleType, capsys: pytest.CaptureFixture) -> None:
    assert mc._save_episode(_Episode(dataset.DatasetError("episode too short: 3 frames"))) is None
    assert "episode too short" in capsys.readouterr().out


def test_any_other_save_error_still_stops_the_session(mc: ModuleType) -> None:
    with pytest.raises(OSError, match="disk full"):
        mc._save_episode(_Episode(OSError("disk full")))
