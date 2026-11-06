"""One program drives a station's arms at a time.

The lock is an exclusive `flock` on `~/.dreamlabs/arms.lock`, the same file `dreamcli` locks before it drives the
arms, so a collector, a device check and a `dreamcli` run on the station's account refuse each other. The kernel
releases the lock when its holder exits for any reason, so a crashed session never leaves a station locked.

The file's first line is `<pid> <holder>`, the format `dreamcli` writes; ours adds a second line of JSON saying who
started it, from where, and which dataset it records into.

@see acquire(), holder()
"""

import datetime
import fcntl
import getpass
import json
import os
import pathlib
from typing import IO, Any

## dreamcli's lock on the arms (`robots.arms_lock`, in its home `~/.dreamlabs`).
LOCK_PATH = pathlib.Path("~/.dreamlabs/arms.lock").expanduser()


class StationBusyError(RuntimeError):
    """Another program holds the station's arms; the message says which."""


def acquire(holder: str, dataset: pathlib.Path | None = None, path: pathlib.Path | None = None) -> IO[str]:
    """Takes the station's lock; it is held until the returned file is closed or the process exits.

    @param holder What holds it, e.g. `"dhk-collect"`.
    @param dataset The dataset folder being recorded into, if any.
    @param path The lock file; None for LOCK_PATH.
    @return The open lock file.
    @throws StationBusyError Another program holds the lock.
    """
    path = path or LOCK_PATH
    handle = _try_lock(path)
    if handle is None:
        raise StationBusyError(describe(_parse(_read(path))))
    details = {
        "user": getpass.getuser(),
        "from": os.environ.get("SSH_CONNECTION", "").split(" ")[0] or "this computer",
        "since": datetime.datetime.now().strftime("%H:%M"),
        "dataset": str(dataset) if dataset is not None else None,
    }
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()} {holder}\n{json.dumps(details)}\n")
    handle.flush()
    return handle


def holder(path: pathlib.Path | None = None) -> dict[str, Any] | None:
    """Who holds the lock now, without keeping it.

    @param path The lock file; None for LOCK_PATH.
    @return `{"pid", "holder", ...}` (our details when a collector holds it), or None when it is free.
    """
    path = path or LOCK_PATH
    handle = _try_lock(path)
    if handle is not None:
        handle.close()
        return None
    return _parse(_read(path))


def describe(found: dict[str, Any]) -> str:
    """The refusal line for a held lock, from its holder() record."""
    who = found["holder"] or "another program"
    pid = f" (pid {found['pid']})" if found["pid"] else ""
    extra = ""
    if found.get("user"):
        extra = f", started by {found['user']} from {found.get('from')} at {found.get('since')}"
    if found.get("dataset"):
        extra += f", recording into {found['dataset']}"
    return f"the station's arms are in use by {who}{pid}{extra}: stop it (Ctrl+C in its terminal), then try again"


def _try_lock(path: pathlib.Path) -> IO[str] | None:
    """The locked file, or None when someone else holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # "a+" never empties the file on opening, so a refused opener cannot wipe out the holder's details.
    handle = open(path, "a+", encoding="utf-8")  # noqa: SIM115 - held open: the lock lasts as long as the file
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse(text: str) -> dict[str, Any]:
    """`{"pid", "holder"}` from the first line, plus our JSON details from the second when they are there."""
    first, _, rest = text.strip().partition("\n")
    pid, _, name = first.partition(" ")
    found: dict[str, Any] = {"pid": pid if pid.isdigit() else None, "holder": name.strip() or None}
    try:
        details = json.loads(rest) if rest else {}
    except ValueError:
        details = {}
    if isinstance(details, dict):
        found.update(details)
    return found
