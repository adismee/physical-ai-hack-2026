"""The Meta Quest controllers, through oculus_reader (the open-source Quest app and its adb reader).

Buttons, as the collector uses them:

| Control | Does |
| --- | --- |
| Grip (middle finger), held | Engages that arm's clutch: the arm's tool follows the hand |
| Index trigger | That arm's gripper: released opens, pulled closes |
| A | Starts recording an episode; pressed again, saves it |
| B | Discards the episode being recorded |
| Right joystick click | Sends both arms back to where they started |

@see Quest, QuestState
"""

import dataclasses
import time
from typing import Any

import numpy as np
import numpy.typing as npt

## A controller pose older than this is treated as lost, so a stalled stream releases the clutch.
STALE_AFTER_S = 0.25
## oculus_reader's keys: per side, its pose, grip value and trigger value.
_POSE_KEY = {"right": "r", "left": "l"}
_GRIP_KEY = {"right": "rightGrip", "left": "leftGrip"}
_TRIGGER_KEY = {"right": "rightTrig", "left": "leftTrig"}
## The buttons the collector reacts to.
BUTTONS = ("A", "B", "RJ")


@dataclasses.dataclass(frozen=True)
class QuestState:
    """One reading of both controllers.

    Attributes:
    - poses: Per side, the controller's `[4, 4]` pose in the Quest's frame, or None when it is not tracked.
    - grip: Per side, the grip button, 0 to 1.
    - trigger: Per side, the index trigger, 0 to 1.
    - pressed: The buttons (BUTTONS) pressed since the last reading.
    """

    poses: dict[str, npt.NDArray[np.float64] | None]
    grip: dict[str, float]
    trigger: dict[str, float]
    pressed: frozenset[str]


class Quest:
    """Reads both controllers from a Quest over adb (USB, or Wi-Fi with `ip_address`)."""

    def __init__(self, ip_address: str | None = None) -> None:
        """Connects to the headset and starts its app; the app is installed on the first run.

        @param ip_address The headset's address for adb over Wi-Fi, or None for USB.
        """
        from oculus_reader.reader import OculusReader

        self._reader = OculusReader(ip_address=ip_address)
        self._last_transforms: Any = None
        self._last_change_s = 0.0
        self._held: set[str] = set()

    def read(self) -> QuestState:
        """The latest controller state; never blocks."""
        transforms, buttons = self._reader.get_transformations_and_buttons()
        now = time.monotonic()
        if transforms is not self._last_transforms:
            self._last_transforms = transforms
            self._last_change_s = now
        is_fresh = bool(transforms) and now - self._last_change_s < STALE_AFTER_S
        return parse(transforms if is_fresh else {}, buttons or {}, self._held)

    def close(self) -> None:
        self._reader.stop()


def parse(transforms: dict[str, Any], buttons: dict[str, Any], held: set[str]) -> QuestState:
    """Turns one oculus_reader sample into a QuestState.

    @param transforms oculus_reader's poses, `{"r": [4, 4], "l": [4, 4]}`; empty when lost.
    @param buttons oculus_reader's buttons: booleans by name and `(value,)` tuples for the analog controls.
    @param held The buttons held at the last reading; updated in place, so `pressed` holds only new presses.
    @return The state.
    """
    poses = {side: _pose(transforms.get(key)) for side, key in _POSE_KEY.items()}
    down = {name for name in BUTTONS if buttons.get(name) is True}
    pressed = frozenset(down - held)
    held.clear()
    held.update(down)
    return QuestState(
        poses=poses,
        grip={side: _analog(buttons.get(key)) for side, key in _GRIP_KEY.items()},
        trigger={side: _analog(buttons.get(key)) for side, key in _TRIGGER_KEY.items()},
        pressed=pressed,
    )


def _pose(value: Any) -> npt.NDArray[np.float64] | None:
    if value is None:
        return None
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    return pose


def _analog(value: Any) -> float:
    """An analog control's value, 0 to 1; 0 when the controller does not report it."""
    if not value:
        return 0.0
    return float(min(1.0, max(0.0, value[0])))
