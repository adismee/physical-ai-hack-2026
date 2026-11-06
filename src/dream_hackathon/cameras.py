"""The station's cameras: each read on its own thread, the newest frame kept at the recording size.

Every frame is center-cropped to 16:9 and resized to the recording size, the same way for every camera: Dream-1 was
trained mostly on 16:9 video, so a 4:3 camera is cropped rather than squashed. Use a 16:9 capture mode (1280x720)
when the camera has one.

@see Camera, open_cameras()
"""

import contextlib
import threading
import time

import cv2
import numpy as np
import numpy.typing as npt

from dream_hackathon import config

## A camera with no new frame for this long is reported as stalled.
STALL_AFTER_S = 0.5
## A camera that keeps failing (unplugged, its stream gone) is reopened this often, until it is back.
REOPEN_EVERY_S = 1.0


class CameraError(RuntimeError):
    """A camera cannot be opened, or stopped delivering frames."""


class Camera:
    """One camera, read continuously on a thread; latest() never waits for the device.

    A camera that fails while open (a loose USB cable) is reopened every REOPEN_EVERY_S until it delivers again, so a
    session does not need restarting when it is plugged back in.

    Attributes:
    - role: `overhead`, `wrist_l` or `wrist_r`.
    """

    def __init__(self, cfg: config.CameraConfig, width: int, height: int) -> None:
        """Opens the device and starts reading.

        @param cfg The camera's config.
        @param width, height The recording size frames are cropped and resized to.
        @throws CameraError The device cannot be opened, or gives no frame within two seconds.
        """
        self.role = cfg.role
        self._cfg = cfg
        self._size = (width, height)
        reader = _open_reader(cfg)
        self._lock = threading.Lock()
        self._frame: npt.NDArray[np.uint8] | None = None
        self._frame_s = 0.0
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(reader,), name=f"camera-{cfg.role}", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 2.0
        while self._frame is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self._frame is None:
            self.close()
            raise CameraError(f"camera '{self.role}' ({cfg.kind}) gave no frame: {self._error or 'timed out'}")

    def latest(self) -> npt.NDArray[np.uint8]:
        """The newest frame, BGR, at the recording size.

        @throws CameraError The camera has given no new frame for STALL_AFTER_S.
        """
        with self._lock:
            frame, age_s = self._frame, time.monotonic() - self._frame_s
        if frame is None or age_s > STALL_AFTER_S:
            raise CameraError(f"camera '{self.role}' stalled ({self._error or f'no frame for {age_s:.1f} s'})")
        return frame

    def problem(self, after_s: float = STALL_AFTER_S) -> str | None:
        """None while frames arrive; once none has for `after_s`, what is wrong, for the dashboard."""
        with self._lock:
            age_s = time.monotonic() - self._frame_s
        if age_s <= after_s:
            return None
        cause = f": {self._error}" if self._error else ""
        return f"no frame for {age_s:.0f} s{cause}; check its USB cable (it reconnects by itself once it is back)"

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _run(self, reader: "_UvcReader | _RealSenseReader | None") -> None:
        misses_since_s: float | None = None
        try:
            while not self._stop.is_set():
                if reader is None:
                    try:
                        reader = _open_reader(self._cfg)
                    except Exception as error:  # noqa: BLE001 - shown by problem() and latest(), tried again
                        self._error = _describe(error)
                        self._stop.wait(REOPEN_EVERY_S)
                        continue
                try:
                    frame = reader()
                except Exception as error:  # noqa: BLE001 - a lost device: reopen it
                    self._error = _describe(error)
                    reader = _release(reader)
                    self._stop.wait(REOPEN_EVERY_S)
                    continue
                if frame is None:
                    # A webcam that was unplugged keeps answering "no frame" at once: reopen it after a while.
                    self._error = "the device returned no frame"
                    misses_since_s = misses_since_s or time.monotonic()
                    if time.monotonic() - misses_since_s > REOPEN_EVERY_S:
                        reader, misses_since_s = _release(reader), None
                    time.sleep(0.01)
                    continue
                misses_since_s = None
                small = to_record_size(frame, *self._size)
                with self._lock:
                    self._frame, self._frame_s, self._error = small, time.monotonic(), None
        finally:
            _release(reader)


def _open_reader(cfg: config.CameraConfig) -> "_UvcReader | _RealSenseReader":
    return _RealSenseReader(cfg) if cfg.kind == "realsense" else _UvcReader(cfg)


def _release(reader: "_UvcReader | _RealSenseReader | None") -> None:
    """Releases a reader, ignoring the errors of one whose device is gone; returns None, the slot's new value."""
    if reader is not None:
        with contextlib.suppress(Exception):  # the device is already gone
            reader.release()
    return None


def _describe(error: BaseException) -> str:
    return str(error) if isinstance(error, CameraError) else f"{type(error).__name__}: {error}"


def open_cameras(cfgs: tuple[config.CameraConfig, ...], width: int, height: int) -> dict[str, Camera]:
    """Opens every camera; closes the ones already open if one fails.

    @return Per role, its Camera.
    """
    cameras: dict[str, Camera] = {}
    try:
        for cfg in cfgs:
            cameras[cfg.role] = Camera(cfg, width, height)
    except BaseException:
        for camera in cameras.values():
            camera.close()
        raise
    return cameras


def to_record_size(frame: npt.NDArray[np.uint8], width: int, height: int) -> npt.NDArray[np.uint8]:
    """Center-crops a frame to the recording's aspect ratio, then resizes it.

    @param frame BGR `[h, w, 3]`.
    @param width, height The recording size.
    @return BGR `[height, width, 3]`.
    """
    h, w = frame.shape[:2]
    if w * height > h * width:  # wider than the target: trim the sides
        keep = h * width // height
        left = (w - keep) // 2
        frame = frame[:, left : left + keep]
    elif w * height < h * width:  # taller: trim top and bottom
        keep = w * height // width
        top = (h - keep) // 2
        frame = frame[top : top + keep]
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


class _UvcReader:
    """Reads one UVC camera with OpenCV, in MJPG so a USB port carries 720p at 30 fps."""

    def __init__(self, cfg: config.CameraConfig) -> None:
        assert cfg.device is not None
        device: int | str = int(cfg.device) if cfg.device.isdigit() else cfg.device
        self._capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._capture.isOpened():
            raise CameraError(f"camera '{cfg.role}': cannot open {cfg.device}")
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._capture.set(cv2.CAP_PROP_FPS, cfg.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def __call__(self) -> npt.NDArray[np.uint8] | None:
        ok, frame = self._capture.read()
        return frame if ok else None

    def release(self) -> None:
        self._capture.release()


class _RealSenseReader:
    """Reads one RealSense camera's color stream."""

    def __init__(self, cfg: config.CameraConfig) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise CameraError(
                "a RealSense camera needs the realsense extra: run `uv sync --extra yam --extra realsense` (or "
                "`--extra makermods --extra realsense`); name both, since a uv sync without one removes it"
            ) from error
        # pipeline.start() of a camera that is not connected blocks for 15 s holding Python's GIL, which freezes every
        # thread of the process (the control loop and the arms' SDK among them): ask first, which takes a tenth of a
        # second and freezes nothing.
        connected = {device.get_info(rs.camera_info.serial_number) for device in rs.context().query_devices()}
        if cfg.serial not in connected:
            others = ", ".join(sorted(connected)) or "none"
            raise CameraError(f"camera '{cfg.role}': RealSense {cfg.serial} is not connected (connected: {others})")
        self._pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(cfg.serial)
        rs_config.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
        try:
            self._pipeline.start(rs_config)
        except RuntimeError as error:
            raise CameraError(f"camera '{cfg.role}': RealSense {cfg.serial}: {error}") from error

    def __call__(self) -> npt.NDArray[np.uint8] | None:
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()).copy() if color else None

    def release(self) -> None:
        self._pipeline.stop()
