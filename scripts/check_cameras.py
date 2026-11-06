"""Lists the cameras on this computer, or shows the station's three cameras in a browser as they will be recorded.

```
uv run python scripts/check_cameras.py --list
uv run python scripts/check_cameras.py --config configs/station.yaml
```

The second form serves the dashboard page with the cameras alone, on the station's loopback address: from your
laptop, connect with `ssh -L 8080:localhost:8080 <user>@<station>` and open http://localhost:8080. Each role's frame
is cropped to 16:9 and resized to the recording size: check that `overhead` sees the whole workspace and each wrist
camera sees its gripper. Ctrl+C stops.
"""

import argparse
import pathlib
import socket
import time

from dream_hackathon import cameras, config, dashboard, layouts


def list_cameras() -> None:
    """Prints the stable names of the UVC devices and the serials of the RealSense cameras.

    `by-id` names a camera by its model and serial; `by-path` by the USB port it is plugged into, which is the one to
    use when two identical webcams share a serial. A webcam has two nodes: its `index0` captures, its `index1` carries
    metadata and cannot.
    """
    for folder in (pathlib.Path("/dev/v4l/by-id"), pathlib.Path("/dev/v4l/by-path")):
        for path in sorted(folder.iterdir()) if folder.exists() else []:
            usable = "" if path.name.endswith("index0") else "   (metadata node: not this one)"
            print(f"uvc        device: {path}  ->  {path.resolve()}{usable}")
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("(RealSense: install the realsense extra to list them)")
        return
    for device in rs.context().query_devices():
        name, serial = device.get_info(rs.camera_info.name), device.get_info(rs.camera_info.serial_number)
        print(f"realsense  serial: {serial}  ({name})")


def preview(station: config.StationConfig, port: int) -> None:
    """Serves the cameras on the dashboard until Ctrl+C."""
    open_cameras = cameras.open_cameras(station.cameras, station.record_width, station.record_height)

    def frames() -> dict:
        return {role: camera.latest() for role, camera in open_cameras.items()}

    board = dashboard.Dashboard(
        port, frames, layouts.video_keys(station.family), None, title=f"{socket.gethostname()}: camera check"
    )
    print(board.how_to_open())
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        board.close()
        for camera in open_cameras.values():
            camera.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="list the cameras on this computer")
    parser.add_argument("--config", help="show this station's cameras in the browser")
    parser.add_argument("--port", type=int, default=dashboard.DEFAULT_PORT, help="the page's port on the station")
    args = parser.parse_args()
    if args.list or not args.config:
        list_cameras()
        return
    try:
        station = config.load(args.config)
    except config.ConfigError as error:
        raise SystemExit(f"config: {error}") from None
    preview(station, args.port)


if __name__ == "__main__":
    main()
