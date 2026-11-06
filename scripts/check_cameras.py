"""Lists the cameras on this computer, or previews the station's three cameras as they will be recorded.

```
uv run python scripts/check_cameras.py --list
uv run python scripts/check_cameras.py --config configs/yam.yaml
```

The preview shows each role's frame cropped to 16:9 and resized to the recording size: check that `overhead` sees the
whole workspace and each wrist camera sees its gripper. Press q in the window to stop.
"""

import argparse
import pathlib

import cv2
import numpy as np

from dream_hackathon import cameras, config, layouts


def list_cameras() -> None:
    """Prints the stable names of the UVC devices and the serials of the RealSense cameras."""
    by_id = pathlib.Path("/dev/v4l/by-id")
    for path in sorted(by_id.iterdir()) if by_id.exists() else []:
        print(f"uvc        device: {path}  ->  {path.resolve()}")
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("(RealSense: install the realsense extra to list them)")
        return
    for device in rs.context().query_devices():
        name, serial = device.get_info(rs.camera_info.name), device.get_info(rs.camera_info.serial_number)
        print(f"realsense  serial: {serial}  ({name})")


def preview(station: config.StationConfig) -> None:
    open_cameras = cameras.open_cameras(station.cameras, station.record_width, station.record_height)
    try:
        while cv2.waitKey(30) & 0xFF != ord("q"):
            tiles = []
            for role in layouts.CAMERA_ROLES:
                frame = open_cameras[role].latest().copy()
                cv2.putText(frame, role, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                tiles.append(frame)
            cv2.imshow("station cameras (q to quit)", np.hstack(tiles))
    finally:
        for camera in open_cameras.values():
            camera.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="list the cameras on this computer")
    parser.add_argument("--config", help="preview this station's cameras")
    args = parser.parse_args()
    if args.list or not args.config:
        list_cameras()
        return
    preview(config.load(args.config))


if __name__ == "__main__":
    main()
