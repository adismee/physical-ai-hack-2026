"""Prints both arms' joints and grippers, read-only: nothing is commanded.

```
uv run python scripts/check_arms.py --config configs/yam.yaml
```

YAM arms float in i2rt's gravity compensation while this runs; MakerMods arms are opened with torque off, so support
them. Move each arm by hand and check the matching side's numbers change. Ctrl+C stops.
"""

import argparse
import time

import numpy as np

from dream_hackathon import arms, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    station = config.load(parser.parse_args().config)
    robot = arms.open_arms(station, energize=False)
    try:
        while True:
            joints, grips = robot.read()
            line = " | ".join(
                f"{side}: joints {np.array2string(np.degrees(joints[side]), precision=1, suppress_small=True)} deg"
                f" gripper {grips[side]:.2f}"
                for side in ("left", "right")
            )
            print(line, flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


if __name__ == "__main__":
    main()
