"""Prints the Quest controllers' positions, grips, triggers and button presses.

```
uv run python scripts/check_quest.py              # over USB
uv run python scripts/check_quest.py --ip 10.0.0.5  # over Wi-Fi, after `adb tcpip 5555`
```

The first run installs the Quest app on the headset (accept the USB debugging prompt inside it). Ctrl+C stops.
"""

import argparse
import time

from dream_hackathon import quest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ip", help="the Quest's address for adb over Wi-Fi (default: USB)")
    headset = quest.Quest(parser.parse_args().ip)
    try:
        while True:
            state = headset.read()
            parts = []
            for side in ("left", "right"):
                pose = state.poses[side]
                where = "not tracked" if pose is None else "xyz [{:+.3f} {:+.3f} {:+.3f}] m".format(*pose[:3, 3])
                parts.append(f"{side}: {where} grip {state.grip[side]:.2f} trigger {state.trigger[side]:.2f}")
            if state.pressed:
                parts.append(f"pressed {sorted(state.pressed)}")
            print(" | ".join(parts), flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        headset.close()


if __name__ == "__main__":
    main()
