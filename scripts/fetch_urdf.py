"""Downloads a MakerMods arm's URDF, for the teleop IK, from makermodslab at the commit this repo pins.

```
uv run python scripts/fetch_urdf.py metal    # or: maker
```

Only the URDF file is needed: the IK reads the arm's joints and links, never its meshes. The file lands where the
station configs look for it (`makermods.urdf`), and its SHA-256 is checked against the pinned commit's.
"""

import argparse
import hashlib
import pathlib
import urllib.request

## makermodslab's commit, the one pyproject.toml pins.
COMMIT = "e377149419eebef7fa0f694b0d616c9c2cd46231"
BASE_URL = f"https://raw.githubusercontent.com/makermods-robotics/makermodslab/{COMMIT}/frontend/public"
## Per arm: the file in makermodslab, where it is saved, and its SHA-256 at COMMIT.
URDFS = {
    "metal": (
        "metal-urdf/metal_with_gripper.urdf",
        "metal_with_gripper.urdf",
        "a4792f244025ba4676de75a64acd6ce658fe65db84355fe70696640d209995b7",
    ),
    "maker": (
        "maker-urdf/robot.urdf",
        "maker_arm.urdf",
        "9771d91866fe4c00d8e6a7f41cc00ff5eff8c5b500edc455d050b25e000706f6",
    ),
}
OUT_DIR = pathlib.Path("~/.cache/dream_hackathon/urdf").expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("arm", choices=sorted(URDFS))
    arm = parser.parse_args().arm
    source, name, sha256 = URDFS[arm]
    with urllib.request.urlopen(f"{BASE_URL}/{source}", timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"{source}: the download does not match makermodslab {COMMIT[:7]}; not saved")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_bytes(data)
    print(f"saved {OUT_DIR / name}")


if __name__ == "__main__":
    main()
