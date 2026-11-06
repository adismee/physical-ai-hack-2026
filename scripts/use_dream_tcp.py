"""Writes Dream's own tool center point into a Metal or Maker station config, for both arms.

```
~/dream-env/bin/python scripts/use_dream_tcp.py configs/station.yaml
```

`dreamcli robot run` drives a Metal or Maker arm about the tool point of Dream's own model of that arm (`dream_ik`),
and `tools/add_eef.py` records the training data's tool poses about the same point. This writes it into the kit's
`arms.<side>.tcp_xyz_m`, so teleoperation pivots about that point too. It is read from the installed Dream library,
not copied by hand, and it is in the flange's frame of both Dream's URDF and the kit's, which agree up to the flange.

It fits the stock MakerMods gripper, mounted the stock way. A different gripper needs its own point measured
(docs/hardware_setup.md, "Tool center point"). Either way, check it on the arm: in a dry run, hold the closed
fingertips on a mark and turn the wrist by hand; the dashboard's tool position should stay within about a centimetre.

Run it in `~/dream-env`, the Dream library's environment: the station's own environment has no `dream_ik`.
"""

import argparse
import pathlib
import re
import sys

import yaml

## The kit's arm types, by Dream's names for their models.
DREAM_ARMS = {"metal": "METAL_ARM", "maker": "MAKERS_ARM"}
_TCP_LINE = re.compile(r"^(?P<indent>\s+)tcp_xyz_m:.*$", re.MULTILINE)


def dream_tcp(arm: str) -> tuple[str, tuple[float, float, float]]:
    """Dream's model of a Metal or Maker arm: its flange link, and its tool point there in metres."""
    from dream_ik import specs

    spec = getattr(specs, DREAM_ARMS[arm])
    offset = spec.tools["tcp"].offset
    return spec.flange, (float(offset[0]), float(offset[1]), float(offset[2]))


def write_tcp(text: str, tcp: tuple[float, float, float]) -> str:
    """The config's text with both arms' `tcp_xyz_m` set to `tcp`; every other line is kept as it was."""
    if len(_TCP_LINE.findall(text)) != 2:
        raise SystemExit("expected one 'tcp_xyz_m:' line per arm (two in all) in the config; edit it by hand")
    values = ", ".join(repr(round(value, 7)) for value in tcp)
    line = f"tcp_xyz_m: [{values}]  # Dream's tool point for the stock gripper (scripts/use_dream_tcp.py)"
    return _TCP_LINE.sub(lambda match: match.group("indent") + line, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="the station config, e.g. configs/station.yaml")
    path = pathlib.Path(parser.parse_args().config)
    text = path.read_text()
    raw = yaml.safe_load(text)
    arm = raw.get("arm")
    if arm not in DREAM_ARMS:
        sys.exit(f"{path}: 'arm' is {arm!r}; Dream's tool point is for a metal or maker arm (a YAM uses i2rt's)")
    try:
        flange, tcp = dream_tcp(arm)
    except ImportError:
        sys.exit("no Dream library here: run it with ~/dream-env/bin/python, the Dream library's environment")
    config_flange = (raw.get("makermods") or {}).get("flange_link")
    if config_flange != flange:
        sys.exit(f"{path}: makermods.flange_link is {config_flange!r}, but Dream's {arm} tool point is in {flange!r}")
    updated = write_tcp(text, tcp)
    written = yaml.safe_load(updated)["arms"]
    if any(written[side]["tcp_xyz_m"] != [round(value, 7) for value in tcp] for side in ("left", "right")):
        sys.exit("the new tcp_xyz_m lines did not read back as written; nothing changed")
    path.write_text(updated)
    print(f"{path}: both arms' tcp_xyz_m set to Dream's {arm} tool point {list(tcp)} m, in {flange}'s frame")
    print("check it on each arm: dry run, fingertips on a mark, turn the wrist; the tool position should stay put")


if __name__ == "__main__":
    main()
