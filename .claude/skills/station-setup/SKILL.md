---
name: station-setup
description: Walk a hackathon organizer through setting up, checking or fixing one station (two arms, three cameras, a Meta Quest and the Dream library) until a dry run is right and the station is ready for participants. Use when asked to set up, check, verify or troubleshoot a station.
---

# Station setup

You and the organizer set up one station together, following `docs/hardware_setup.md`. You run the checks, read their output and edit the config. The organizer does everything physical, everything that needs `sudo`, and everything that powers or moves an arm. Keep `CLAUDE.md`'s safety rules throughout; this skill never relaxes them.

Work through the steps in order, one at a time. After each, say what you found and what comes next. When a check fails, find the cause before moving on, using `docs/hardware_setup.md` and the `dreamlabs-docs` MCP server for the Dream library.

## 0. Ask first

- Which arms this station has: `yam`, `metal` or `maker`.
- Whether any camera is an Intel RealSense.
- Whether the station is new or was set up before. If it was, read `configs/station.yaml` and run the checks from step 5 to find what changed.

## 1. The computer

1. Check the tools: `git lfs version`, `uv --version`, `adb version`. For anything missing, give the organizer the install line from `docs/hardware_setup.md` (step 1) to run; it needs `sudo`.
2. Install the station's environment: `uv sync --extra yam` or `uv sync --extra makermods`, plus `--extra realsense` if needed. If the Quest app fails to install from git LFS, check that `git lfs install` was run.
3. Run the hardware-free tests in the station's environment: `uv run pytest tests/test_dataset.py tests/test_teleop.py tests/test_ik.py tests/test_collect.py tests/test_config.py`.

## 2. The station config

1. Copy the template: `cp configs/<arm>.yaml configs/station.yaml`.
2. Ports:
   - YAM: `ip link show` lists the CAN interfaces; i2rt's guide gives them fixed names, `can_follower_l` and `can_follower_r`. If they are down, give the organizer the `sudo ip link set ... up type can bitrate 1000000` lines.
   - Metal and Maker: `ls -l /dev/serial/by-id/` lists the adapters. Ask the organizer to unplug one arm's adapter and see which entry disappears, so left and right are not swapped.
3. Cameras: `uv run python scripts/check_cameras.py --list`. Use the `/dev/v4l/by-id/...` names, not `/dev/videoN`. Ask the organizer which camera is the overhead one and which is on each wrist.
4. Set `dataset_dir` to where this station's datasets should go, and `task` to the event's default task if there is one.
5. Load-check the file (the command is in `CLAUDE.md`), and fix whatever key it names.

## 3. The Dream library

1. Check that `~/dream-env/bin/dreamcli` and `~/.dreamlabs/credentials` exist. Never print the credentials. If either is missing, the organizer runs the two steps on the Dream Getting started page with the team's API key (the `dreamlabs-docs` MCP server has them).
2. The organizer runs `dreamcli setup` for this arm type (rule 1: you never run it). It writes `~/.dreamlabs/station.yaml`. During its camera step, they give each camera the same role as in `configs/station.yaml`.

## 4. Metal and Maker arms only

1. The organizer calibrates each arm's zero in makermodslab, if it was not done already.
2. Copy `gripper_open_deg` and `gripper_closed_deg` for each arm from `~/.dreamlabs/station.yaml` into `configs/station.yaml`, matching arms by port. Never guess them.
3. Download the URDF: `uv run python scripts/fetch_urdf.py metal` (or `maker`).
4. Tool center point: ask the organizer to measure each arm's (`docs/hardware_setup.md`, "Tool center point") and tell you the numbers, then write them into `arms.<side>.tcp_xyz_m`. Never fill them in yourself; if they are unsure, the station waits.
5. Remind the organizer that both arms are brakeless: start every session with them resting where they can go limp.

## 5. Device checks

1. Quest: the organizer turns on Developer Mode, plugs in the headset and accepts USB debugging. Then you run `uv run python scripts/check_quest.py` while they move the controllers and press A, B and the right joystick. Every control should show up, on the right side.
2. Cameras: `uv run python scripts/check_cameras.py --config configs/station.yaml` opens a preview window. The organizer checks that each view is correct, then fixes the cameras in place.
3. Arms: ask first (rule 2), and have the organizer support MakerMods arms. Then run `uv run python scripts/check_arms.py --config configs/station.yaml` while they move each arm by hand; the matching side's numbers should change. Stop it with Ctrl+C.

## 6. Dry run

Ask first (rule 2). The organizer runs `uv run dhk-collect --config configs/station.yaml --dry-run` and reads the status lines to you, or you run it if they agree; it never commands a motor. Check both things in `docs/hardware_setup.md` step 5:

1. With the grips released, moving an arm by hand moves its tool position the right way. On a Metal or Maker arm, a joint going the wrong way means a sign in `makermods.joint_signs` is flipped.
2. With a grip held, moving the hand forward, left and up moves that tool the same way in the arm's base frame (the frames are in `docs/hardware_setup.md`). Fix an axis in that arm's `quest_to_base`, keeping the matrix a rotation.

Then set each arm's `workspace_min_m` and `workspace_max_m` from the tool positions at the edges of the table.

## 7. First live test (the organizer's)

Give the organizer the live command and the e-stop reminder. It records into a throwaway folder, so the participants' dataset stays empty:

```bash
uv run dhk-collect --config configs/station.yaml --task "station test" --dataset-dir ~/hackathon_data/station-test
```

They drive both arms slowly, record two short episodes with A, and quit with Ctrl+C. Then check the result with `~/dream-env/bin/python tools/finetune.py ~/hackathon_data/station-test --arm <arm> --check`. On a Metal or Maker arm, run `tools/add_eef.py` on it first and check the `-eef` folder.

## 8. Hand-off

Summarize for the organizer:

- The station config's arm type, ports, cameras and `dataset_dir`.
- What was checked, and what passed.
- Anything left open, such as an unmeasured tool center point or a camera to re-mount.

The station is ready when the config loads, the dry run's directions are right and the test dataset passes `--check`.
