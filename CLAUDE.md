# CLAUDE.md

Guidance for Claude Code in this repository. The repo is the Dream hackathon kit: Meta Quest teleoperation of two follower arms (YAM through i2rt, Metal and Maker arms through MakerMods' stack), LeRobot v2.1 recording, and Dream API fine-tuning. `README.md` maps the code and docs.

## Who you are helping

- **Organizers** set up every station before the event. To set one up, check one or fix one, follow the `station-setup` skill (`/station-setup`), which walks `docs/hardware_setup.md`.
- **Participants** arrive at a station that is already set up. They collect demonstrations (`docs/data_collection.md`) and fine-tune (`docs/fine_tuning.md`). Never send a participant through setup. If a device does not work, tell them to call the staff.

## Safety rules (always)

1. **Never run anything that powers, moves or releases the arms.** The person runs these, with a hand on the emergency stop: `dhk-collect` without `--dry-run`, `dreamcli robot run` or `dreamcli robot up` without `--dry-run`, `dreamcli setup`, `dreamcli tcp`, `dreamcli motors set-mit`, and makermodslab's calibration. Give the exact command, say what will move, and wait for them.
2. **Ask before running anything that opens the arms, even read-only** (`scripts/check_arms.py`, `dhk-collect --dry-run`). YAM arms switch to gravity compensation, and MakerMods arms are opened with torque off and must be supported by hand or resting.
3. **Never fill in or estimate an arm's tool center point** (`arms.<side>.tcp_xyz_m`). Ask the person to measure and confirm it (`docs/hardware_setup.md`, "Tool center point"), and continue only once they have. A URDF link, another arm's numbers or the YAM's are not this arm's tool center point.
4. **Never guess `gripper_open_deg` or `gripper_closed_deg`.** Copy them from the config `dreamcli setup` wrote (`~/.dreamlabs/station.yaml`), matching each arm by its port.
5. **Never print, copy or commit an API key.** Credentials live in `~/.dreamlabs/credentials`; read it only to check that it exists.
6. **`tools/finetune.py` without `--check` queues a paid job** (the price is on the platform's Billing page). Confirm with the person before running it. `--check` is free and runs locally.
7. **`dhk-dataset drop` deletes recorded episodes.** Show the person what `dhk-dataset list` says and get their yes for the exact episode numbers first.

## Commands

- Station environment: `uv sync --extra yam` or `uv sync --extra makermods`, plus `--extra realsense` when a camera is a RealSense. One arm family per station.
- The station's config is `configs/station.yaml`, a filled-in copy of `configs/yam.yaml`, `metal.yaml` or `maker.yaml`; it is not committed. Load-check it with `uv run python -c "from dream_hackathon import config; config.load('configs/station.yaml')"`.
- The Dream steps run in `~/dream-env`, the Dream library's environment: `~/dream-env/bin/python tools/add_eef.py ...` (Metal and Maker arms only) and `~/dream-env/bin/python tools/finetune.py ...`.
- Datasets: `uv run dhk-dataset list <folder>`, `uv run dhk-dataset drop <folder> <n>...`.
- Stations have no screen: `dhk-collect` and `scripts/check_cameras.py --config ...` serve a dashboard on the station's port 8080, reached with `ssh -L 8080:localhost:8080 <user>@<station>`. Run long sessions inside tmux.
- Tests need no hardware: `uv sync --group dream && uv run pytest`.
- Dream's documentation: https://platform.dreamlabs.ai/api-documentation/, or the `dreamlabs-docs` MCP server when the person has added it to Claude Code (the Dream platform's Setup API page offers it).

## Changing the code

Format and lint with ruff (line length 119), and keep `uv run pytest` green. The recorded layouts and camera roles must keep matching what `dreamcli robot run` sends (`tests/test_layouts.py` checks this against the installed Dream library). Write everything here from public SDKs and docs.
