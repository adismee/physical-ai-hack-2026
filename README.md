# Dream Hackathon Kit

Teach a pair of robot arms a task with a Meta Quest, then fine-tune Dream-1, the Dream Labs robot model, on your demonstrations and run it on the same arms.

Built for the [MakerMods Physical AI Hackathon](https://luma.com/pah) (November 7-8, San Francisco), presented with MakerMods and Dream Labs.

This kit drives two follower arms from the Quest's controllers, records each demonstration as a [LeRobot v2.1](https://github.com/huggingface/lerobot) dataset (the format Dream fine-tuning reads), and hands the dataset to the [Dream API](https://platform.dreamlabs.ai/api-documentation/fine-tuning/). It supports three kinds of arms, each driven through its maker's own open-source SDK:

| Arms | Driven by |
| --- | --- |
| [i2rt YAM](https://i2rt.com/) | [i2rt's Python SDK](https://github.com/i2rt-robotics/i2rt) |
| [MakerMods](https://github.com/makermods-robotics) Metal Arm (Damiao motors) | [MakerMods' LeRobot followers](https://github.com/makermods-robotics/lerobot) and [makermodslab](https://github.com/makermods-robotics/makermodslab) |
| MakerMods Maker Arm v1 (RobStride motors) | same |

Every station has one overhead camera and one camera on each wrist (USB webcams or Intel RealSense).

## For participants

Your station is ready: the arms, cameras, Quest and the Dream library are set up, and its config is `configs/station.yaml`. The station has no screen: you work on it over SSH from your laptop, and watch it in your laptop's browser. You do two things:

1. **Collect demonstrations** ([docs/data_collection.md](docs/data_collection.md)). Hold a controller's grip to drive that arm and pull its trigger to close its gripper. Press A to start and stop recording an episode, and B to throw one away.

   On your laptop (this line never runs on the station):

   ```bash
   ssh -L 8080:localhost:8080 <user>@<station>
   ```

   Put your station's login (from the organizers) in place of `<user>@<station>`, and wait for the station's prompt before going on. Then on the station, at that prompt:

   ```bash
   tmux new -A -s collect                          # keeps the session running if your connection drops
   cd ~/physical-ai-hacks
   uv run dhk-collect --config configs/station.yaml --dry-run      # practice: the arms do not move
   uv run dhk-collect --config configs/station.yaml --dataset-dir ~/hackathon_data/myteam-cubes
   ```

   The episodes need no labels: Dream labels every dataset from its overhead camera. `--task "..."` adds a task text of your own, if you want one.

   While `dhk-collect` runs, open http://localhost:8080 on your laptop: the dashboard. It shows the three cameras as they are recorded, whether you are recording, both arms and the episodes you saved, which you can play back. `uv run dhk-dataset drop <folder> <episode>` removes one you do not want.

   Your episodes are saved on the station in the `--dataset-dir` folder, here `~/hackathon_data/myteam-cubes`, as a LeRobot v2.1 dataset: not in the repo, and not uploaded until you fine-tune. [Where your data is stored](docs/data_collection.md#where-your-data-is-stored) shows what is inside and how to copy it to your laptop.

2. **Fine-tune Dream-1 on them** ([docs/fine_tuning.md](docs/fine_tuning.md)). First save your team's API key on the station (the guide shows how). Then use `--arm yam`, `metal` or `maker`, whichever your station has:

   ```bash
   # Metal or Maker arms only: add the tool poses their models train on (writes ~/hackathon_data/myteam-cubes-eef)
   ~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/myteam-cubes --arm metal
   # Check, upload and fine-tune (on Metal and Maker arms, pass the -eef folder)
   ~/dream-env/bin/python tools/finetune.py ~/hackathon_data/myteam-cubes --arm yam --name myteam-cubes --suffix myteam-cubes
   ```

When your model is ready, [run it on the arms](docs/running_your_model.md).

### Plan your time

- **A fine-tuning job takes about 7 to 8 hours**, and may wait in a queue first. The first run of a new model then takes up to an hour to start. Queue your job early, ideally the evening of day one, and test on day two.
- **A job costs the price the Dream platform's Billing page shows** ($200 at the time of writing), held from the account's prepaid credits when it is queued. Check with the organizers before you queue one.
- **Tens of good demonstrations beat hundreds of rushed ones.** Plan for 50 or more episodes of one task, each 10 to 30 seconds. [docs/data_collection.md](docs/data_collection.md) has tips.
- **No time to fine-tune?** Dream-1's base model runs on these arms without any fine-tuning; see [docs/running_your_model.md](docs/running_your_model.md).

## Safety

These are real, strong arms. Keep the emergency stop (or the arms' power switch) within reach whenever they are powered, keep people out of their reach while they move, and practice with `--dry-run`, which never commands a motor. MakerMods arms have no brakes: start each session with them resting where they can go limp, because that is where they return and are released at the end.

## For organizers: setting up a station

[docs/hardware_setup.md](docs/hardware_setup.md) takes a station from parts on a table to a working dry run: the computer, the arms, the cameras, the Quest and the Dream library.

Claude Code can walk you through it with the repo's `station-setup` skill: start `claude` in the repo and run `/station-setup`. If you have added Dream's documentation server to Claude Code (the Dream platform's Setup API page offers it), Claude reads the Dream docs through it as well. Claude runs the checks and fills in the config with you. You do everything physical, and everything that powers or moves an arm (`CLAUDE.md` holds its rules).

No arms yet? On a YAM config, `uv run dhk-collect --config configs/yam.yaml --sim` drives i2rt's simulated arms with the Quest and records blank camera frames, so you can try the whole flow.

## Repository layout

| Path | What it is |
| --- | --- |
| `src/dream_hackathon/collect.py` | The teleop-and-record loop (`dhk-collect`) |
| `src/dream_hackathon/arms/` | The YAM and MakerMods arm adapters over the vendors' SDKs |
| `src/dream_hackathon/quest.py`, `teleop.py`, `ik.py` | Quest input, the clutched hand-to-tool mapping, and the teleop IK |
| `src/dream_hackathon/cameras.py`, `dataset.py` | Camera capture and the LeRobot v2.1 writer |
| `src/dream_hackathon/dashboard.py`, `dataset_cli.py` | The browser dashboard (over SSH) and `dhk-dataset` (list and drop episodes) |
| `src/dream_hackathon/station_lock.py` | One program drives the arms at a time (shared with `dreamcli`) |
| `src/dream_hackathon/embodiments/` | The Dream embodiment configs your fine-tunes are registered with |
| `configs/` | One template station config per arm type; a station's own is `configs/station.yaml` (not committed) |
| `tools/` | Dream API steps, run in the Dream library's environment: `add_eef.py`, `finetune.py` |
| `scripts/` | Device checks and the URDF download |
| `docs/` | Collection, fine-tuning, running and setup guides |
| `CLAUDE.md`, `.claude/skills/` | Claude Code's rules for this repo and the `station-setup` skill |

Run the tests with `uv sync --group dream && uv run pytest` (no hardware needed).

## Credits and licenses

This kit is licensed under the Apache License 2.0 (see `LICENSE`). It builds on [i2rt](https://github.com/i2rt-robotics/i2rt) (YAM SDK), [makermodslab](https://github.com/makermods-robotics/makermodslab) and [MakerMods' LeRobot fork](https://github.com/makermods-robotics/lerobot) (Metal and Maker arms, Apache-2.0), [oculus_reader](https://github.com/rail-berkeley/oculus_reader) (Quest app and reader, Apache-2.0), [mink](https://github.com/kevinzakka/mink) and [MuJoCo](https://github.com/google-deepmind/mujoco) (IK), and the [Dream library](https://pypi.org/project/dreamlabs/) (Apache-2.0). Each keeps its own license.
