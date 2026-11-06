# Dream Hackathon Kit

Teach a pair of robot arms a task with a Meta Quest, then fine-tune Dream-1, the Dream Labs robot model, on your demonstrations and run it on the same arms.

This kit drives two follower arms from the Quest's controllers, records each demonstration as a [LeRobot v2.1](https://github.com/huggingface/lerobot) dataset (the format Dream fine-tuning reads), and hands the dataset to the [Dream API](https://platform.dreamlabs.ai/api-documentation/fine-tuning/). It supports three kinds of arms, each driven through its maker's own open-source SDK:

| Arms | Driven by | Cameras |
| --- | --- | --- |
| [i2rt YAM](https://i2rt.com/) | [i2rt's Python SDK](https://github.com/i2rt-robotics/i2rt) | three: one overhead, one on each wrist |
| [MakerMods](https://github.com/makermods-robotics) Metal Arm (Damiao motors) | [MakerMods' LeRobot followers](https://github.com/makermods-robotics/lerobot) and [makermodslab](https://github.com/makermods-robotics/makermodslab) | same |
| MakerMods Maker Arm v1 (RobStride motors) | same | same |

Cameras can be any USB webcam (UVC) or Intel RealSense, chosen per camera.

## How it works

1. **Collect** (this repo, on the station's computer). Hold a controller's grip to drive that arm and pull its trigger to close its gripper; press A to start and stop recording. Each episode is saved into one dataset folder.
2. **Fine-tune** (the Dream API). `tools/finetune.py` checks the dataset, uploads it and starts a fine-tuning job; for Metal and Maker arms, `tools/add_eef.py` first adds the tool poses their models train on.
3. **Run** (`dreamcli`, the Dream library's robot runner). `dreamcli robot run --checkpoint <your model>` runs your fine-tune on the arms.

## Quick start

On an Ubuntu computer (22.04 or 24.04) with the arms, cameras and Quest plugged in. [docs/hardware_setup.md](docs/hardware_setup.md) walks through every step; this is the short version.

```bash
# 1. Install (git-lfs fetches the Quest app; uv installs Python and the dependencies)
sudo apt install git git-lfs adb && git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/dreamlabs-co/dream-hackathon-kit.git && cd dream-hackathon-kit
uv sync --extra yam                  # a YAM station; or: --extra makermods (Metal or Maker arms)
                                     # add --extra realsense if any camera is a RealSense

# 2. Configure: copy the config for your arms and fill in its ports and cameras
cp configs/yam.yaml configs/my_station.yaml        # or metal.yaml / maker.yaml

# 3. Check each device on its own
uv run python scripts/check_arms.py --config configs/my_station.yaml
uv run python scripts/check_quest.py
uv run python scripts/check_cameras.py --config configs/my_station.yaml

# 4. Try the controls without moving the arms, then collect
uv run dhk-collect --config configs/my_station.yaml --dry-run
uv run dhk-collect --config configs/my_station.yaml --task "put the red cube in the bowl"
```

Then fine-tune ([docs/fine_tuning.md](docs/fine_tuning.md)) and run your model ([docs/running_your_model.md](docs/running_your_model.md)).

No arms yet? On a YAM config, `uv run dhk-collect --config configs/yam.yaml --sim` drives i2rt's simulated arms with the Quest and records blank camera frames, so you can try the whole flow.

## Planning your hackathon

- **A fine-tuning job takes about 7 to 8 hours**, and may wait in a queue first. The first run of a new model then takes up to an hour to start. Collect and queue your job early, ideally the evening of day one, and test on day two.
- **A job costs the price the Dream platform's Billing page shows** ($200 at the time of writing), held from your account's prepaid credits when it is queued.
- **Tens of good demonstrations beat hundreds of rushed ones.** Plan for 50 or more episodes of one task, each 10 to 30 seconds. [docs/data_collection.md](docs/data_collection.md) has tips.
- **No time to fine-tune?** Dream-1's base model runs on these arms without any fine-tuning (`dreamcli robot run --task "..."`); see [docs/running_your_model.md](docs/running_your_model.md).

## Safety

These are real, strong arms. Keep the emergency stop (or the arms' power switch) within reach whenever they are powered, keep people out of their reach while they move, and start with `--dry-run`, which never commands a motor. MakerMods arms have no brakes: start each session with them resting where they can go limp, because that is where they return and are released at the end.

## Repository layout

| Path | What it is |
| --- | --- |
| `src/dream_hackathon/collect.py` | The teleop-and-record loop (`dhk-collect`) |
| `src/dream_hackathon/arms/` | The YAM and MakerMods arm adapters over the vendors' SDKs |
| `src/dream_hackathon/quest.py`, `teleop.py`, `ik.py` | Quest input, the clutched hand-to-tool mapping, and the teleop IK |
| `src/dream_hackathon/cameras.py`, `dataset.py` | Camera capture and the LeRobot v2.1 writer |
| `src/dream_hackathon/embodiments/` | The Dream embodiment configs your fine-tunes are registered with |
| `configs/` | One station config per arm type |
| `tools/` | Dream API steps, run in the Dream library's environment: `add_eef.py`, `finetune.py` |
| `scripts/` | Device checks and the URDF download |
| `docs/` | Setup, collection, fine-tuning and running guides |

Run the tests with `uv sync --group dream && uv run pytest` (no hardware needed).

## Credits and licenses

This kit is licensed under the Apache License 2.0 (see `LICENSE`). It builds on [i2rt](https://github.com/i2rt-robotics/i2rt) (YAM SDK), [makermodslab](https://github.com/makermods-robotics/makermodslab) and [MakerMods' LeRobot fork](https://github.com/makermods-robotics/lerobot) (Metal and Maker arms, Apache-2.0), [oculus_reader](https://github.com/rail-berkeley/oculus_reader) (Quest app and reader, Apache-2.0), [mink](https://github.com/kevinzakka/mink) and [MuJoCo](https://github.com/google-deepmind/mujoco) (IK), and the [Dream library](https://pypi.org/project/dreamlabs/) (Apache-2.0). Each keeps its own license.
