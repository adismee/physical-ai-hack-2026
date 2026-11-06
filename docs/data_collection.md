# Collecting demonstrations

Your station is already set up; this is where you start. You teach the model by showing it the task many times. Each time, from the start of the task to its end, is one episode. This guide covers the controls, a recording session, where your data is stored, and what makes demonstrations a model learns well from.

## Quick start

1. **On your laptop**, connect to the station with the dashboard's port forwarded. Run this on your laptop only: run on the station, it takes the dashboard's port there and your laptop gets nothing.

   ```bash
   ssh -L 8080:localhost:8080 <user>@<station>
   ```

   Replace `<user>@<station>` with your station's login, which the organizers give you (for example `team3@10.0.0.13`): typed as is, the shell reads `<` and `>` as redirections and says `parse error near '\n'`. Run this line on its own and wait for the station's prompt (`<user>@<station-name>:~$`) before typing anything else; it may ask for the station's password first.

2. **On the station**, at that prompt (tmux is on the station; your laptop does not need it), start a tmux session (or rejoin yours), so a dropped connection does not end it:

   ```bash
   tmux new -A -s collect
   cd ~/physical-ai-hacks
   ```

3. With the arms in their start pose, put on the Quest and practice once with the arms still (Ctrl+C ends it):

   ```bash
   uv run dhk-collect --config configs/station.yaml --dry-run
   ```

   While it runs, open http://localhost:8080 in your laptop's browser: the dashboard shows the cameras, the arms and your episodes. The page is there only while `dhk-collect` runs; before that, the browser cannot connect.

4. Start the session, with your team's own dataset folder:

   ```bash
   uv run dhk-collect --config configs/station.yaml --dataset-dir ~/hackathon_data/myteam-cubes
   ```

   No task text or other labels are needed: Dream's service labels every dataset from its overhead camera when it processes it. To name the task yourself anyway, add `--task "put the red cube in the bowl"`.

5. For each episode: set the scene, press A, do the task, press A again to save it (B throws it away). Press Ctrl+C when you are done.

Your episodes are saved in `~/hackathon_data/myteam-cubes` on the station ([where your data is stored](#where-your-data-is-stored)). The sections below explain each step.

## The controls

| Control | Does |
| --- | --- |
| Grip (middle finger), held | Drives that controller's arm: the gripper follows your hand |
| Grip released | The arm stops where it is; move your hand and grip again to keep going |
| Index trigger | That arm's gripper: released is open, pulled is closed |
| A | Starts recording an episode; pressed again, saves it |
| B | Throws away the episode being recorded |
| Right joystick click | Sends both arms back to where they were when the session started (not while recording) |
| Ctrl+C (in the terminal) | Ends the session; an episode still recording is thrown away and the arms return to their start |

The arm follows how your hand moves after you grip, not where your hand is. If you run out of reach, release, move your hand back, and grip again, like lifting a mouse off the table. The app reports each controller relative to the headset, so keep your head still and facing the arms while you grip.

## Working on a station over SSH

The station has no screen. Connect from your laptop with the dashboard's port forwarded. This line runs on your laptop, never on the station:

```bash
ssh -L 8080:localhost:8080 <user>@<station>
```

Replace `<user>@<station>` with your station's login, which the organizers give you (for example `team3@10.0.0.13`): typed as is, the shell reads `<` and `>` as redirections and says `parse error near '\n'`. Run this line on its own and wait for the station's prompt (`<user>@<station-name>:~$`) before typing anything else; it may ask for the station's password first. Then, on the station, start a tmux session, so that a dropped Wi-Fi connection does not end your session:

```bash
tmux new -A -s collect          # after a drop, reconnect from the laptop and run the same line to get back to it
cd ~/physical-ai-hacks
```

While `dhk-collect` runs, open http://localhost:8080 in your laptop's browser. The dashboard shows:

- The three cameras, cropped and sized as they are recorded.
- Whether an episode is recording, and for how long.
- Each arm: driving or holding, its tool position and its gripper.
- Warnings (a stalled camera, an arm that cannot follow) and the session's events.
- The episodes saved so far, with each camera's video to play back.

Tick "sound cues" and the laptop beeps when recording starts (one beep), when an episode is saved (two) or discarded (a low one). That lets a teammate at the laptop follow the operator. The page only watches: nothing on it moves the arms or changes your data.

If the page does not load:

- **"This site can't be reached"**: `dhk-collect` is not running (the page exists only while it runs), or your laptop's SSH connection was opened without `-L 8080:localhost:8080`. Reconnect from the laptop with it.
- **`dhk-collect` says `dashboard: port 8080 is not free`**: something on the station already holds the port, usually an `ssh -L 8080:...` that was run on the station itself. Type `exit` in that terminal to close it (the prompt stays the same: it is the same computer), or run on with `--dashboard-port 8081` and forward that port instead.

If the connection drops outside tmux, the session ends safely: the episode being recorded is discarded, the arms return to where they started and are released, and the rest of the output goes to `~/.cache/dream_hackathon/collect.log`. Only one program can drive the arms at a time. If someone else is using the station, `dhk-collect` says who, from where and since when.

## A recording session

1. Start with the arms resting in their start pose: on a MakerMods station, supported or folded, since they go limp there at the end.
2. Practice the controls without moving the arms. A dry run shows each arm's tool position once a second; hold a grip and watch it follow your hand:

   ```bash
   uv run dhk-collect --config configs/station.yaml --dry-run
   ```

3. Start the session. Give your dataset a folder of its own, named for your team and task:

   ```bash
   uv run dhk-collect --config configs/station.yaml --dataset-dir ~/hackathon_data/myteam-cubes
   ```

   You do not need to label the episodes: when Dream processes the dataset, its service writes dense text labels for it from the overhead camera. A task text of your own is optional: `--task "put the red cube in the bowl"` records it with each episode, so write it the way you will ask the robot. Without it, each episode records the neutral text `do the demonstrated task`, since the dataset format needs one.

4. For each episode: set the scene, press A, do the task, press A to save. If it went wrong, press B instead and do it again.
5. Press Ctrl+C when you are done. The arms return to where they started.

Every session adds its episodes to the same `--dataset-dir`, so you can stop and continue later, or switch operators. Use one dataset folder per task; if you name your tasks with `--task`, a different text in the same folder is fine too, when one model should learn several tasks. If anything does not respond (an arm, a camera, the Quest), call the organizers rather than changing the station's config.

Each saved episode is printed with its number. To check the dataset at any point against Dream's own checks, without uploading anything (see [fine-tuning](fine_tuning.md)):

```bash
# YAM
~/dream-env/bin/python tools/finetune.py ~/hackathon_data/myteam-cubes --arm yam --check
# Metal or Maker arm: add_eef checks the dataset it writes; --replace rebuilds it after you collect more
~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/myteam-cubes --arm metal --replace
```

## Removing episodes

B throws away the episode you are recording. To remove one you notice later, after watching it on the dashboard, list the episodes and drop it:

```bash
uv run dhk-dataset list ~/hackathon_data/myteam-cubes
uv run dhk-dataset drop ~/hackathon_data/myteam-cubes 3        # asks first; several numbers drop several
```

The episodes after it are renumbered, so check `list` again before dropping another. On a Metal or Maker arm, rebuild the dataset you fine-tune on afterwards with `tools/add_eef.py ... --replace`. Stop `dhk-collect` before dropping episodes from the folder it records into.

## Where your data is stored

Your data stays on the station, in the folder you pass with `--dataset-dir` (without it, the folder the config's `dataset_dir` names). It is not in the repo, and nothing is uploaded until you fine-tune. Each folder is one LeRobot v2.1 dataset, and every session adds to it:

```
~/hackathon_data/myteam-cubes/
  meta/
    info.json                       the dataset's shape: frame rate, cameras, state and action
    episodes.jsonl                  one line per episode: its number, task and length
    tasks.jsonl                     the task texts
    episodes_stats.jsonl            each episode's statistics, which Dream's checks read
  data/chunk-000/
    episode_000000.parquet          one row per frame: observation.state, action, timestamp, ...
    episode_000001.parquet
  videos/chunk-000/
    observation.images.overhead/episode_000000.mp4
    observation.images.wrist_l/episode_000000.mp4
    observation.images.wrist_r/episode_000000.mp4
```

On a Metal or Maker arm the video folders are `observation.images.scene`, `observation.images.left_wrist` and `observation.images.right_wrist`, and `tools/add_eef.py` writes the dataset you fine-tune on beside yours, as `~/hackathon_data/myteam-cubes-eef`.

Two more places on the station:

- `~/.cache/dream_hackathon/recording/` holds the episode being recorded. Saving it moves it into your dataset; B or Ctrl+C deletes it.
- `~/.cache/dream_hackathon/collect.log` holds a session's output after the SSH connection drops.

To keep a copy on your laptop, run this on the laptop:

```bash
scp -r <user>@<station>:hackathon_data/myteam-cubes .
```

## What the dataset holds

Each episode records 30 frames per second of:

- `observation.state`: both arms' measured joints (radians) and grippers.
- `action`: the joints and grippers commanded at that moment.
- One video per camera, cropped to 16:9 and recorded at 640x360.

On a YAM, the order is the right arm's six joints, its gripper (0 closed to 1 open), then the left arm's. On a Metal or Maker arm, the left arm comes first and the gripper is its opening in millimetres; `tools/add_eef.py` adds the tool poses before fine-tuning.

## Demonstrations that train well

- **Do the same task the same way.** The model copies what you show it. Pick one strategy (which arm grasps, from which side) and keep to it.
- **Vary the scene, not the method.** Move the objects to different positions and orientations across episodes, within the area the model will see when it runs.
- **Plan for 50 or more episodes** of each task, each 10 to 30 seconds. Every episode must be at least 1.6 seconds; a shorter one is not saved.
- **Start and end calmly.** Start recording with the arms still, and save shortly after the task is done.
- **Throw away mistakes.** A failed grasp you recover from is fine. An episode that did not do the task is better discarded with B.
- **Keep the cameras where they are.** Do not move a camera between episodes, or between collecting and running the model.
- **Move smoothly and not too fast.** The arms follow at a capped speed (`teleop.max_joint_speed_rad_s`); jerky hand motion makes jerky demonstrations.
