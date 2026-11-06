# Collecting demonstrations

You teach the model by showing it the task many times. Each time, from the start of the task to its end, is one episode. This guide covers the controls, a recording session, and what makes demonstrations a model learns well from.

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

## A recording session

1. Start with the arms resting in their start pose: on a MakerMods station, supported or folded, since they go limp there at the end.
2. Check the controls without moving the arms (see [hardware setup](hardware_setup.md), step 5):

   ```bash
   uv run dhk-collect --config configs/my_station.yaml --dry-run
   ```

3. Start the session. The task text is what the model learns to follow, so write it the way you will ask the robot:

   ```bash
   uv run dhk-collect --config configs/my_station.yaml --task "put the red cube in the bowl"
   ```

4. For each episode: set the scene, press A, do the task, press A to save. If it went wrong, press B instead and do it again.
5. Press Ctrl+C when you are done. The arms return to where they started.

Every session adds its episodes to the config's `dataset_dir`, so you can stop and continue later, or switch operators. Use one dataset folder per task; a different `--task` text in the same folder is fine too, when one model should learn several tasks.

Each saved episode is printed with its number. To check the dataset at any point, without uploading anything (in `~/dream-env`, see [fine-tuning](fine_tuning.md)):

```bash
~/dream-env/bin/python tools/finetune.py ~/hackathon_data/my-task --arm yam --check
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
