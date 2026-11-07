# MakerMods Physical AI Hackathon — Research Notes

**Event:** MakerMods Physical AI Hackathon, Nov 7–8 2026
**Venue:** Founders Inc., Building C, 3rd Floor, 2 Marina Blvd, San Francisco
**Track:** Hill Climb (score-based, no pitching — eval score only)
**Our hardware:** MakerMods Metal Arms (Damiao motors)

---

## Schedule

### Saturday Nov 7 (Build Day)
- 9:00 AM: Doors + kickoff (tracks, rules, scoring)
- 9:10 AM: Quick-starts (arms, Dream-1, LiveKit, Dimensional)
- 9:40 AM: Caldera on stage
- 9:50 AM: Team formation + track declaration
- 10:00 AM: Hacking begins, station booking opens
- 12:30 PM: Lunch
- 3:00–6:00 PM: Workshop lecture series (hacking continues alongside)
  - 3:00 Jeremy, Dream Labs — Dream-1
  - 3:30 Kushil, Dream Labs — infrastructure
  - 4:00 David Chen, LiveKit
  - 4:30 Dimensional
  - 4:45 Enact
  - 5:15 Lucen Robotics
  - 5:30 Open Q&A
- 6:30 PM: Dinner
- 10:00 PM: Venue closes

### Sunday Nov 8 (Demo Day)
- 10:00 AM: Doors reopen, hacking continues
- 12:30 PM: Lunch
- **3:00 PM: All demo submissions due**
- 3:00 PM: Live demos + judging on stage
- 5:30 PM: Awards
- 7:00 PM: Wrap-up

---

## Hill Climb Track

One shared long-horizon task on Metal Arm stations. Score is purely eval-based — no pitching. Full challenge rules will be published before/at the event.

**The model stack:** collect demonstrations → fine-tune Dream-1 → build a harness around it (what to do next, success detection, error recovery). Two teams with the same model can get very different scores based on the harness.

**Key timeline constraint:** Fine-tuning takes 7–8h + up to 1h cold-start. Queue the job Saturday evening to have it ready Sunday morning.

---

## Sponsors & Partners

### MakerMods Robotics

Full-stack robotics company: affordable hackable arms + software for physical AI.

**Hardware we're on:**

| Product | Price | Specs |
|---|---|---|
| Metal Arm (pair) | $4,999 | 6-axis, Damiao motors, 3kg payload, ±0.1mm repeatability, 100mm jaw stroke, CAN 1Mbps |
| Maker Arm (pair) | $2,796 | 6-axis, RobStride motors, 1.5kg payload, 3D-printed frame, 80mm jaw stroke |
| SO-101 | $299 | 5+1 axes, beginner |

**Software stack:**
- `makermodslab` — browser-based policy loop: calibrate, teleoperate, record, curate, train, deploy, eval. Supports DAgger (live corrections during rollout feed back into dataset). Remote op over LiveKit.
- `maker-arm-sdk` — MIT-licensed Python SDK for direct CAN control. Five CLI tools: `doctor`, `scan`, `zero`, `check`, `teleop`.
- `makermods-inspect-robots` — open eval framework. Define a task once; run any LLM/VLA against any embodiment (real arms, Isaac Lab, mock). NumPy-only core.
- Their LeRobot fork is what the hackathon kit imports: `MetalFollower` and `MakerFollower` drive follower arms over CAN.

**Things to know:**
- Arms are **brakeless** — they go limp on power-off. Start each session with arms resting in a supported pose.
- TCP for stock gripper: `[0.135, 0.0, 0.0]` m from flange — do not guess or use another arm's value.
- Gripper calibration (`gripper_open_deg` / `gripper_closed_deg`) lives in `~/.dreamlabs/station.yaml` after `dreamcli setup`.

---

### Dream Labs

Builds the "robot brain" (AI policy), not hardware. Emerged from stealth Oct 2026, $90M seed.

**Dream-1** is a **World Action Model (WAM)**: jointly predicts video and actions. Outputs end-effector poses (Cartesian position + orientation + grip), not joint angles. The robot's IK solver converts poses to joint commands — that's what enables cross-embodiment generalization. Three cameras: one overhead/scene, one per wrist.

**Dream-1-flash** is a faster/cheaper variant.

**Pricing:**

| Service | Cost |
|---|---|
| Dream-1 inference | $3/hr |
| Dream-1-flash inference | $2/hr |
| Fine-tuning, 500 steps | $100 (~3–4h) |
| Fine-tuning, 1,000 steps (default) | $200 (~7–8h) |
| Fine-tuning, 2,000 steps | $400 (~14–16h) |

Cost is held from prepaid credits when the job is queued (not on completion). A queued job cannot be cancelled.

**Key Python API calls:**
```python
client.validate_dataset(path, embodiment_config)   # local, free, no key needed
client.upload_dataset(path, name, embodiment)       # returns ds_...
client.finetune(dataset_name, suffix, ...)          # returns ftj_..., queues paid job
client.get_finetuning_status(job_id)               # poll until model published
```

Fine-tuned model id: `dream-1-ft:<account>-<suffix>`

**Running models:**
```bash
# Base Dream-1 (no fine-tune)
dreamcli robot run --task "..." --dry-run

# Fine-tuned, Metal/Maker arms
dreamcli robot run --checkpoint dream-1-ft:<account>-suffix --task "..." --dry-run

# Fine-tuned, YAM (joint action space)
dreamcli robot run --checkpoint dream-1-ft:<account>-suffix --task "..." --action-space joint --dry-run
```

Always dry-run a new model first. First cold-start of a new model: up to 1h.

**Docs MCP server:** Available from the Dream platform's Setup API page — add it to Claude Code for in-context API docs.

---

### LiveKit

Real-time WebRTC infrastructure. Official co-organizer of this hackathon.

**Products relevant here:**
- **LiveKit Agents** — Python/Node SDK for AI agents in a LiveKit "room." Handles STT → LLM → TTS, turn detection, interruption, tool calls, multi-agent handoffs. Apache 2.0.
- **LiveKit Portal** — production-grade stack for robot teleoperation and remote inference. Key problem it solves: over the internet, camera frames and robot state arrive out of order. Portal reconstructs time-aligned observations so learned policies see a coherent snapshot. Has an SO-101 / LeRobot teleop recipe that's directly relevant to our arm stack.
- **LiveKit Cloud** — managed hosting (free tier covers hackathon scale); also self-hostable.

**For Hill Climb:** Portal is the right starting point for teleoperation over the internet. LiveKit credits are provided to participants.

Install: `pip install livekit-agents`

---

### Dimensional (DimensionalOS / dimos)

Python-native "agentic OS for physical space." No ROS required. Early-stage SF startup.

- PyPI: `dimos`; GitHub: `dimensionalOS/dimos`
- Architecture: agent-native, blueprint-based module composition
- **MCP server** — exposes robot tools to any LLM editor (Cursor, Claude Code)
- Supported hardware: Unitree Go2/G1 quadrupeds, XArm, AgileX Piper (not MakerMods Metal/Maker natively)
- Ruthwik Dasyam (Dimensional) is on the judge panel

**For Hill Climb:** Native arm support is XArm/Piper — writing a Metal/Maker adapter is non-trivial for a weekend. The MCP server interface is the fastest integration path if you want LLM-driven natural language control.

---

### Enact Intelligence

Swiss physical-AI startup, exited stealth Oct 2026 with $30M pre-seed (backed by Yann LeCun, Arthur Mensch/Mistral). Founded by ex-1X Robotics (Manuel Yves Galliker). Team from DeepMind, 1X, Amazon Robotics, Helsing, Physical Intelligence, ETH Zurich.

**Core tech:** Measures force directly inside robot motors — turning actuators into sensors. Gives robots proprioception/touch. This is a gap in the Metal/Maker arm setup. Not directly usable at the hackathon (no hardware integration), but interesting for the talks.

---

### Lucen Robotics

Affordable modular robot hardware, single shared actuator across all form factors.

| Product | Price |
|---|---|
| Lucen Arm (7-DoF) | $1,999 |
| Lucen Open Arm (dual-arm bimanual) | $5,499 |
| Lucen Quadruped | $9,000 |

Tool-free reconfigurable, same SDK regardless of morphology. Direct competitor/complement to MakerMods in the same price bracket. Aimed at policy learning research.

---

### Caldera

Unknown/stealth — no public robotics presence found. Had an early 9:40 AM slot (often reserved for a sponsor/partner intro). Likely pre-announcement.

---

## Data Collection Tips

### Controls (Quest)

| Control | Does |
|---|---|
| Grip (middle finger), held | Drives that arm — gripper follows your hand |
| Grip released | Arm holds position; re-grip to continue (like lifting a mouse) |
| Index trigger | Gripper: released = open, pulled = closed |
| A | Start/stop recording an episode; first press of session opens the arms |
| B | Discard the episode being recorded |
| Right joystick click | Returns both arms to start pose (not while recording) |
| Ctrl+C | Ends session; active episode is discarded, arms return to start |

**Critical:** The app tracks controllers *relative to the headset*. Turning your head while gripping moves the arm. Keep your head still and facing the arms while driving.

### Dashboard Tips

- **Sound cues:** tick "sound cues" — 1 beep = recording started, 2 beeps = saved, low beep = discarded. Lets a laptop teammate follow the operator without watching the screen.
- The dashboard shows each device's status and what to do if one is red. Fix buttons: "Bring ... up" (CAN link down), "Restart the Quest app" (controllers stopped), "Restart adb" (headset unauthorized).
- Nothing on the dashboard page moves the arms.

---

## Dream-1 Fine-Tuning Deep Dive

### Data Format: LeRobot v2.1

`dhk-collect` writes directly to this format.

```
~/hackathon_data/myteam-task/
  meta/
    info.json              # fps, cameras, state/action dims
    episodes.jsonl         # index, task, length per episode
    tasks.jsonl
    episodes_stats.jsonl   # Dream validator reads this
  data/chunk-000/
    episode_000000.parquet # one row per frame: state, action, timestamp
  videos/chunk-000/
    observation.images.scene/episode_000000.mp4       # 640x360, 16:9, 30fps
    observation.images.left_wrist/episode_000000.mp4
    observation.images.right_wrist/episode_000000.mp4
```

Camera names by arm type:
- **YAM:** `overhead`, `wrist_l`, `wrist_r`
- **Metal/Maker:** `scene`, `left_wrist`, `right_wrist`

State/action: 14 values — left arm ×6 joints + left gripper (mm) + right arm ×6 joints + right gripper (mm).

### The `add_eef.py` Step (Metal/Maker Arms Only)

Dream's MakerMods models predict EEF poses, not raw joints. YAM skips this step.

```bash
~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/myteam-task --arm metal
# writes ~/hackathon_data/myteam-task-eef
# re-run with --replace after adding more episodes
```

Converts 14-value joint dataset → 20-value EEF action vector (xyz + rot6 + grip per arm) + 34-value state (adds raw joints for proprioception). Uses `dream_ik.solver.ChunkIK` with the same TCP as `dreamcli` at inference. Runs Dream's dataset validator at the end.

### Fine-Tuning Parameters

```bash
~/dream-env/bin/python tools/finetune.py ~/hackathon_data/myteam-task-eef \
    --arm metal \
    --name myteam-task \
    --suffix myteam-task \
    --lr-schedule cosine    # recommended for single-task
    # --learning-rate 1e-5  # default, usually fine
```

| Flag | Effect |
|---|---|
| `--arm yam/metal/maker` | Required. Sets embodiment + EEF vs joint training |
| `--name` | Dataset name on platform (new, lowercase, hyphens ok) |
| `--suffix` | End of model id (new, 1–40 chars) |
| `--lr-schedule cosine` | Recommended for single-task fine-tunes |
| `--check` | Validate locally only — free, no upload, no API key |
| `--dataset-id ds_...` | Resume after upload is done |
| `--job-id ftj_...` | Resume after job is queued |

Pipeline: validate → register embodiment (once per account) → upload → wait for processing (~30s polls) → queue job → wait for model (~60s polls, prints loss).

### What Makes Good Training Data

- **50+ episodes per task**, each 10–30s. Episodes under 1.6s are rejected.
- **Consistent method:** pick one strategy (which arm leads, which approach direction) and repeat it. The model imitates exactly what you show.
- **Vary the scene, not the method:** move objects to different positions/orientations across episodes.
- **Discard failures aggressively:** B during collection, "Delete selected" in dashboard. A recovery is ok; a task-not-completed episode should go.
- **Start and end calmly:** begin recording with arms still; stop shortly after task completion.
- **Keep cameras fixed** between collection and evaluation — camera swaps are a silent failure mode.
- **Move smoothly:** jerky demos → jerky policy.
- **Task text matters:** use the same phrasing at inference as during collection (or the auto-label Dream applies). Phrasing variation drops success rate.

### Multi-Session / Multi-Task Training

Every session added to the same `--dataset-dir` becomes one dataset. To train on two separate folders together (e.g. two different tasks or two collection sessions in different folders), put both inside a parent folder and pass the parent to `finetune.py` — the platform reads each subfolder as one session. For Metal/Maker arms, run `add_eef.py` on each subfolder first and put the `-eef` folders in the parent.

### Managing Episodes

During collection: **B** discards the episode in progress. After the fact, use the dashboard's "Delete selected" (tick episodes, confirm; refused while recording). Without a running session:

```bash
uv run dhk-dataset list ~/hackathon_data/myteam-task
uv run dhk-dataset drop ~/hackathon_data/myteam-task 3   # asks first; renumbers remaining
```

Episodes renumber after each drop — re-run `list` before dropping another. After dropping from a Metal/Maker dataset, re-run `add_eef.py ... --replace`.

### Tips for Hill Climb Score

1. **Try base Dream-1 zero-shot first** — it scored 59.3% mean on RoboDojo-RC Tier 1, 95% on Stack Bowls. If the base model handles the task well enough, invest time in the harness instead of fine-tuning.

2. **Queue fine-tuning Saturday evening** — job runs on Dream's servers overnight. Cold-start of a new model after training: up to 1h.

3. **Use `--lr-schedule cosine`** for single-task fine-tunes.

4. **Check the dataset locally before uploading** (free): `tools/finetune.py ... --check`

5. **Watch training loss** — `finetune.py` prints it every minute. Loss staying high or spiking → quality issues (mixed strategies, bad episodes not discarded).

6. **Re-run `add_eef.py --replace`** every time you add more episodes to the Metal arm dataset.

7. **Design workspace away from joint limits** — precision tasks near limits are a documented Dream-1 weakness.

8. **The harness is the differentiator.** Two teams with the same fine-tuned model will score differently based on: what they do when an action fails, whether they detect success, whether they retry intelligently. This is where to invest time beyond data collection.

9. **Use Live View** (`dreamcli robot up`, or `dreamcli robot up joint` for YAM fine-tunes) to watch the cameras and start runs from the browser instead of the terminal.

### Relevant Local Files

| File | What it is |
|---|---|
| `docs/data_collection.md` | Collection guide + quality tips |
| `docs/fine_tuning.md` | Step-by-step fine-tuning guide |
| `docs/running_your_model.md` | Running fine-tuned vs base model |
| `tools/finetune.py` | Full upload + fine-tune pipeline |
| `tools/add_eef.py` | EEF pose conversion for Metal/Maker arms |
| `src/dream_hackathon/embodiments/makermods.json` | MakerMods vector layout |
| `configs/station.yaml` | Station config (not committed) |
