# Fine-tuning Dream-1 on your demonstrations

Fine-tuning trains a copy of Dream-1 on your dataset through the Dream API. The result is a model of your own, with an id like `dream-1-ft:a1b2c3d4-my-task`, that you then run on the arms. `tools/finetune.py` runs every step of Dream's [fine-tuning tutorial](https://platform.dreamlabs.ai/api-documentation/fine-tuning/) in one command; the tutorial explains each step in depth.

## Before you start

- **The Dream library** installed in `~/dream-env`, with your API key saved: the two steps on the [Getting started page](https://platform.dreamlabs.ai/api-documentation/). The tools below run there, not in this repo's environment.
- **Credits** on the account for the job: the [Billing](https://platform.dreamlabs.ai/billing) page shows the price and your balance ($200 per job at the time of writing).
- **Time**: about 7 to 8 hours of training, plus any wait in the queue, and up to an hour for your model's first start.

## 1. Metal and Maker arms: add the tool poses

Dream's models for MakerMods arms train on each gripper's pose beside the joints. `tools/add_eef.py` computes those poses with Dream's own model of your arm, the same one `dreamcli` uses when it runs your model. It writes a new dataset next to yours, `<name>-eef`, and checks it:

```bash
~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/my-task --arm metal    # or --arm maker
```

YAM datasets need no extra step: their fine-tunes train on the joints.

## 2. Upload and fine-tune

```bash
# YAM
~/dream-env/bin/python tools/finetune.py ~/hackathon_data/my-task --arm yam --name my-task-1 --suffix my-task
# Metal or Maker arm: the -eef dataset
~/dream-env/bin/python tools/finetune.py ~/hackathon_data/my-task-eef --arm metal --name my-task-1 --suffix my-task
```

- `--name` is the dataset's name on the platform, and `--suffix` the end of your model's id. Both use lowercase letters, digits and hyphens, and each must be new on your account.
- The first run registers your robot's embodiment, the description of what each recorded value means, as `hackathon-yam`, `hackathon-metal` or `hackathon-maker`, and later runs reuse it. `--embodiment` picks another name.
- `--lr-schedule cosine` tends to suit a model trained on a single task; `--learning-rate` takes 1e-6 to 1e-4 (the default is 1e-5).

The command checks the dataset on your computer first, then uploads it, waits for the platform to process it, queues the job and waits for your model, printing the status as it goes. It ends with your model's id and the command to try it.

You can stop it with Ctrl+C at any point once the upload is done. Pick up again with the id it printed:

```bash
~/dream-env/bin/python tools/finetune.py --arm yam --dataset-id ds_... --suffix my-task   # after the upload
~/dream-env/bin/python tools/finetune.py --arm yam --job-id ftj_...                       # after the job is queued
```

A queued job runs to its end and cannot be cancelled. If training fails, the job ends with no model and costs nothing; the error says why.

## Several sessions, several tasks

Every session you collect into the same folder becomes one dataset, and one fine-tune can learn several tasks if each episode carries its own task text. To train on two folders together, put both inside one parent folder and upload the parent: the platform reads each subfolder as one session. On Metal and Maker arms, run `tools/add_eef.py` on each folder first and put the `-eef` folders in the parent.

Next: [run your model](running_your_model.md).
