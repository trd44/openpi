# Running the pi0.5 forklift policy on the real robot

This is a step-by-step guide for whoever ends up running the trained model on
the actual forklift. It assumes you have:

- a forklift PC with ROS 2 installed and the `crayler` workspace sourced (so
  `ros2 topic list` shows the `hopper_msgs` / `crayler_nav_msgs` topics this
  document references), and
- a separate machine with a GPU that can run the openpi policy server (or, in
  a pinch, the same machine if it has enough VRAM — see "Troubleshooting" at
  the end).

The two machines talk over a websocket, so the GPU can be anywhere on the
network as long as the forklift PC can reach it.

> **Schema reference.** `TOPICS.md` (next to this file) is the authoritative
> spec for which ROS 2 topics the model reads, which it writes, and exactly
> how each field is filled in. If anything in this guide contradicts that
> file, `TOPICS.md` wins.

---

## 0. Available trained checkpoints

| Variant | Hugging Face repo | openpi train config |
|---|---|---|
| LoRA finetune | https://huggingface.co/tduggan93/pi05-forklift-lora | `pi05_forklift_lora` |
| Full finetune | https://huggingface.co/tduggan93/pi05-forklift-full *(may not be uploaded yet — check the page)* | `pi05_forklift` |

Either checkpoint is loaded the same way. The LoRA one is smaller and fits on
~24 GB of VRAM; the full finetune wants more.

---

## 1. On the GPU machine — start the policy server

Clone openpi and set up its environment (one-time, ~5 min on a fast machine):

```bash
git clone https://github.com/Physical-Intelligence/openpi
cd openpi
uv sync                                  # installs the pinned environment
```

Log in to Hugging Face so you can pull the checkpoint:

```bash
uv run huggingface-cli login
# paste a "Read"-scope token from https://huggingface.co/settings/tokens
```

Start the server. Two flavors — pick the one that matches the checkpoint you
want to run:

```bash
# LoRA checkpoint
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_forklift_lora \
    --policy.dir=tduggan93/pi05-forklift-lora

# Full finetune checkpoint (if/when it's uploaded)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_forklift \
    --policy.dir=tduggan93/pi05-forklift-full
```

You should see a line like `Listening on 0.0.0.0:8000`. Leave it running.

If the GPU machine is remote, note its IP (e.g. `192.168.1.42`) — the forklift
PC will need it.

---

## 2. On the forklift PC — set up the inference node

Source your ROS 2 workspace as you normally would so that `hopper_msgs` and
`sensor_msgs` are importable from Python:

```bash
source /opt/ros/<your-distro>/setup.bash         # e.g. humble, iron, jazzy
source <your-crayler-workspace>/install/setup.bash
```

Install the small extra dependencies that the inference node needs.
**Do not** install the full openpi tree on the robot — only the lightweight
client package that openpi ships:

```bash
# from a checkout of this repo on the forklift PC, or copied over
pip install -e packages/openpi-client
pip install opencv-python numpy
```

`rclpy` comes from your ROS 2 install; everything else is plain Python.

Confirm the input topics are actually publishing right now (good sanity check
before you start the policy):

```bash
ros2 topic hz /zed2i_top/zed2i/warped/left/image_rect_color/compressed
ros2 topic hz /crayler/linears_stamped
ros2 topic hz /crayler/drive_stamped
```

If any of those is silent, the policy can't run — fix the upstream pipeline
first.

---

## 3. First run — dry run, no actuation

**Always do this first.** It connects to the policy server, runs the full
observation → inference → action pipeline at 10 Hz, and prints what it would
publish, but does **not** actually send commands to the robot:

```bash
python3 examples/forklift/run_inference_ros2.py \
    --host=<gpu-ip> --port=8000 \
    --task="Engage the forks under the pallet on the ground" \
    --dry-run
```

Expected output (every second, summarizing the previous 10 ticks):

```
[INFO] Connecting to policy server at ws://192.168.1.42:8000 ...
[INFO] Server metadata: {...}
[INFO] tick    10  action=[drv=+0.123 str=+0.005 lft=+0.040 shf=+0.000 tlt=+0.001] published=False
[INFO] tick    20  action=[drv=+0.118 str=+0.011 lft=+0.041 shf=+0.000 tlt=+0.001] published=False
...
```

Things to verify before you let it actuate:

1. The connection succeeds and you see at least 30+ ticks of healthy output.
2. `drv` is in the rough range you'd expect (the model was trained on driving
   velocities roughly in `[-0.6, 0.8]`).
3. `shf` is always `+0.000` — that axis is intentionally suppressed.
4. The robot is positioned similarly to how it was during data collection: a
   pallet visible in front of it, fork height/tilt in a sensible starting
   configuration.

---

## 4. Live run — robot moves

Same command, drop `--dry-run`:

```bash
python3 examples/forklift/run_inference_ros2.py \
    --host=<gpu-ip> --port=8000 \
    --task="Engage the forks under the pallet on the ground"
```

Have the e-stop in your hand. The first time you do this, run it for a few
seconds, kill the script with Ctrl-C, and check that the robot stopped cleanly.

Available task prompts (these are the ones the model was trained on; pick the
one that matches what you want it to do):

| Prompt | Scenario |
|---|---|
| `"Engage the forks under the pallet on the ground"` | pallet sitting on the ground |
| `"Engage the forks under the pallet on the truck"` | pallet sitting on a truck |
| `"Place the loaded pallet into a slot on the ground"` | pallet on forks, deposit on ground |
| `"Place the loaded pallet into a slot on the truck"` | pallet on forks, deposit on truck |

If you want to run a phase that wasn't in the training set (e.g.
"approach pallet from 5m away"), the model wasn't trained for that. The
training data only covers the **closed-loop visual-servo engagement phase**,
so start the robot already pointed at and within ~5 m of the target.

---

## 5. Stopping

Ctrl-C in the inference-node terminal. The node has no special shutdown
sequence — it stops publishing, and whatever the controller's standing policy
is when it stops receiving new commands takes over (typically a hold).

If you want belt-and-suspenders, publish a single zero-command before exit:

```bash
ros2 topic pub --once /joint_commands hopper_msgs/msg/Commands \
    "{position_commands: [0,0,0,0,0], velocity_commands: [0,0,0,0,0], active_control: '\\x00\\x00\\x00\\x00\\x00', is_stop: true}"
```

(Adjust the `is_stop` semantics for your controller if it uses a different
flag.)

---

## What the node actually publishes

See **`TOPICS.md`** for the full mapping. Short version: every 100 ms it
publishes one `hopper_msgs/Commands` on `/joint_commands` with

- `velocity_commands[0]` = drive velocity (model output 0)
- `position_commands[1]` = steer angle (model output 1)
- `position_commands[2]` = lift position (model output 2)
- `position_commands[3]` = 0 (shift is always OFF in stage_3)
- `position_commands[4]` = tilt position (model output 4)
- `active_control` = `\x02\x01\x01\x00\x01` (DRIVE=VEL, STEER=POS, LIFT=POS, SHIFT=OFF, TILT=POS)
- everything else zeroed out, `is_stop=False`, `is_joy_control=False`.

The model emits a 10-step action chunk (1 second of action at 10 Hz). The
node calls the server every 10 ticks and replays the chunk in between, so the
server only does inference at 1 Hz.

---

## Troubleshooting

**"Still waiting for server..."**, repeated forever.
The forklift PC can't reach the GPU host. Check `ping <gpu-ip>` and that port
8000 is open. If you need a different port, pass `--port=<x>` on both sides.

**"waiting for first message on each input topic"** appears once and never
clears.
One of the three input topics isn't publishing. Run the `ros2 topic hz`
checks from §2 again. If the image topic is publishing under a different
name on this robot (e.g. nodelet remapping), edit the `TOPIC_*` constants at
the top of `run_inference_ros2.py` *and* re-record / re-train if the data
were collected against a different topic name.

**`cv2.imdecode returned None`.**
The image stream is not JPEG-compressed BGR. Check
`ros2 topic echo --no-arr <image-topic> | grep format` — the model was
trained on `format: "bgra8; jpeg compressed bgr8"`. If your forklift uses a
different encoding, you need to convert on this side, or change the topic
the camera node publishes to.

**Actions are all near zero.**
Most common cause: the prompt doesn't match a trained one. Pi0.5 is sensitive
to phrasing. Try one of the four prompts from §4 verbatim. Second most
common cause: the camera view doesn't resemble the training distribution
(wrong angle, lens dirty, very different lighting). Compare a screenshot
against any frame from the dataset.

**Actions look reasonable but the robot doesn't move / moves wrong.**
The downstream controller might require a different `active_control` byte
than the one this node sends. `\x02\x01\x01\x00\x01` matches the convention
seen in the training bags; if your controller expects something different,
edit `ACTIVE_CONTROL` at the top of `run_inference_ros2.py`. Worth diffing
against a live `ros2 topic echo /joint_commands` recording from a successful
classical-control run.

**OOM when starting the server.**
Use the LoRA config (`pi05_forklift_lora`) — fits on a single 24 GB GPU. If
that still OOMs, drop batch_size further or switch to CPU at very low rate
just for sanity. See the GPU-side handoff notes for more.

**Different ROS 2 distro than the one the bags were recorded on.**
The recorded bags don't pin a distro — they only pin message-package names
(`hopper_msgs`, `crayler_nav_msgs`). Any ROS 2 distro that can build those
packages should work. The node uses only stable rclpy / sensor_msgs APIs.
