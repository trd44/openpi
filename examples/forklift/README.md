# Forklift (Crayler) — pi0.5 fine-tuning

Convert the `2025_extracted_pallet_handling/stage_3/` ROS2 mcap bags into a LeRobot v2.1
dataset and fine-tune `pi05_base` on it.

> **Other docs in this directory**
> - **`TOPICS.md`** — exhaustive mapping of every ROS 2 topic the model reads and
>   writes, with field paths and value conventions. The source of truth.
> - **`INFERENCE.md`** — operator-facing handoff guide for running the trained
>   model on the real robot via the policy server + a ROS 2 client.
> - **`run_inference_ros2.py`** — the rclpy inference node referenced by
>   `INFERENCE.md`.
> - **`eval_on_training_data.py`** — offline evaluator: replays the training
>   dataset through the policy server and reports where the model
>   systematically misestimates actions (see "Evaluating the fit" below).

## Trained checkpoints

These are registered by short name in [`forklift_models.py`](forklift_models.py), so
you can select any of them with `--model <name>` when serving (see "Serving a model"
and "Evaluating the fit" below). Add a new checkpoint there once and it's selectable
everywhere.

| Short name | Hugging Face repo | Train config |
|---|---|---|
| `lora` | https://huggingface.co/tduggan93/pi05-forklift-lora | `pi05_forklift_lora` |
| `full` | https://huggingface.co/tduggan93/pi05-forklift-full | `pi05_forklift` |
| `full-10000` | https://huggingface.co/tduggan93/pi05-forklift-full-10000 | `pi05_forklift` |
| `full-15000` | https://huggingface.co/tduggan93/pi05-forklift-full-15000 | `pi05_forklift` |
| `full-29999` | https://huggingface.co/tduggan93/pi05-forklift-full-29999 | `pi05_forklift` |

## Serving a model

`serve.py` resolves a short name (or a raw HF repo / `gs://` / local checkpoint dir),
downloads it from Hugging Face if needed, and starts the standard openpi websocket
policy server. The eval client and the ROS 2 inference node talk to it identically
regardless of which checkpoint is loaded.

```bash
uv run examples/forklift/serve.py --model full-15000        # by short name
uv run examples/forklift/serve.py --model full-10000
uv run examples/forklift/serve.py --list                    # show the registry

# a local checkpoint you just trained (give it the matching train config)
uv run examples/forklift/serve.py \
    --model checkpoints/pi05_forklift_lora/forklift_lora_v1/29999 \
    --config pi05_forklift_lora
```

(`scripts/serve_policy.py policy:checkpoint --policy.config=... --policy.dir=...` still
works for `gs://`/local dirs, but it can't fetch a bare HF repo id — that's exactly
what `serve.py` adds.)

## Dataset scope

`stage_3` contains the **closed-loop visual-servo engagement phases** of pallet handling:
- `EnterPallet` — drive forks under the pallet (with active lift servoing, `vary_z=true`)
- `EnterSlot` — drive the loaded pallet into the destination slot (`vary_z=false`)

The other phases that exist in the behavior tree (`ApproachPallet`, `ApproachSlot`,
`LiftPallet`) are pure path-following / drive-straight and were intentionally excluded
from the dataset. The LIFT axis is still actively commanded inside the `Enter*` phases
so it is kept in both state and action vectors.

Episode counts in stage_3:

| scenario | EnterPallet | EnterSlot |
|---|---|---|
| `g_g` (ground → ground) | 36 | 40 |
| `g_t` (ground → truck)  | 19 | 19 |
| `t_g` (truck → ground)  | 18 | 20 |

## Schema

| Feature | Shape | Source |
|---|---|---|
| `image` | (224, 224, 3) uint8 | `/zed2i_top/zed2i/warped/left/image_rect_color/compressed` |
| `state` | (6,) float32 | `/joint_states` (single topic, name-lookup) |
| `actions` | (5,) float32 | `/crayler/controls_stamped` (per-axis `x_d` reference) |
| `pallet_delta` | (3,) float32 | `/pallet_slot_pose_info` (`pallet_to_fork_mid` for EnterPallet, `slot_to_fork_mid` for EnterSlot) |
| `pallet_delta_valid` | (1,) bool | False until the synthetic topic publishes its first valid transform |
| `task` | str | path-derived language string |

`state` order: `[lift, shift, steering_angle, steering_angle_rate, wheel_velocity,
tilting_angle]` — all read from `/joint_states` by `name.index(...)` lookup.
`lift` sums two joints (`lift_lift_fixed + fork_plate_lift`); `wheel_velocity` is
the mean of the four motor `velocity` entries scaled by `2.07345 / (2π)` (rad/s →
body m/s using the wheel circumference).

`actions` order: `[drive, steer, lift, shift, tilt]`, all read from per-axis
`PlcPidState.x_d` in `/crayler/controls_stamped`. STEER specifically reads from
`steering_rate` (a velocity), **not** `steering` (a position) — the platform's
classical controller commands steering as a rate.

| axis | source `PlcPidState` | unit | observed range (sampled) |
|---|---|---|---|
| `drive` | `controls.driving.x_d`        | m/s   | -0.4 … 0.8 |
| `steer` | `controls.steering_rate.x_d`  | rad/s | -0.17 … 0.46 |
| `lift`  | `controls.lifting.x_d`        | m     | 0 … 1.82 |
| `shift` | `controls.shifting.x_d`       | m     | -0.08 … 0.08 |
| `tilt`  | `controls.tilting.x_d`        | rad   | -0.02 … 0.20 |

`pallet_delta` is stored in the dataset but is **not** wired into the model input by
default — it's there for ablations. Add `"observation/pallet_delta": "pallet_delta"`
to the repack and concatenate it into `state` inside `ForkliftInputs` if you want to
experiment with it.

Pi0.5 expects up to 3 image inputs (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`).
`stage_3` only has the top-mounted camera, so the wrist slots are zero-filled and
masked off in `ForkliftInputs`.

## Topic alignment (10 Hz, ZOH)

ROS2 topics here publish at mixed rates (~5 Hz for the camera, ~25 Hz for `/joint_states`
and `/crayler/controls_stamped`, ~24 Hz for `/pallet_slot_pose_info`). The converter:

1. Reads all messages and bins them per-topic by `header.stamp` (falling back to log time).
2. Picks an episode start time = the latest first-sample time across the three required
   topics (image, joint_states, controls_stamped), so every emitted frame has all
   features valid from frame 0.
3. Steps a uniform 10 Hz grid and pulls the **most recent message with `stamp <= t_k`**
   per topic (zero-order hold).
4. For `/pallet_slot_pose_info`, frames before its first publish carry `pallet_delta=0`
   with `pallet_delta_valid=False`.

## 1. Convert

Install the converter's extra deps once (the openpi env doesn't pull these by default):

```bash
uv pip install mcap mcap-ros2-support opencv-python
```

Smoke-test on a single episode:

```bash
uv run examples/forklift/convert_stage3_to_lerobot.py \
    --data_dir /media/tim/external-ssd/TIM/2025_extracted_pallet_handling/stage_3 \
    --repo_id local/forklift_stage3 \
    --limit_episodes 1
```

Then convert everything:

```bash
uv run examples/forklift/convert_stage3_to_lerobot.py \
    --data_dir /media/tim/external-ssd/TIM/2025_extracted_pallet_handling/stage_3 \
    --repo_id local/forklift_stage3
```

The dataset is written to `$HF_LEROBOT_HOME/local/forklift_stage3/` (defaults to
`~/.cache/huggingface/lerobot/local/forklift_stage3/`). Inspect
`meta/info.json` to verify `codebase_version: "v2.1"`, `fps: 10`, and the schema.

To push to HF Hub, add `--push_to_hub` and set `--repo_id <your_user>/forklift_stage3`.

## 1b. Sanity-check the dataset

```bash
uv run examples/forklift/inspect_dataset.py --repo_id local/forklift_stage3
```

Verifies `meta/info.json` (codebase v2.1, fps=10, schema), prints episode counts per
task, samples a random frame and dumps it as PNG, and reports per-dim min/max/mean/std
on `state` and `actions` plus the `pallet_delta_valid` fraction. Flags any state or
action dim that is constant across the sample (often a sign of a misnamed field) and
any episode that never has `pallet_delta_valid=True`.

## 2. Compute norm stats

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_forklift
```

This writes `assets/pi05_forklift/local--forklift_stage3/norm_stats.json`.

## 3. Fine-tune

Two configs are provided in `src/openpi/training/config.py`. Both load `pi05_base`
weights from `gs://openpi-assets/checkpoints/pi05_base/params`.

### Full finetune

```bash
uv run scripts/train.py pi05_forklift --exp-name=forklift_v1 --overwrite
```

Defaults: `batch_size=128`, `peak_lr=5e-5`, EMA on, 20k steps. Needs a multi-GPU node
(or a very beefy single GPU) for the full Pi0Config.

### LoRA finetune (single-GPU friendly)

```bash
uv run scripts/train.py pi05_forklift_lora --exp-name=forklift_lora_v1 --overwrite
```

Defaults: `batch_size=32`, `peak_lr=1e-4` cosine-decaying to `1e-5`, EMA off, 20k
steps. Only the LoRA adapter parameters on the PaliGemma backbone (`gemma_2b_lora`)
and the action expert (`gemma_300m_lora`) are unfrozen — everything else is frozen
via `freeze_filter`. This fits on a single 24-32 GB GPU.

For a 10-step smoke run before committing to a real run:

```bash
uv run scripts/train.py pi05_forklift_lora --exp-name=smoke --overwrite --num_train_steps=10
```

## Evaluating the fit (against the training data)

`eval_on_training_data.py` checks how well a fine-tuned checkpoint reproduces the
actions in the dataset it was trained on. It uses the **same policy-server +
websocket-client flow** as real inference (`scripts/serve_policy.py` + a client),
so what it measures is exactly what the robot would receive — only the "robot"
is the recorded dataset, replayed open-loop. At each step it feeds the recorded
observation to the server, gets the 10-step action chunk back, and compares it to
the recorded actions for those steps.

It reports, per action axis (`drive, steer_rate, lift, shift, tilt`):

- **bias** (mean signed error) — a consistent over/under-command, flagged when it
  exceeds 0.25σ of that axis;
- **MAE / RMSE / normalized RMSE** — magnitude of error (nRMSE < ~0.5 means clearly
  better than just predicting the mean; ~1.0 means no better);
- **Pearson r / R² / fit slope** — a slope far from 1.0 is a *scale* error (the model
  systematically under- or over-shoots), flagged separately from bias;
- **error vs. horizon step** — whether the chunk degrades toward its tail;
- **per-task** breakdown (which of the four prompts is fit worst);
- **worst episodes**, both a top-K list and anything beyond `mean + Nσ`, so you can
  pull those rosbags/videos.

Outputs land in `data/forklift/eval/`: `eval_report.json` (every number),
`per_episode.csv`, and diagnostic plots (pred-vs-gt scatter, residual histograms,
per-episode RMSE bar).

### Run it directly

```bash
# 1. Serve the model you want to evaluate (leave running).
uv run examples/forklift/serve.py --model full-15000

# 2. In another shell, replay the training data through it.
uv run examples/forklift/eval_on_training_data.py --repo-id tduggan93/forklift
```

The run is labelled by the served model automatically (picked up from the server
metadata), so results land in `data/forklift/eval/full-15000/`. To compare models,
serve each in turn and re-run the eval — each gets its own subfolder. Pass `--label`
to override the name.

Useful flags:
- `--replan-steps 1` — densest (10× slower) evaluation.
- `--max-episodes N` — quick look.
- `--prompt-override "..."` — probe prompt sensitivity.
- `--samples-per-obs N` — **best-of-N multimodality diagnostic.** pi0.5 is stochastic
  and the expert is multimodal, so single-sample L2-to-the-logged-action is pessimistic.
  With `N>1` the report adds a per-axis "1-sample vs best-of-N" comparison: a *large*
  gap means the model can produce the expert action but not on every draw (multimodal,
  fine); a *small* gap means it genuinely can't fit that axis. Costs N× inference —
  pair it with `--max-episodes`, e.g. `--samples-per-obs 8 --max-episodes 20`.

### Run it via docker compose (matches the inference container flow)

`compose.yml` brings up the policy server and the evaluator as two containers
talking over the host network — the same pattern as `examples/libero/compose.yml`:

```bash
MODEL=full-15000 EVAL_ARGS="--repo-id tduggan93/forklift" \
docker compose -f examples/forklift/compose.yml up --build
```

`MODEL` is one of the short names above (default `full-15000`); results land in
`./data/forklift/eval/<model>/`. Set `HF_TOKEN` in your environment first if the
checkpoint or dataset is private.

## Notes / known limits

- **One camera only.** The right ZED and BL0 stereo `camera_info` topics exist in the
  bags but no image data was recorded for them. If you later add `user_study_TE`, you
  will get a 2- or 3-camera variant and should mint a separate repo_id and config.
- **v2.1, not v3.** openpi pins lerobot to commit
  `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`, which uses the v2.1 on-disk format. The
  v3 format is not loadable by the pinned data loader.
- **`pallet_delta` not in model input.** Stored in the parquet for ablation; wire it
  into `ForkliftInputs` when ready.
- **Other datasets in the repo.** `user_study_TE` (human teleop, multi-camera) and
  `2026_02_24_statistical_measurements` (visualization-only image — not training-grade)
  are not converted by this script.
