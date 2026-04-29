# Forklift (Crayler) — pi0.5 fine-tuning

Convert the `2025_extracted_pallet_handling/stage_3/` ROS2 mcap bags into a LeRobot v2.1
dataset and fine-tune `pi05_base` on it.

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
| `state` | (10,) float32 | `linears_stamped` + `drive_stamped` |
| `actions` | (5,) float32 | `/joint_commands` per-axis active channel |
| `pallet_delta` | (3,) float32 | `/pallet_slot_pose_info` (`pallet_to_fork_mid` for EnterPallet, `slot_to_fork_mid` for EnterSlot) |
| `pallet_delta_valid` | (1,) bool | False until the synthetic topic publishes its first valid transform |
| `task` | str | path-derived language string |

`state` order: `[lin_pot_left, lin_pot_right, lin_pot_vertical, length_hubmast,
length_seitenhub, steering_angle, steering_angle_rate, front_left_speed,
front_right_speed, rho]`.

`actions` order: `[drive, steer, lift, shift, tilt]`. Each value is the command on
whichever channel is active for that axis according to the `Commands.msg`
`active_control` flag (one of POS/VEL/EFF/FF; OFF → 0). In `stage_3` the active
mapping is:

| axis | active control in stage_3 | range observed |
|---|---|---|
| `drive` | VEL | -0.6 … 0.8 |
| `steer` | POS | -0.6 … 0.6 rad |
| `lift`  | POS | 0 … 1.7 |
| `shift` | OFF (always 0) | — |
| `tilt`  | POS | -0.02 … 0.19 rad |

`shift` is kept as a placeholder so a runtime that emits a 5-axis Commands msg has a
slot. If you ever record data where SHIFT is commanded, the converter will pick it
up automatically (it reads `active_control[i]` per frame).

`pallet_delta` is stored in the dataset but is **not** wired into the model input by
default — it's there for ablations. Add `"observation/pallet_delta": "pallet_delta"`
to the repack and concatenate it into `state` inside `ForkliftInputs` if you want to
experiment with it.

Pi0.5 expects up to 3 image inputs (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`).
`stage_3` only has the top-mounted camera, so the wrist slots are zero-filled and
masked off in `ForkliftInputs`.

## Topic alignment (10 Hz, ZOH)

ROS2 topics here publish at mixed rates (~5 Hz for the camera, ~25 Hz for control/state,
~24 Hz for `/pallet_slot_pose_info`, ~5 Hz for `/joint_commands`). The converter:

1. Reads all messages and bins them per-topic by `header.stamp` (falling back to log time).
2. Picks an episode start time = the latest first-sample time across the four required
   topics (image, linears, drive, joint_commands), so every emitted frame has all
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
