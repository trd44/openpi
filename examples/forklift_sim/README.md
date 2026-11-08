# Forklift simulation → LeRobot dataset

Converts the Gazebo simulation rosbags in `data/simulation/` (rosbag2 v5, mcap) into a
LeRobot v2.1 dataset. One bag directory = one episode covering a full mission
(FollowPath → EnterPallet → LiftPallet …), with a per-frame language task.

The core features (`image`, `state`, `actions`, `task`) keep the exact semantics of the
real stage_3 dataset from `examples/forklift/convert_stage3_to_lerobot.py`, so the sim
dataset works with the existing `LeRobotForkliftDataConfig` / `ForkliftInputs` training
pipeline unchanged — only the `repo_id` needs to point at the sim dataset.

## Usage

```bash
uv pip install mcap mcap-ros2-support opencv-python
uv run examples/forklift_sim/convert_sim_to_lerobot.py \
    --data_dir data/simulation \
    --repo_id local/forklift_sim
```

Useful flags: `--limit_episodes N` (smoke test), `--cameras top left right third_person`
(which virtual cameras to store; all four by default, `top` is mandatory).

## Topic mapping (sim vs. real)

| Feature | Real (stage_3) | Simulation |
|---|---|---|
| `image` | `/zed2i_top/.../compressed` (ZED2i top) | `/virtual_camera_top/image_raw/vis/compressed` |
| `left_image`, `right_image` | — (wrist slots zero-padded) | `/virtual_camera_{left,right}/image_raw/vis/compressed` — fork-level views, stored for future wrist-slot experiments (not yet consumed by training) |
| `third_person_image` | — | `/virtual_camera_third_person/image_raw/vis/compressed` — external chase view of the whole forklift (common VLA conditioning; not yet consumed by training) |
| `state` (6-d) | `/joint_states` | `/joint_states` — identical joint names, identical computation |
| `actions` (5-d) | `/crayler/controls_stamped` (`x_d` per axis) | `/joint_controller_states` references (see below) |
| `task` / prompt | derived from directory path, one per episode | `/current_task`, per frame |
| `pallet_delta` | `/pallet_slot_pose_info` | `/pallet_slot_pose_info` (same message type) |

All images are JPEG-decoded, resized to 224×224 (`INTER_AREA`), stored RGB uint8.
Everything is re-sampled to a uniform 10 Hz grid with zero-order hold, like the real
converter.

### `state` (6-d, from `/joint_states`)

| Index | Name | Computation |
|---|---|---|
| 0 | `lift` | `position[lift_lift_fixed] + position[fork_plate_lift]` (two-stage mast) |
| 1 | `shift` | `position[fork_side_shift]` |
| 2 | `steering_angle` | `position[rear_linkage_part2]` |
| 3 | `steering_angle_rate` | `velocity[rear_linkage_part2]` |
| 4 | `wheel_velocity` | mean velocity of the 4 motor joints × (2.07345 m / 2π) → body m/s |
| 5 | `tilting_angle` | `position[linkage_part2_linkage]` |

### Actions from `/joint_controller_states`

`data/simulation/MAPPING.md` documents how the real `Controls` fields map into
`JointControllerStates` (array order `[drive, steer, lift, shift, tilt]`, confirmed by
`msg.labels`). The 5-d `actions` vector reads:

| Index | Axis | Source | Why |
|---|---|---|---|
| 0 | DRIVE | `velocity_controller_states[0].reference` | The sim commands driving through the **velocity** controller (always active; the position slot MAPPING.md names for `driving` is never active and its reference stays 0). Same physical quantity as the real `driving.x_d` velocity setpoint. |
| 1 | STEER | `velocity_controller_states[1].reference` | Matches the real convention: STEER is a rate command (`steering_rate.x_d`). |
| 2 | LIFT  | `position_controller_states[2].reference` | Position setpoint, as real. |
| 3 | SHIFT | `position_controller_states[3].reference` | Position setpoint, as real. |
| 4 | TILT  | `position_controller_states[4].reference` | Position setpoint, as real. |

### Extra features kept for future training

- `controls_reference`, `controls_reference_filt`, `controls_feedback`,
  `controls_error`, `controls_output`, `controls_output_ff`, `controls_output_fb`,
  `controls_d_error_dt` — each `(6,)` float32, ordered
  `[driving, steering, steering_rate, lifting, shifting, tilting]` (the six named
  controls of MAPPING.md; `steering` is the **steering-angle** position setpoint).
  `controls_is_active` `(6,)` bool carries the per-controller active flags.
- `pallet_delta` `(3,)` `[dx, dy, dyaw]` + `pallet_delta_valid` — pose of the active
  target (slot for EnterSlot/ExitSlot/LowerPallet frames, pallet otherwise) in the
  fork-mid frame, from `/pallet_slot_pose_info`.
- `odom_gt` `(6,)` `[x, y, yaw, vx, vy, wyaw]` — ground-truth base pose/twist from
  `/odometry_gt` (sim-only luxury; useful for eval metrics or auxiliary losses).

### Task strings

`/current_task` tags map to prompts (EnterPallet/EnterSlot reuse the real dataset's
strings so sim and real share those prompt embeddings):

| Raw tag | Prompt |
|---|---|
| `FollowPath` | Drive to the pallet |
| `EnterPallet` | Engage the forks under the pallet on the ground |
| `LiftPallet` | Lift the pallet |
| `EnterSlot` | Place the loaded pallet into a slot on the ground |

Unknown tags are used verbatim with a warning.

## Reviewing episodes and editing language

No off-the-shelf LeRobot tool handles this dataset's combination of v2.1 format,
image (non-video) storage, and per-frame tasks — the official `lerobot-edit-dataset`
CLI targets v3.0 and can't edit language, and community editors (LERO,
lerobot-dataset-editor) assume MP4-video datasets with one task per episode. So this
folder ships a two-script review loop:

```bash
# 1. Render one MP4 per episode (all cameras tiled, task string burned in)
#    plus an index.html gallery:
uv run examples/forklift_sim/render_episode_videos.py --repo_id local/forklift_sim
open ~/.cache/huggingface/lerobot/local/forklift_sim_review/index.html

# 2. Export the per-segment task table:
uv run examples/forklift_sim/edit_tasks.py export --repo_id local/forklift_sim

# 3. Watch videos, fill the `new_task` column of task_edits.csv for any segment you
#    want to relabel (per-episode text is fine, e.g. "Engage the forks under the
#    left pallet"; blank = keep), then:
uv run examples/forklift_sim/edit_tasks.py apply --repo_id local/forklift_sim
```

`apply` rewrites `meta/tasks.jsonl`, `meta/episodes.jsonl`, `meta/info.json` and the
per-frame `task_index` columns in place, backing up the originals to
`<dataset>/task_edit_backup/` first. It is idempotent, and new task strings get new
indices so untouched episodes are unaffected.

## Timing quirks

Sensor topics in the sim bags carry **sim-time** header stamps (small values, ~100 s)
while mcap `log_time` is wall-clock epoch. The converter samples everything in sim
time; `/current_task` (headerless `std_msgs/String`) is remapped from log time to sim
time by interpolating over `/clock`.

Bags whose recorder was killed before finalizing (corrupt mcap footer, e.g.
`rosbag2_2026_07_02-09_24_22`) are read with the streaming mcap reader, which recovers
every message written before the truncation point.
