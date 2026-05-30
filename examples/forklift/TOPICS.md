# Forklift policy — ROS 2 topic mapping

Every field that the model consumes or produces is sourced from / published to a
specific ROS 2 topic. This document is the authoritative reference for both the
training-data conversion (`convert_stage3_to_lerobot.py`) and the real-robot
inference node (`run_inference_ros2.py`).

All topics below are ROS 2. The bag files in `2025_extracted_pallet_handling/stage_3/`
were recorded with `rosbag2` v5 (mcap) and use `*/msg/*` message namespaces.

---

## Inputs to the model

### `base_0_rgb` — top-mounted ZED2i color image

| | |
|---|---|
| **LeRobot feature** | `image` |
| **ROS topic** | `/zed2i_top/zed2i/warped/left/image_rect_color/compressed` |
| **Message type** | `sensor_msgs/msg/CompressedImage` |
| **Field used** | `data` (JPEG-encoded BGR8) |
| **Preprocessing** | `cv2.imdecode` → BGR uint8 (720×1280×3) → `cv2.resize` to **224×224** with `INTER_AREA` → BGR-to-RGB → contiguous uint8 |
| **Native rate** | ~5 Hz |

### `left_wrist_0_rgb`, `right_wrist_0_rgb`

| | |
|---|---|
| **ROS topic** | none — not present in stage_3 |
| **Provided to model as** | `np.zeros_like(base_image)` |
| **Image mask** | `False` for pi0.5 (`True` for pi0-FAST) — i.e. the model attends to the empty slots only when running pi0-FAST |

### `state` — 6-dimensional proprioception

All 6 dims come from the **same** `/joint_states` message via name lookup in
`name[]`. Values are read as `position[i]` or `velocity[i]` where `i = name.index(<joint>)`.

| Index | Name | ROS topic | Message type | Computation |
|---|---|---|---|---|
| 0 | `lift`               | `/joint_states` | `sensor_msgs/msg/JointState` | `position[name.index("lift_lift_fixed")] + position[name.index("fork_plate_lift")]` (sum of the two-stage mast) |
| 1 | `shift`              | `/joint_states` | `sensor_msgs/msg/JointState` | `position[name.index("fork_side_shift")]` |
| 2 | `steering_angle`     | `/joint_states` | `sensor_msgs/msg/JointState` | `position[name.index("rear_linkage_part2")]` |
| 3 | `steering_angle_rate`| `/joint_states` | `sensor_msgs/msg/JointState` | `velocity[name.index("rear_linkage_part2")]` |
| 4 | `wheel_velocity`     | `/joint_states` | `sensor_msgs/msg/JointState` | `mean(velocity of front_left_motor, front_right_motor, rear_left_motor, rear_right_motor) * (2.07345 / (2π))` — converts rad/s to body m/s using the wheel circumference 2.07345 m |
| 5 | `tilting_angle`      | `/joint_states` | `sensor_msgs/msg/JointState` | `position[name.index("linkage_part2_linkage")]` |

The `name[]` array of `/joint_states` is stable within a session, so both the
converter and the inference node build the lookup once off the first message
and cache it.

### `prompt` — language instruction

A single string passed to the tokenizer. **Not read from a topic at inference
time** — the operator passes it on the command line. During data conversion the
prompt was derived from the directory path:

| `(scenario, operation)` | Prompt string |
|---|---|
| `g_g`, `EnterPallet` | `"Engage the forks under the pallet on the ground"` |
| `g_g`, `EnterSlot`   | `"Place the loaded pallet into a slot on the ground"` |
| `g_t`, `EnterPallet` | `"Engage the forks under the pallet on the ground"` |
| `g_t`, `EnterSlot`   | `"Place the loaded pallet into a slot on the truck"` |
| `t_g`, `EnterPallet` | `"Engage the forks under the pallet on the truck"` |
| `t_g`, `EnterSlot`   | `"Place the loaded pallet into a slot on the ground"` |

The `/current_task` topic (`std_msgs/msg/String`) was used during training only as a
sanity check that the directory name matched the runtime tag — its content is not
fed to the model.

### Stored in the dataset but not currently fed to the model — `pallet_delta`

Kept for ablation experiments. If you decide to wire it in, the source is:

| Index | Name | ROS topic | Message type | Field path |
|---|---|---|---|---|
| 0 | `dx`   | `/pallet_slot_pose_info` | `crayler_nav_msgs/msg/PalletSlotPoseInfo` | `pallet_to_fork_mid.pose.position.x` (EnterPallet) **or** `slot_to_fork_mid.pose.position.x` (EnterSlot) |
| 1 | `dy`   | `/pallet_slot_pose_info` | `crayler_nav_msgs/msg/PalletSlotPoseInfo` | `pallet_to_fork_mid.pose.position.y` (EnterPallet) **or** `slot_to_fork_mid.pose.position.y` (EnterSlot) |
| 2 | `dyaw` | `/pallet_slot_pose_info` | `crayler_nav_msgs/msg/PalletSlotPoseInfo` | yaw extracted from `*.pose.orientation` (quaternion → ZYX yaw) |

`pallet_delta_valid` is `True` once the chosen `pallet_to_fork_*` / `slot_to_fork_*`
sub-message has a non-zero header stamp; otherwise the row is zero-filled with
`pallet_delta_valid = False`.

---

## Outputs from the model

The model emits a chunk of length `action_horizon=10` (10 future timesteps),
each timestep a 5-dimensional `float32` action. The runtime publishes these
actions one at a time at 10 Hz on **`/joint_commands`** as a single
`hopper_msgs/msg/Commands` message.

### Where each action dim is sourced during training

All five action dims come from `/crayler/controls_stamped`
(`hopper_msgs/msg/ControlsStamped`), reading the **`x_d`** (raw reference setpoint)
field of each per-axis `PlcPidState`. Note that STEER is read from the
`steering_rate` PID (a velocity setpoint), **not** the `steering` PID (a position
setpoint) — the platform's classical controller commands steering as a rate.

| Index | Axis  | Training source field |
|---|---|---|
| 0 | DRIVE | `controls.driving.x_d` |
| 1 | STEER | `controls.steering_rate.x_d` |
| 2 | LIFT  | `controls.lifting.x_d` |
| 3 | SHIFT | `controls.shifting.x_d` |
| 4 | TILT  | `controls.tilting.x_d` |

### Where each action dim goes at inference time

At inference, the 5-dim action gets packed into a `hopper_msgs/Commands` message
on `/joint_commands` as follows. The `active_control` byte tells the lower-level
controller which channel (POS / VEL / EFF / FF) to interpret per axis:

| Index | Axis  | `Commands` field set | `active_control[i]` value | What the value represents |
|---|---|---|---|---|
| 0 | DRIVE | `velocity_commands[0]` | `2` (`VEL`) | Forward driving velocity command |
| 1 | STEER | `velocity_commands[1]` | `2` (`VEL`) | Steering rate setpoint, rad/s |
| 2 | LIFT  | `position_commands[2]` | `1` (`POS`) | Lift-mast position setpoint |
| 3 | SHIFT | `position_commands[3]` | `1` (`POS`) | Side-shift position setpoint |
| 4 | TILT  | `position_commands[4]` | `1` (`POS`) | Tilt-mast position setpoint |

So the published byte is `active_control = b"\x02\x02\x01\x01\x01"`.

All other `Commands` fields are zero-filled by the inference node:

| Field | Value |
|---|---|
| `header.stamp` | wall clock at publish time (`rclpy.clock.Clock().now()`) |
| `header.frame_id` | `"front"` (matches the bag) |
| `position_commands[0]`, `position_commands[1]` | `0.0` |
| `velocity_commands[2..4]`                       | `0.0` |
| `effort_commands[0..4]`, `ff_commands[0..4]`    | `0.0` |
| `is_stop`, `is_joy_control`                     | `False` |

The `Commands.msg` enumerations referenced above are defined in `hopper_msgs`:

```
# command order
uint8 DRIVE = 0
uint8 STEER = 1
uint8 LIFT  = 2
uint8 SHIFT = 3
uint8 TILT  = 4

# active control definition
uint8 OFF = 0
uint8 POS = 1
uint8 VEL = 2
uint8 EFF = 3
uint8 FF  = 4
```

---

## Quick reference

```
Training conversion only (read from mcap bags):
  /crayler/controls_stamped              hopper_msgs/ControlsStamped       <- action targets
  /pallet_slot_pose_info                 crayler_nav_msgs/PalletSlotPoseInfo  <- pallet_delta (optional)
  /current_task                          std_msgs/String                   <- task-string sanity check

Subscriptions at inference (10 Hz inference loop):
  /zed2i_top/zed2i/warped/left/image_rect_color/compressed   sensor_msgs/CompressedImage
  /joint_states                                              sensor_msgs/JointState

Publication at inference (10 Hz, one /joint_commands message per tick):
  /joint_commands                                            hopper_msgs/Commands
```
