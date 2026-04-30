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
| **Image mask** | `False` for pi0.5 (`True` for pi0-FAST, matching `LiberoInputs`) — i.e. the model attends to the empty slots only when running pi0-FAST |

### `state` — 10-dimensional proprioception

The 10 dims are concatenated in this exact order, all `float32`:

| Index | Name | ROS topic | Message type | Field path |
|---|---|---|---|---|
| 0 | `lin_pot_left` | `/crayler/linears_stamped` | `hopper_msgs/msg/LinearsStamped` | `linears.lin_pot_left` |
| 1 | `lin_pot_right` | `/crayler/linears_stamped` | `hopper_msgs/msg/LinearsStamped` | `linears.lin_pot_right` |
| 2 | `lin_pot_vertical` | `/crayler/linears_stamped` | `hopper_msgs/msg/LinearsStamped` | `linears.lin_pot_vertical` |
| 3 | `length_hubmast` | `/crayler/linears_stamped` | `hopper_msgs/msg/LinearsStamped` | `linears.length_hubmast` (uint16 → float32) |
| 4 | `length_seitenhub` | `/crayler/linears_stamped` | `hopper_msgs/msg/LinearsStamped` | `linears.length_seitenhub` (uint16 → float32) |
| 5 | `steering_angle` | `/crayler/drive_stamped` | `hopper_msgs/msg/DriveStamped` | `drive.steering_angle` |
| 6 | `steering_angle_rate` | `/crayler/drive_stamped` | `hopper_msgs/msg/DriveStamped` | `drive.steering_angle_rate` |
| 7 | `front_left_speed` | `/crayler/drive_stamped` | `hopper_msgs/msg/DriveStamped` | `drive.front_left_speed` |
| 8 | `front_right_speed` | `/crayler/drive_stamped` | `hopper_msgs/msg/DriveStamped` | `drive.front_right_speed` |
| 9 | `rho` | `/crayler/drive_stamped` | `hopper_msgs/msg/DriveStamped` | `drive.rho` (central articulation pitch) |

Both topics publish at ~25 Hz natively.

### `prompt` — language instruction

A single string passed to the tokenizer. **Not read from a topic at inference
time** — the operator passes it on the command line. During data conversion the
prompt was derived from the directory path:

| `(scenario, operation)` | Prompt string |
|---|---|
| `g_g`, `EnterPallet` | `"Engage the forks under the pallet on the ground"` |
| `g_g`, `EnterSlot` | `"Place the loaded pallet into a slot on the ground"` |
| `g_t`, `EnterPallet` | `"Engage the forks under the pallet on the ground"` |
| `g_t`, `EnterSlot` | `"Place the loaded pallet into a slot on the truck"` |
| `t_g`, `EnterPallet` | `"Engage the forks under the pallet on the truck"` |
| `t_g`, `EnterSlot` | `"Place the loaded pallet into a slot on the ground"` |

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

Mapping from model output index → `Commands` message field, with the
`active_control` byte that the inference node sets on every published message
(this matches the publishing convention seen in stage_3):

| Index | Axis | `Commands` field set | `active_control[i]` value | What the value represents |
|---|---|---|---|---|
| 0 | DRIVE | `velocity_commands[0]` | `2` (`VEL`) | Forward driving velocity command |
| 1 | STEER | `position_commands[1]` | `1` (`POS`) | Steering angle setpoint, radians |
| 2 | LIFT  | `position_commands[2]` | `1` (`POS`) | Lift-mast position setpoint |
| 3 | SHIFT | `position_commands[3]` | `0` (`OFF`) | Side-shift — model output is ignored at publish time because the controller does not accept side-shift commands during the visual-servo phases that the policy was trained on. The slot is preserved in the action vector so future training data can populate it without changing the schema. |
| 4 | TILT  | `position_commands[4]` | `1` (`POS`) | Tilt-mast position setpoint |

All other `Commands` fields are zero-filled by the inference node:

| Field | Value |
|---|---|
| `header.stamp` | wall clock at publish time (`rclpy.clock.Clock().now()`) |
| `header.frame_id` | `"front"` (matches the bag) |
| `position_commands[0]`, `velocity_commands[1..4]` | `0.0` |
| `effort_commands[0..4]`, `ff_commands[0..4]` | `0.0` |
| `is_stop`, `is_joy_control` | `False` |

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
Subscriptions (10 Hz inference loop):
  /zed2i_top/zed2i/warped/left/image_rect_color/compressed   sensor_msgs/CompressedImage
  /crayler/linears_stamped                                   hopper_msgs/LinearsStamped
  /crayler/drive_stamped                                     hopper_msgs/DriveStamped

Optional subscription (recorded only, not fed to current model):
  /pallet_slot_pose_info                                     crayler_nav_msgs/PalletSlotPoseInfo

Publication (10 Hz, one /joint_commands message per tick):
  /joint_commands                                            hopper_msgs/Commands
```
