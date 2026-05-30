"""Convert stage_3 forklift demos (ROS2 mcap) to a LeRobot v2.1 dataset for pi0.5 finetuning.

Source layout (input):
    <data_dir>/{g_g,g_t,t_g}/{EnterPallet,EnterSlot}/measurement_NNN/measurement_NNN_0.mcap

Each measurement directory becomes one LeRobot episode. Topics are re-sampled to a
uniform 10 Hz grid (zero-order hold / last-known-value).

State (6-d) is read from /joint_states (single topic for proprio).
Action (5-d) is read from /crayler/controls_stamped, which captures the real-time
controller's setpoints including commands sent over shared memory that bypass
/joint_commands. STEER specifically uses ``steering_rate.x_d`` (velocity), matching
the platform's classical-control convention.

See ``TOPICS.md`` next to this file for the full field-by-field spec.

Install the extra deps before running:
    uv pip install mcap mcap-ros2-support opencv-python

Usage:
    uv run examples/forklift/convert_stage3_to_lerobot.py \
        --data_dir /media/tim/external-ssd/TIM/2025_extracted_pallet_handling/stage_3 \
        --repo_id local/forklift_stage3
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from typing import Any

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mcap_ros2.reader import read_ros2_messages
import numpy as np
import tyro


IMAGE_TOPIC = "/zed2i_top/zed2i/warped/left/image_rect_color/compressed"
JOINT_STATES_TOPIC = "/joint_states"
CONTROLS_TOPIC = "/crayler/controls_stamped"
PALLET_INFO_TOPIC = "/pallet_slot_pose_info"
CURRENT_TASK_TOPIC = "/current_task"

REQUIRED_TOPICS = {IMAGE_TOPIC, JOINT_STATES_TOPIC, CONTROLS_TOPIC}
OPTIONAL_TOPICS = {PALLET_INFO_TOPIC, CURRENT_TASK_TOPIC}
ALL_TOPICS = REQUIRED_TOPICS | OPTIONAL_TOPICS

IMAGE_HW = 224
STATE_DIM = 6
ACTION_DIM = 5
PALLET_DELTA_DIM = 3

# Joint names used for the state vector (read from sensor_msgs/JointState.name lookup).
_JOINT_LIFT_FIXED = "lift_lift_fixed"
_JOINT_FORK_PLATE_LIFT = "fork_plate_lift"
_JOINT_FORK_SIDE_SHIFT = "fork_side_shift"
_JOINT_REAR_LINKAGE_PART2 = "rear_linkage_part2"
_JOINT_LINKAGE_PART2_LINKAGE = "linkage_part2_linkage"
_JOINT_FRONT_LEFT_MOTOR = "front_left_motor"
_JOINT_FRONT_RIGHT_MOTOR = "front_right_motor"
_JOINT_REAR_LEFT_MOTOR = "rear_left_motor"
_JOINT_REAR_RIGHT_MOTOR = "rear_right_motor"

REQUIRED_JOINTS = (
    _JOINT_LIFT_FIXED,
    _JOINT_FORK_PLATE_LIFT,
    _JOINT_FORK_SIDE_SHIFT,
    _JOINT_REAR_LINKAGE_PART2,
    _JOINT_LINKAGE_PART2_LINKAGE,
    _JOINT_FRONT_LEFT_MOTOR,
    _JOINT_FRONT_RIGHT_MOTOR,
    _JOINT_REAR_LEFT_MOTOR,
    _JOINT_REAR_RIGHT_MOTOR,
)

# Wheel rad/s -> body m/s scaling. 2.07345 m is the wheel circumference; dividing by
# 2*pi converts radians-per-second to revolutions-per-second, then multiplying by the
# circumference gives meters-per-second along the driving direction.
_WHEEL_RADPS_TO_MPS = 2.07345 / (2.0 * math.pi)

# Path-derived language strings. Keys are (scenario, operation).
TASK_STRINGS: dict[tuple[str, str], str] = {
    ("g_g", "EnterPallet"): "Engage the forks under the pallet on the ground",
    ("g_g", "EnterSlot"): "Place the loaded pallet into a slot on the ground",
    ("g_t", "EnterPallet"): "Engage the forks under the pallet on the ground",
    ("g_t", "EnterSlot"): "Place the loaded pallet into a slot on the truck",
    ("t_g", "EnterPallet"): "Engage the forks under the pallet on the truck",
    ("t_g", "EnterSlot"): "Place the loaded pallet into a slot on the ground",
}


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _decode_image(msg: Any) -> np.ndarray:
    """Decode a sensor_msgs/CompressedImage (jpeg bgr8) and resize to IMAGE_HW square RGB uint8."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode returned None")
    bgr = cv2.resize(bgr, (IMAGE_HW, IMAGE_HW), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


@dataclass
class _JointIndex:
    """Cached name -> array-index lookup for sensor_msgs/JointState messages.

    The ``name`` array of a /joint_states message is stable within a bag, so once
    we build the lookup off the first message we can keep using it.
    """

    idx: dict[str, int]

    @classmethod
    def from_msg(cls, msg: Any) -> "_JointIndex":
        return cls(idx={n: i for i, n in enumerate(msg.name)})

    def lookup(self, name: str) -> int:
        try:
            return self.idx[name]
        except KeyError as e:
            raise RuntimeError(f"joint {name!r} not present in /joint_states.name") from e


def _state_from_joint_states(msg: Any, jx: _JointIndex) -> np.ndarray:
    """Build the 6-d state vector from a /joint_states message.

    Order:
      [0] lift                = position[lift_lift_fixed] + position[fork_plate_lift]
      [1] shift               = position[fork_side_shift]
      [2] steering_angle      = position[rear_linkage_part2]
      [3] steering_angle_rate = velocity[rear_linkage_part2]
      [4] wheel_velocity      = mean(velocity of 4 motor joints) * (circumference / 2π)
      [5] tilting_angle       = position[linkage_part2_linkage]
    """
    p = msg.position
    v = msg.velocity
    lift = p[jx.lookup(_JOINT_LIFT_FIXED)] + p[jx.lookup(_JOINT_FORK_PLATE_LIFT)]
    shift = p[jx.lookup(_JOINT_FORK_SIDE_SHIFT)]
    steering_angle = p[jx.lookup(_JOINT_REAR_LINKAGE_PART2)]
    steering_angle_rate = v[jx.lookup(_JOINT_REAR_LINKAGE_PART2)]
    wheel_v = (
        v[jx.lookup(_JOINT_FRONT_LEFT_MOTOR)]
        + v[jx.lookup(_JOINT_FRONT_RIGHT_MOTOR)]
        + v[jx.lookup(_JOINT_REAR_LEFT_MOTOR)]
        + v[jx.lookup(_JOINT_REAR_RIGHT_MOTOR)]
    ) * 0.25 * _WHEEL_RADPS_TO_MPS
    tilting_angle = p[jx.lookup(_JOINT_LINKAGE_PART2_LINKAGE)]
    return np.array(
        [lift, shift, steering_angle, steering_angle_rate, wheel_v, tilting_angle],
        dtype=np.float32,
    )


def _action_from_controls(controls_msg: Any) -> np.ndarray:
    """Per-axis action target from /crayler/controls_stamped using the ``x_d`` reference.

    Axis order: [DRIVE, STEER, LIFT, SHIFT, TILT].

    NOTE: STEER reads from ``steering_rate`` (velocity), not ``steering`` (position) —
    this matches the platform's classical-control convention. The forklift's runtime
    controls steering as a rate, and ``steering`` is rarely active.
    """
    c = controls_msg.controls if hasattr(controls_msg, "controls") else controls_msg
    return np.array(
        [
            c.driving.x_d,
            c.steering_rate.x_d,
            c.lifting.x_d,
            c.shifting.x_d,
            c.tilting.x_d,
        ],
        dtype=np.float32,
    )


def _pallet_delta_from_msg(msg: Any, operation: str) -> tuple[np.ndarray, bool]:
    """Extract [dx, dy, dyaw] from PalletSlotPoseInfo for the active target.

    For EnterPallet, we use ``pallet_to_fork_mid`` (pose of the pallet expressed in the
    fork-spacers/pickup frame). For EnterSlot, we use ``slot_to_fork_mid``. Returns
    (zeros, False) if the relevant transform is not yet populated (header.stamp == 0).
    """
    if operation == "EnterSlot":
        target = msg.slot_to_fork_mid
    else:
        target = msg.pallet_to_fork_mid
    valid = target.header.stamp.sec != 0 or target.header.stamp.nanosec != 0
    if not valid:
        return np.zeros(PALLET_DELTA_DIM, dtype=np.float32), False
    p = target.pose.position
    q = target.pose.orientation
    return np.array([p.x, p.y, _quat_to_yaw(q.x, q.y, q.z, q.w)], dtype=np.float32), True


@dataclass
class _TopicSeries:
    """Sorted (timestamp_ns, message) buffer with ffill lookup."""

    times: np.ndarray  # int64 ns
    msgs: list[Any]

    @classmethod
    def from_pairs(cls, pairs: list[tuple[int, Any]]) -> "_TopicSeries":
        if not pairs:
            return cls(times=np.empty(0, dtype=np.int64), msgs=[])
        pairs.sort(key=lambda x: x[0])
        times = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
        msgs = [p[1] for p in pairs]
        return cls(times=times, msgs=msgs)

    def latest_at(self, t_ns: int) -> Any | None:
        if self.times.size == 0:
            return None
        idx = int(np.searchsorted(self.times, t_ns, side="right")) - 1
        if idx < 0:
            return None
        return self.msgs[idx]

    @property
    def first_ns(self) -> int | None:
        return int(self.times[0]) if self.times.size else None


def _read_bag(bag_path: Path) -> dict[str, _TopicSeries]:
    buffers: dict[str, list[tuple[int, Any]]] = {t: [] for t in ALL_TOPICS}
    for m in read_ros2_messages(str(bag_path), topics=list(ALL_TOPICS)):
        topic = m.channel.topic
        msg = m.ros_msg
        header = getattr(msg, "header", None)
        if header is not None and (header.stamp.sec != 0 or header.stamp.nanosec != 0):
            ts = _stamp_ns(header.stamp)
        else:
            ts = m.log_time_ns
        buffers[topic].append((ts, msg))
    return {t: _TopicSeries.from_pairs(p) for t, p in buffers.items()}


def _find_bag(measurement_dir: Path) -> Path:
    candidates = sorted(measurement_dir.glob("*.mcap"))
    if not candidates:
        raise FileNotFoundError(f"no .mcap in {measurement_dir}")
    return candidates[0]


def _iter_measurements(data_dir: Path):
    for scenario_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        scenario = scenario_dir.name
        for op_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            operation = op_dir.name
            for meas_dir in sorted(p for p in op_dir.iterdir() if p.is_dir()):
                yield scenario, operation, meas_dir


def _build_frames(
    series: dict[str, _TopicSeries], operation: str, fps: int
) -> list[dict[str, Any]]:
    period_ns = 1_000_000_000 // fps

    required_first = []
    for t in REQUIRED_TOPICS:
        first = series[t].first_ns
        if first is None:
            return []
        required_first.append(first)
    start_ns = max(required_first)

    end_ns = min(int(series[t].times[-1]) for t in REQUIRED_TOPICS)
    if end_ns <= start_ns:
        return []

    # Cache joint name -> index from the first /joint_states message of the bag.
    js_first = series[JOINT_STATES_TOPIC].msgs[0]
    jx = _JointIndex.from_msg(js_first)
    missing = [n for n in REQUIRED_JOINTS if n not in jx.idx]
    if missing:
        print(f"  [skip] /joint_states is missing required joints: {missing}")
        return []

    frames: list[dict[str, Any]] = []
    t_ns = start_ns
    while t_ns <= end_ns:
        img_msg = series[IMAGE_TOPIC].latest_at(t_ns)
        js_msg = series[JOINT_STATES_TOPIC].latest_at(t_ns)
        ctrl_msg = series[CONTROLS_TOPIC].latest_at(t_ns)
        if img_msg is None or js_msg is None or ctrl_msg is None:
            t_ns += period_ns
            continue

        pallet_msg = series[PALLET_INFO_TOPIC].latest_at(t_ns)
        if pallet_msg is None:
            pallet_delta = np.zeros(PALLET_DELTA_DIM, dtype=np.float32)
            pallet_delta_valid = False
        else:
            pallet_delta, pallet_delta_valid = _pallet_delta_from_msg(pallet_msg, operation)

        frames.append(
            {
                "image": _decode_image(img_msg),
                "state": _state_from_joint_states(js_msg, jx),
                "actions": _action_from_controls(ctrl_msg),
                "pallet_delta": pallet_delta,
                "pallet_delta_valid": np.array([pallet_delta_valid], dtype=bool),
            }
        )
        t_ns += period_ns
    return frames


def _episode_task(scenario: str, operation: str, series: dict[str, _TopicSeries]) -> str:
    fallback = TASK_STRINGS.get((scenario, operation))
    if fallback is None:
        fallback = f"{scenario} {operation}"

    ct = series.get(CURRENT_TASK_TOPIC)
    if ct is not None and ct.times.size > 0:
        seen = {m.data for m in ct.msgs}
        if operation not in seen:
            print(
                f"  [warn] /current_task did not contain '{operation}' (saw: {seen}); "
                f"using fallback string"
            )
    return fallback


def _build_features() -> dict[str, dict[str, Any]]:
    return {
        "image": {
            "dtype": "image",
            "shape": (IMAGE_HW, IMAGE_HW, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [
                "lift",
                "shift",
                "steering_angle",
                "steering_angle_rate",
                "wheel_velocity",
                "tilting_angle",
            ],
        },
        "actions": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [
                "drive",  # /crayler/controls_stamped -> driving.x_d   (velocity)
                "steer",  # /crayler/controls_stamped -> steering_rate.x_d (velocity)
                "lift",   # /crayler/controls_stamped -> lifting.x_d   (position)
                "shift",  # /crayler/controls_stamped -> shifting.x_d  (position)
                "tilt",   # /crayler/controls_stamped -> tilting.x_d   (position)
            ],
        },
        "pallet_delta": {
            "dtype": "float32",
            "shape": (PALLET_DELTA_DIM,),
            "names": ["dx", "dy", "dyaw"],
        },
        "pallet_delta_valid": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["valid"],
        },
    }


def main(
    data_dir: str,
    repo_id: str = "local/forklift_stage3",
    *,
    fps: int = 10,
    push_to_hub: bool = False,
    private: bool = False,
    limit_episodes: int | None = None,
) -> None:
    src = Path(data_dir)
    out = HF_LEROBOT_HOME / repo_id
    if out.exists():
        shutil.rmtree(out)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="forklift",
        fps=fps,
        features=_build_features(),
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_episodes = 0
    for scenario, operation, meas_dir in _iter_measurements(src):
        if limit_episodes is not None and n_episodes >= limit_episodes:
            break
        try:
            bag_path = _find_bag(meas_dir)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue
        print(f"[{n_episodes:04d}] {scenario}/{operation}/{meas_dir.name}: reading {bag_path.name}")
        series = _read_bag(bag_path)
        frames = _build_frames(series, operation, fps)
        if not frames:
            print(f"  [skip] no aligned frames")
            continue
        task_str = _episode_task(scenario, operation, series)
        for f in frames:
            ds.add_frame({**f, "task": task_str})
        ds.save_episode()
        n_episodes += 1
        print(f"  -> {len(frames)} frames, task='{task_str}'")

    print(f"Done. Wrote {n_episodes} episodes to {out}.")

    if push_to_hub:
        ds.push_to_hub(
            tags=["forklift", "pi05", "ros2", "mcap"],
            private=private,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
