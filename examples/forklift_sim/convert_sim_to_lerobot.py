"""Convert simulation forklift rosbags (ROS2 mcap) to a LeRobot v2.1 dataset for pi0.5 finetuning.

Source layout (input):
    <data_dir>/rosbag2_YYYY_MM_DD-HH_MM_SS/rosbag2_YYYY_MM_DD-HH_MM_SS_0.mcap

Each bag directory becomes one LeRobot episode. Topics are re-sampled to a uniform
10 Hz grid (zero-order hold / last-known-value) in **simulation time**.

The core features (``image``, ``state``, ``actions``, per-frame ``task``) keep the
exact semantics of the real stage_3 dataset produced by
``examples/forklift/convert_stage3_to_lerobot.py``, so a sim dataset trains with the
same ``LeRobotForkliftDataConfig`` / ``ForkliftInputs`` pipeline unchanged.

Key differences from the real-robot bags (see README.md next to this file):
  - Images come from ``/virtual_camera_*/image_raw/vis/compressed`` instead of the
    ZED2i. ``top`` fills the ``image`` slot; ``left``/``right`` are stored as extra
    cameras for future wrist-slot experiments.
  - Actions come from ``/joint_controller_states`` (hopper_msgs/JointControllerStates)
    instead of ``/crayler/controls_stamped``. The Controls<->JointControllerStates
    field mapping is documented in ``data/simulation/MAPPING.md``. Empirically the sim
    commands DRIVE through the *velocity* controller (the position slot the mapping
    table names for ``driving`` is never active and its reference stays 0), so the
    ``drive`` action reads ``velocity_controller_states[DRIVE].reference`` — which is
    the same physical quantity (a velocity setpoint) as ``driving.x_d`` on the real
    robot.
  - Sensor headers carry sim time while ``/current_task`` (headerless) only has
    wall-clock log time; ``/clock`` is used to map it into sim time.
  - The task/prompt is per-frame, derived from ``/current_task`` (a bag spans a whole
    mission: FollowPath -> EnterPallet -> LiftPallet ...).

Install the extra deps before running:
    uv pip install mcap mcap-ros2-support opencv-python

Usage:
    uv run examples/forklift_sim/convert_sim_to_lerobot.py \
        --data_dir data/simulation \
        --repo_id local/forklift_sim
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from typing import Any, Literal

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import NonSeekingReader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
import tyro

JOINT_STATES_TOPIC = "/joint_states"
CONTROLLER_STATES_TOPIC = "/joint_controller_states"
CURRENT_TASK_TOPIC = "/current_task"
PALLET_INFO_TOPIC = "/pallet_slot_pose_info"
ODOMETRY_GT_TOPIC = "/odometry_gt"
CLOCK_TOPIC = "/clock"

CameraName = Literal["top", "third_person", "left", "right"]


def _camera_topic(cam: CameraName) -> str:
    return f"/virtual_camera_{cam}/image_raw/vis/compressed"


# Feature key each camera is stored under. "top" fills the real dataset's "image" slot.
CAMERA_FEATURE_KEYS: dict[CameraName, str] = {
    "top": "image",
    "left": "left_image",
    "right": "right_image",
    "third_person": "third_person_image",
}

IMAGE_HW = 224
STATE_DIM = 6
ACTION_DIM = 5
PALLET_DELTA_DIM = 3
ODOM_DIM = 6

# Axis order of JointControllerStates arrays (== msg.labels).
AXIS_DRIVE, AXIS_STEER, AXIS_LIFT, AXIS_SHIFT, AXIS_TILT = range(5)
EXPECTED_LABELS = ["drive", "steer", "lift", "shift", "tilt"]

# The six named controls of data/simulation/MAPPING.md, as (controller_array, axis).
# "position" -> position_controller_states, "velocity" -> velocity_controller_states.
MAPPING_CONTROLS: list[tuple[str, str, int]] = [
    ("driving", "position", AXIS_DRIVE),
    ("steering", "position", AXIS_STEER),
    ("steering_rate", "velocity", AXIS_STEER),
    ("lifting", "position", AXIS_LIFT),
    ("shifting", "position", AXIS_SHIFT),
    ("tilting", "position", AXIS_TILT),
]
CONTROL_NAMES = [name for name, _, _ in MAPPING_CONTROLS]
# PidState float subfields kept per control (MAPPING.md subfield table).
PID_FIELDS = (
    "reference",       # x_d
    "reference_filt",  # x_d_filt
    "feedback",        # x
    "error",           # e
    "output",          # u
    "output_ff",       # u_ff
    "output_fb",       # u_fb
    "d_error_dt",      # (no PlcPidState counterpart)
)

# Joint names for the state vector, identical to the real robot.
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

# Wheel rad/s -> body m/s scaling (wheel circumference 2.07345 m), same as real.
_WHEEL_RADPS_TO_MPS = 2.07345 / (2.0 * math.pi)

# /current_task raw tag -> language prompt. EnterPallet/EnterSlot strings match the
# real stage_3 dataset so the sim and real datasets share those task embeddings.
TASK_STRINGS: dict[str, str] = {
    "FollowPath": "Drive to the pallet",
    "EnterPallet": "Engage the forks under the pallet on the ground",
    "LiftPallet": "Lift the pallet",
    "ExitPallet": "Back the forks out from under the pallet",
    "EnterSlot": "Place the loaded pallet into a slot on the ground",
    "LowerPallet": "Lower the pallet",
    "ExitSlot": "Back the forks out of the slot",
}
# Raw tags whose frames target the slot instead of the pallet (for pallet_delta).
SLOT_TASKS = {"EnterSlot", "ExitSlot", "LowerPallet"}


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
    """Cached name -> array-index lookup for sensor_msgs/JointState messages."""

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
    """Build the 6-d state vector from a /joint_states message (identical to real).

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


def _action_from_controller_states(msg: Any) -> np.ndarray:
    """5-d action from /joint_controller_states, same semantics as the real dataset.

    Axis order: [DRIVE, STEER, LIFT, SHIFT, TILT].

    DRIVE and STEER read the *velocity* controller reference (the sim, like the real
    platform, commands both as rates — the DRIVE/STEER position controllers are
    inactive or only briefly active with near-zero references). LIFT/SHIFT/TILT read
    the position controller reference.
    """
    pos = msg.position_controller_states
    vel = msg.velocity_controller_states
    return np.array(
        [
            vel[AXIS_DRIVE].reference,
            vel[AXIS_STEER].reference,
            pos[AXIS_LIFT].reference,
            pos[AXIS_SHIFT].reference,
            pos[AXIS_TILT].reference,
        ],
        dtype=np.float32,
    )


def _controls_from_controller_states(msg: Any) -> dict[str, np.ndarray]:
    """Full PidState record for the six MAPPING.md controls.

    Returns one (6,)-float32 array per PidState subfield plus a (6,)-bool ``is_active``,
    each ordered like CONTROL_NAMES: [driving, steering, steering_rate, lifting,
    shifting, tilting].
    """
    out: dict[str, np.ndarray] = {}
    states = []
    for _, array_name, axis in MAPPING_CONTROLS:
        arr = msg.position_controller_states if array_name == "position" else msg.velocity_controller_states
        states.append(arr[axis])
    for field in PID_FIELDS:
        out[f"controls_{field}"] = np.array([getattr(s, field) for s in states], dtype=np.float32)
    out["controls_is_active"] = np.array([bool(s.is_active) for s in states], dtype=bool)
    return out


def _pallet_delta_from_msg(msg: Any, raw_task: str) -> tuple[np.ndarray, bool]:
    """Extract [dx, dy, dyaw] from PalletSlotPoseInfo for the active target.

    Slot-directed tasks use ``slot_to_fork_mid``; everything else uses
    ``pallet_to_fork_mid``. Returns (zeros, False) while the transform is
    unpopulated (header.stamp == 0).
    """
    target = msg.slot_to_fork_mid if raw_task in SLOT_TASKS else msg.pallet_to_fork_mid
    valid = target.header.stamp.sec != 0 or target.header.stamp.nanosec != 0
    if not valid:
        return np.zeros(PALLET_DELTA_DIM, dtype=np.float32), False
    p = target.pose.position
    q = target.pose.orientation
    return np.array([p.x, p.y, _quat_to_yaw(q.x, q.y, q.z, q.w)], dtype=np.float32), True


def _odom_from_msg(msg: Any) -> np.ndarray:
    """[x, y, yaw, vx, vy, wyaw] ground-truth base pose/twist from /odometry_gt."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    tl = msg.twist.twist.linear
    ta = msg.twist.twist.angular
    return np.array(
        [p.x, p.y, _quat_to_yaw(q.x, q.y, q.z, q.w), tl.x, tl.y, ta.z],
        dtype=np.float32,
    )


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


def _read_bag_files(bag_paths: list[Path], topics: set[str]) -> dict[str, _TopicSeries]:
    """Read topics from one episode's mcap file(s), robust to truncated bags.

    Uses the streaming (non-seeking) mcap reader so that a bag with a corrupt footer
    (e.g. the recorder was killed before finalizing) still yields every message
    written before the truncation point. Timestamps:
      - topics with a header: header sim-time stamp
      - /clock: the clock value itself (used later to remap headerless topics)
      - headerless topics (/current_task): raw log_time, remapped to sim time below
    """
    buffers: dict[str, list[tuple[int, Any]]] = {t: [] for t in topics}
    clock_pairs: list[tuple[int, int]] = []  # (log_time_ns, sim_time_ns)
    read_topics = set(topics) | {CLOCK_TOPIC}
    for bag_path in bag_paths:
        with open(bag_path, "rb") as f:
            reader = NonSeekingReader(f, decoder_factories=[DecoderFactory()])
            try:
                for _schema, channel, message, msg in reader.iter_decoded_messages(topics=list(read_topics)):
                    if channel.topic == CLOCK_TOPIC:
                        clock_pairs.append((message.log_time, _stamp_ns(msg.clock)))
                        continue
                    header = getattr(msg, "header", None)
                    if header is not None and (header.stamp.sec != 0 or header.stamp.nanosec != 0):
                        ts = _stamp_ns(header.stamp)
                    else:
                        ts = message.log_time
                    buffers[channel.topic].append((ts, msg))
            except Exception as e:  # truncated/corrupt mcap: keep what we got
                print(f"  [warn] {bag_path.name}: stopped reading early ({type(e).__name__}: {e})")

    # Remap headerless topics from wall log-time to sim time via /clock.
    headerless = [t for t in topics if buffers[t] and getattr(buffers[t][0][1], "header", None) is None]
    if headerless and len(clock_pairs) >= 2:
        clock_pairs.sort()
        log_t = np.array([c[0] for c in clock_pairs], dtype=np.float64)
        sim_t = np.array([c[1] for c in clock_pairs], dtype=np.float64)
        for t in headerless:
            buffers[t] = [(int(np.interp(float(ts), log_t, sim_t)), m) for ts, m in buffers[t]]
    elif headerless:
        print(f"  [warn] no /clock data; cannot align headerless topics {headerless}")
        for t in headerless:
            buffers[t] = []

    return {t: _TopicSeries.from_pairs(p) for t, p in buffers.items()}


def _iter_bag_dirs(data_dir: Path):
    for bag_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        mcaps = sorted(bag_dir.glob("*.mcap"))
        if not mcaps:
            print(f"  [skip] {bag_dir.name}: no .mcap files")
            continue
        yield bag_dir, mcaps


class _TaskWarner:
    """Warn once per unknown /current_task tag across the whole run."""

    def __init__(self) -> None:
        self.unknown: set[str] = set()

    def prompt_for(self, raw: str) -> str:
        prompt = TASK_STRINGS.get(raw)
        if prompt is None:
            if raw not in self.unknown:
                print(f"  [warn] unknown /current_task tag {raw!r}; using it verbatim as the prompt")
                self.unknown.add(raw)
            return raw
        return prompt


def _build_frames(
    series: dict[str, _TopicSeries],
    cameras: list[CameraName],
    fps: int,
    task_warner: _TaskWarner,
) -> list[dict[str, Any]]:
    period_ns = 1_000_000_000 // fps

    required_topics = [JOINT_STATES_TOPIC, CONTROLLER_STATES_TOPIC] + [_camera_topic(c) for c in cameras]
    required_first = []
    for t in required_topics:
        first = series[t].first_ns
        if first is None:
            print(f"  [skip] no messages on required topic {t}")
            return []
        required_first.append(first)
    start_ns = max(required_first)
    end_ns = min(int(series[t].times[-1]) for t in required_topics)
    if end_ns <= start_ns:
        return []

    js_first = series[JOINT_STATES_TOPIC].msgs[0]
    jx = _JointIndex.from_msg(js_first)
    missing = [n for n in REQUIRED_JOINTS if n not in jx.idx]
    if missing:
        print(f"  [skip] /joint_states is missing required joints: {missing}")
        return []

    cs_first = series[CONTROLLER_STATES_TOPIC].msgs[0]
    labels = [str(x) for x in cs_first.labels]
    if labels != EXPECTED_LABELS:
        print(f"  [skip] unexpected controller labels {labels} (want {EXPECTED_LABELS})")
        return []

    frames: list[dict[str, Any]] = []
    t_ns = start_ns
    while t_ns <= end_ns:
        js_msg = series[JOINT_STATES_TOPIC].latest_at(t_ns)
        cs_msg = series[CONTROLLER_STATES_TOPIC].latest_at(t_ns)
        cam_msgs = {c: series[_camera_topic(c)].latest_at(t_ns) for c in cameras}
        if js_msg is None or cs_msg is None or any(m is None for m in cam_msgs.values()):
            t_ns += period_ns
            continue

        task_msg = series[CURRENT_TASK_TOPIC].latest_at(t_ns)
        if task_msg is None and series[CURRENT_TASK_TOPIC].times.size > 0:
            task_msg = series[CURRENT_TASK_TOPIC].msgs[0]  # backfill leading frames
        raw_task = task_msg.data if task_msg is not None else "FollowPath"

        pallet_msg = series[PALLET_INFO_TOPIC].latest_at(t_ns)
        if pallet_msg is None:
            pallet_delta = np.zeros(PALLET_DELTA_DIM, dtype=np.float32)
            pallet_delta_valid = False
        else:
            pallet_delta, pallet_delta_valid = _pallet_delta_from_msg(pallet_msg, raw_task)

        odom_msg = series[ODOMETRY_GT_TOPIC].latest_at(t_ns)
        odom = _odom_from_msg(odom_msg) if odom_msg is not None else np.zeros(ODOM_DIM, dtype=np.float32)

        frame: dict[str, Any] = {
            "state": _state_from_joint_states(js_msg, jx),
            "actions": _action_from_controller_states(cs_msg),
            "pallet_delta": pallet_delta,
            "pallet_delta_valid": np.array([pallet_delta_valid], dtype=bool),
            "odom_gt": odom,
            "task": task_warner.prompt_for(raw_task),
        }
        frame.update(_controls_from_controller_states(cs_msg))
        for c, msg in cam_msgs.items():
            frame[CAMERA_FEATURE_KEYS[c]] = _decode_image(msg)
        frames.append(frame)
        t_ns += period_ns
    return frames


def _build_features(cameras: list[CameraName]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for c in cameras:
        features[CAMERA_FEATURE_KEYS[c]] = {
            "dtype": "image",
            "shape": (IMAGE_HW, IMAGE_HW, 3),
            "names": ["height", "width", "channel"],
        }
    features["state"] = {
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
    }
    features["actions"] = {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": [
            "drive",  # velocity_controller_states[DRIVE].reference (velocity)
            "steer",  # velocity_controller_states[STEER].reference (steering rate)
            "lift",   # position_controller_states[LIFT].reference  (position)
            "shift",  # position_controller_states[SHIFT].reference (position)
            "tilt",   # position_controller_states[TILT].reference  (position)
        ],
    }
    for field in PID_FIELDS:
        features[f"controls_{field}"] = {
            "dtype": "float32",
            "shape": (len(CONTROL_NAMES),),
            "names": list(CONTROL_NAMES),
        }
    features["controls_is_active"] = {
        "dtype": "bool",
        "shape": (len(CONTROL_NAMES),),
        "names": list(CONTROL_NAMES),
    }
    features["pallet_delta"] = {
        "dtype": "float32",
        "shape": (PALLET_DELTA_DIM,),
        "names": ["dx", "dy", "dyaw"],
    }
    features["pallet_delta_valid"] = {
        "dtype": "bool",
        "shape": (1,),
        "names": ["valid"],
    }
    features["odom_gt"] = {
        "dtype": "float32",
        "shape": (ODOM_DIM,),
        "names": ["x", "y", "yaw", "vx", "vy", "wyaw"],
    }
    return features


def main(
    data_dir: str,
    repo_id: str = "local/forklift_sim",
    *,
    cameras: tuple[CameraName, ...] = ("top", "left", "right", "third_person"),
    fps: int = 10,
    push_to_hub: bool = False,
    private: bool = False,
    limit_episodes: int | None = None,
) -> None:
    if "top" not in cameras:
        raise ValueError("'top' must be included: it fills the base 'image' feature")
    cams = list(cameras)

    src = Path(data_dir)
    out = HF_LEROBOT_HOME / repo_id
    if out.exists():
        shutil.rmtree(out)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="forklift",
        fps=fps,
        features=_build_features(cams),
        image_writer_threads=10,
        image_writer_processes=5,
    )

    topics = (
        {JOINT_STATES_TOPIC, CONTROLLER_STATES_TOPIC, CURRENT_TASK_TOPIC, PALLET_INFO_TOPIC, ODOMETRY_GT_TOPIC}
        | {_camera_topic(c) for c in cams}
    )
    task_warner = _TaskWarner()

    n_episodes = 0
    for bag_dir, mcaps in _iter_bag_dirs(src):
        if limit_episodes is not None and n_episodes >= limit_episodes:
            break
        print(f"[{n_episodes:04d}] {bag_dir.name}: reading {len(mcaps)} mcap file(s)")
        series = _read_bag_files(mcaps, topics)
        frames = _build_frames(series, cams, fps, task_warner)
        if not frames:
            print("  [skip] no aligned frames")
            continue
        for f in frames:
            ds.add_frame(f)
        ds.save_episode()
        n_episodes += 1
        tasks = list(dict.fromkeys(f["task"] for f in frames))
        print(f"  -> {len(frames)} frames, tasks={tasks}")

    print(f"Done. Wrote {n_episodes} episodes to {out}.")

    if push_to_hub:
        ds.push_to_hub(
            tags=["forklift", "pi05", "ros2", "mcap", "simulation"],
            private=private,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
