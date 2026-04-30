#!/usr/bin/env python3
"""ROS 2 inference node for the pi0.5 forklift policy.

Subscribes to the same topics that were used during data conversion, builds the
observation, talks to a running ``openpi`` policy server over a websocket, and
publishes the predicted action back to ``/joint_commands``.

  +------------------------------+        websocket         +-------------------+
  |  GPU host                    |  <-------------------->  |  Forklift PC      |
  |  uv run scripts/serve_policy |   obs / action chunk     |  this node        |
  +------------------------------+                          +-------------------+

This node is intentionally self-contained: it imports ROS 2 message types and the
small ``openpi_client`` package, nothing else from the openpi tree. You can drop
it into any ROS 2 workspace that has ``hopper_msgs`` and ``sensor_msgs`` on the
``ament`` index.

See ``TOPICS.md`` (in this directory) for the authoritative input/output mapping
and ``INFERENCE.md`` for end-to-end run instructions.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage
from hopper_msgs.msg import Commands, LinearsStamped, DriveStamped

# openpi_client lives in openpi/packages/openpi-client. Install it once with:
#     uv pip install -e packages/openpi-client
# (or `pip install -e <path-to-openpi-client>` in whatever Python the robot uses).
from openpi_client import action_chunk_broker, websocket_client_policy


# --- topic / shape constants (must match TOPICS.md exactly) -------------------
TOPIC_IMAGE = "/zed2i_top/zed2i/warped/left/image_rect_color/compressed"
TOPIC_LINEARS = "/crayler/linears_stamped"
TOPIC_DRIVE = "/crayler/drive_stamped"
TOPIC_JOINT_COMMANDS = "/joint_commands"

IMAGE_HW = 224
STATE_DIM = 10
ACTION_DIM = 5
ACTION_HORIZON = 10  # must match Pi0Config(action_horizon=10) used in training
INFER_HZ = 10.0      # must match the dataset fps

# Per-axis active_control to set on every published Commands message.
# (Kept in sync with TOPICS.md and the training-data convention.)
_ACTIVE_OFF = 0
_ACTIVE_POS = 1
_ACTIVE_VEL = 2
ACTIVE_CONTROL = bytes(
    [
        _ACTIVE_VEL,  # DRIVE
        _ACTIVE_POS,  # STEER
        _ACTIVE_POS,  # LIFT
        _ACTIVE_OFF,  # SHIFT
        _ACTIVE_POS,  # TILT
    ]
)


# QoS for image-like topics (often BEST_EFFORT on real systems).
_QOS_SENSOR = qos_profile_sensor_data
# QoS for control-bus topics: RELIABLE depth=10 is a safe default.
_QOS_CONTROL = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


@dataclass
class _Latest:
    """Thread-safe holder for the most recent message of a topic."""

    msg: object | None = None
    lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.lock = threading.Lock()

    def set(self, msg: object) -> None:
        with self.lock:
            self.msg = msg

    def get(self) -> object | None:
        with self.lock:
            return self.msg


class ForkliftInferenceNode(Node):
    """ROS 2 node that pipes sensor topics through a remote pi0.5 policy server."""

    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        publish_actions: bool,
        action_horizon: int = ACTION_HORIZON,
        infer_hz: float = INFER_HZ,
    ) -> None:
        super().__init__("forklift_pi05_inference")
        self._prompt = prompt
        self._publish_actions = publish_actions
        self._period = 1.0 / infer_hz

        self._latest_image = _Latest()
        self._latest_linears = _Latest()
        self._latest_drive = _Latest()

        # Subscriptions
        self.create_subscription(CompressedImage, TOPIC_IMAGE, self._on_image, _QOS_SENSOR)
        self.create_subscription(LinearsStamped, TOPIC_LINEARS, self._on_linears, _QOS_CONTROL)
        self.create_subscription(DriveStamped, TOPIC_DRIVE, self._on_drive, _QOS_CONTROL)

        # Publisher
        self._cmd_pub = self.create_publisher(Commands, TOPIC_JOINT_COMMANDS, _QOS_CONTROL)

        # Policy client + chunk broker (one inference call yields ACTION_HORIZON
        # actions; the broker hands them out one tick at a time).
        self.get_logger().info(f"Connecting to policy server at ws://{host}:{port} ...")
        ws_policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.get_logger().info(f"Server metadata: {ws_policy.get_server_metadata()}")
        self._policy = action_chunk_broker.ActionChunkBroker(
            policy=ws_policy, action_horizon=action_horizon
        )

        # Loop timer
        self._timer = self.create_timer(self._period, self._tick)
        self._tick_idx = 0
        self._first_warning_emitted = False

    # --- callbacks ---------------------------------------------------------
    def _on_image(self, msg: CompressedImage) -> None:
        self._latest_image.set(msg)

    def _on_linears(self, msg: LinearsStamped) -> None:
        self._latest_linears.set(msg)

    def _on_drive(self, msg: DriveStamped) -> None:
        self._latest_drive.set(msg)

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _decode_image(msg: CompressedImage) -> np.ndarray:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("cv2.imdecode returned None")
        bgr = cv2.resize(bgr, (IMAGE_HW, IMAGE_HW), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb)

    @staticmethod
    def _state_from(linears: LinearsStamped, drive: DriveStamped) -> np.ndarray:
        lin = linears.linears
        drv = drive.drive
        return np.array(
            [
                lin.lin_pot_left,
                lin.lin_pot_right,
                lin.lin_pot_vertical,
                float(lin.length_hubmast),
                float(lin.length_seitenhub),
                drv.steering_angle,
                drv.steering_angle_rate,
                drv.front_left_speed,
                drv.front_right_speed,
                drv.rho,
            ],
            dtype=np.float32,
        )

    def _build_commands_msg(self, action: np.ndarray) -> Commands:
        """Pack a 5-d action vector into a hopper_msgs/Commands message.

        Per-axis convention (matches TOPICS.md):
          DRIVE  -> velocity_commands[0]   active_control[0] = VEL
          STEER  -> position_commands[1]   active_control[1] = POS
          LIFT   -> position_commands[2]   active_control[2] = POS
          SHIFT  -> 0.0 (ignored)          active_control[3] = OFF
          TILT   -> position_commands[4]   active_control[4] = POS
        """
        msg = Commands()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = "front"

        msg.position_commands = [
            0.0,
            float(action[1]),  # STEER
            float(action[2]),  # LIFT
            0.0,               # SHIFT (ignored)
            float(action[4]),  # TILT
        ]
        msg.velocity_commands = [
            float(action[0]),  # DRIVE
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        msg.effort_commands = [0.0] * 5
        msg.ff_commands = [0.0] * 5
        msg.active_control = ACTIVE_CONTROL
        msg.is_stop = False
        msg.is_joy_control = False
        return msg

    def _warn_once(self, text: str) -> None:
        if not self._first_warning_emitted:
            self.get_logger().warn(text)
            self._first_warning_emitted = True

    # --- main loop ---------------------------------------------------------
    def _tick(self) -> None:
        img_msg = self._latest_image.get()
        lin_msg = self._latest_linears.get()
        drv_msg = self._latest_drive.get()
        if img_msg is None or lin_msg is None or drv_msg is None:
            self._warn_once(
                "waiting for first message on each input topic before inference starts ..."
            )
            return

        try:
            image = self._decode_image(img_msg)
        except Exception as e:
            self.get_logger().error(f"image decode failed: {e}")
            return

        state = self._state_from(lin_msg, drv_msg)

        obs = {
            "observation/image": image,
            "observation/state": state,
            "prompt": self._prompt,
        }

        try:
            result = self._policy.infer(obs)
        except Exception as e:
            self.get_logger().error(f"policy inference failed: {e}")
            return

        action = np.asarray(result["actions"], dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            # ActionChunkBroker yields one timestep at a time, so we expect (ACTION_DIM,)
            self.get_logger().error(
                f"unexpected action shape {action.shape}, expected ({ACTION_DIM},); skipping publish"
            )
            return

        cmd = self._build_commands_msg(action)

        if self._publish_actions:
            self._cmd_pub.publish(cmd)

        self._tick_idx += 1
        if self._tick_idx % int(INFER_HZ) == 0:
            self.get_logger().info(
                f"tick {self._tick_idx:5d}  "
                f"action=[drv={action[0]:+.3f} str={action[1]:+.3f} "
                f"lft={action[2]:+.3f} shf={action[3]:+.3f} tlt={action[4]:+.3f}]  "
                f"published={self._publish_actions}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="hostname/IP of the openpi policy server")
    parser.add_argument("--port", type=int, default=8000, help="policy-server websocket port")
    parser.add_argument(
        "--task",
        required=True,
        help='Language prompt, e.g. "Engage the forks under the pallet on the ground". '
        "Must be one of the six prompts the model was trained on (see TOPICS.md), "
        "or paraphrasing thereof at your own risk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference and log actions but do NOT publish to /joint_commands. "
        "Use this on first bring-up to confirm the loop works before letting the policy drive.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=ACTION_HORIZON,
        help="Number of timesteps per inference call. Must match the trained model.",
    )
    parser.add_argument(
        "--infer-hz",
        type=float,
        default=INFER_HZ,
        help="Loop rate. Must match the dataset fps used during training.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)

    rclpy.init()
    node = ForkliftInferenceNode(
        host=args.host,
        port=args.port,
        prompt=args.task,
        publish_actions=not args.dry_run,
        action_horizon=args.action_horizon,
        infer_hz=args.infer_hz,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
