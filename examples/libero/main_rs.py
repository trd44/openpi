"""
Script to use the openpi model in robosuite environments. 
"""
import collections
import dataclasses
import logging
import math
import pathlib
import os
import time
from typing import List

import imageio
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
# from robosuite.utils.input_utils import * # Uncomment for keyboard debugging if needed

# Assuming openpi_client is installed in the environment
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

SEED = 3
ENV_NAME = "Hanoi"  #: Robosuite environment (e.g., Stack, Lift, PickPlace)
PROMPT = f"Place the green block in the right area."
REPLAN_STEPS = 50
HORIZON = 5000
VIDEO_NAME =f"fixed 27k rs {REPLAN_STEPS} horizon {HORIZON} {ENV_NAME} seed {SEED} Panda {PROMPT}.mp4"  #: Filename for the output video

@dataclasses.dataclass
class Args:
    """Arguments for running Robosuite with OpenPI Websocket Policy"""
    # --- Server Connection ---
    host: str = "127.0.0.1" #: Hostname of the OpenPI policy server
    port: int = 8000 #: Port of the OpenPI policy server

    # --- Policy Interaction ---
    resize_size: int = 224 #: Target size for image resizing (must match model training)
    replan_steps: int = REPLAN_STEPS #: Number of steps per action chunk from policy server

    # --- Robosuite Environment ---
    env_name: str = ENV_NAME 
    robots: str = "Panda" #: Robot model to use
    controller: str = "OSC_POSE" #: Robosuite controller name
    horizon: int = HORIZON #: Max steps per episode
    skip_steps: int = 50 #: Number of initial steps to skip (e.g., wait for objects to settle)

    # --- Rendering & Video ---
    render_mode: str = "headless" #: Rendering mode: 'headless' (save video) or 'human' (live view via X11)
    video_out_path: str = "data/robosuite_videos" #: Directory to save videos
    video_filename: str = VIDEO_NAME
    camera_names: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
        ) #: Cameras for observation/video
    camera_height: int = 256 #: Rendered camera height (before potential resize)
    camera_width: int = 256 #: Rendered camera width (before potential resize)

    # --- Misc ---
    seed: int = SEED #: Random seed


def run_robosuite_with_openpi(args: Args) -> None:
    """Runs a Robosuite environment controlled by an OpenPI policy server."""
    np.random.seed(args.seed)
    logging.info(f"Running Robosuite env '{args.env_name}' with OpenPI server at {args.host}:{args.port}")
    logging.info(f"Rendering mode: {args.render_mode}")

    # --- Environment Setup ---
    controller_config = load_controller_config(default_controller=args.controller)

    has_renderer = (args.render_mode == 'human')
    # Need offscreen renderer for observations even if rendering on-screen
    has_offscreen = True

    # Verify required cameras are requested
    required_cams = ["agentview", "robot0_eye_in_hand"] # Cameras needed for OpenPI preprocessing
    for cam in required_cams:
        if cam not in args.camera_names:
            logging.warning(f"Required camera '{cam}' not in requested camera_names. Adding it.")
            args.camera_names.append(cam)

    env = suite.make(
        env_name=args.env_name,
        robots=args.robots,
        controller_configs=controller_config,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen,
        control_freq=20, # Robosuite default
        horizon=args.horizon,
        use_object_obs=True, # Get state observations
        use_camera_obs=True, # Get camera observations
        camera_names=args.camera_names,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        render_camera="agentview" if has_renderer else None,
        ignore_done=True, # Let horizon end the episode
        hard_reset=False, # Faster resets can sometimes be unstable, switch if needed
        random_reset=False
    )
    try:
        env.seed(args.seed)
        logging.info(f"Environment seeded with {args.seed} using env.seed()")
    except AttributeError:
        logging.warning("env.seed() method not found. Seeding might rely on env.reset(seed=...) instead.")
    except Exception as e:
        logging.error(f"Error calling env.seed(): {e}")
        # Decide if this is fatal

    # --- Websocket Client Setup ---
    logging.info(f"Connecting to OpenPI server at ws://{args.host}:{args.port}...")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    # Consider adding a loop here to wait for server connection

    # --- Video Saving Setup ---
    video_full_path = pathlib.Path(args.video_out_path) / args.video_filename
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    frames = []

    # --- Simulation Loop ---
    try:
        obs = env.reset()
    except Exception as e:
        logging.error(f"Failed to reset environment: {e}")
        if "unexpected keyword argument 'seed'" in str(e):
             logging.error("Environment reset failed likely because it expects env.seed() instead of env.reset(seed=...).")
        return # Cannot proceed if reset fails

    action_plan = collections.deque()
    t = 0
    total_reward = 0.0
    start_episode_time = time.time()
    logging.info("Starting episode...")

    while t < args.horizon:
        loop_start_time = time.time()

        # Skip initial steps if needed
        if t < args.skip_steps:
            action = np.zeros(env.action_dim)
            if env.robots[0].gripper.dof > 0: # Assumes gripper is last dim(s)
                # Action space is typically [-1, 1]. -1 often means close gripper.
                action[6] = -1.0 # Adjust index if action space is different
            obs, reward, done, info = env.step(action)
            t += 1
            if has_renderer: env.render()
            # Save frame for video if rendering offscreen
            # elif has_offscreen and "agentview_image" in obs:
            #      frames.append(obs["agentview_image"]) # Save raw agentview
            continue

        # --- Preprocess Observations ---
        try:
            # Ensure keys exist (important!)
            # Define the exact keys needed from the observation dictionary
            required_obs_keys = [
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos"
            ]
            # Verify that ALL required keys are present in the observation dictionary
            if not all(k in obs for k in required_obs_keys):
                logging.error(
                    f"Missing required observation keys at step {t}. \
                    Needed: {required_obs_keys}. Available: {list(obs.keys())}"
                    )
                break # Stop if any key is missing

            img_obs = obs["agentview_image"]
            wrist_obs = obs["robot0_eye_in_hand_image"]
            # gripper_qpos_obs = translate_12_state_to_8(obs)

            # Rotate 180 degrees
            img = np.ascontiguousarray(img_obs[::-1, ::-1], dtype=np.uint8) # Rotate
            wrist_img = np.ascontiguousarray(wrist_obs[::-1, ::-1], dtype=np.uint8) # Rotate

            # Resize and pad
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))

            # Save raw frame for video
            if has_offscreen:
                frames.append(img)

        except KeyError as e:
            logging.error(f"Observation key error at step {t}: {e}. Available keys: {obs.keys()}")
            break
        except Exception as e:
             logging.error(f"Error during preprocessing at step {t}: {e}")
             break
        # eef_pos = obs.get('robot0_eef_pos', np.zeros(3, dtype=np.float32))
        # eef_quat = obs.get('robot0_eef_quat', np.array([0., 0., 0., 1.], dtype=np.float32))
        # eef_axis_angle = _quat2axisangle(eef_quat)
        # # eef_euler = quaternion_to_euler(eef_quat)  # (3,)

        # eef_state = np.concatenate([eef_pos, eef_axis_angle])  # (6,)
        # STATE (END EFFECTOR)
        eef_pos = obs.get('robot0_eef_pos', np.zeros(3, dtype=np.float32))
        eef_quat = obs.get('robot0_eef_quat', np.array([0., 0., 0., 1.], dtype=np.float32))
        eef_gripper = obs.get('robot0_gripper_qpos', np.zeros(2, dtype=np.float32))
        # Convert quaternion to axis angle
        eef_axis_angle = _quat2axisangle(eef_quat)

        eef_state = np.concatenate((eef_pos, eef_axis_angle, eef_gripper)).astype(np.float32)

        # gripper_qpos = np.array(obs.get('robot0_gripper_qpos', [0., 0.]), dtype=np.float32)
        # # If only one value, duplicate to 2D for RLDS compatibility
        # if gripper_qpos.shape[0] == 1:
        #     gripper_state = np.array([gripper_qpos[0], gripper_qpos[0]], dtype=np.float32)
        # else:
        #     gripper_state = gripper_qpos[:2]

        # state_vec = np.concatenate([eef_state, gripper_state]).astype(np.float32)  # shape (8,)

        # --- Get Action from OpenPI Server ---
        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": eef_state,
                # "prompt": f"Complete the {args.env_name} task",
                "prompt": PROMPT,
            }
            # print(f"EEF Pos = {obs['robot0_eef_pos']}")
            # print(f"Axis angle = {_quat2axisangle(obs['robot0_eef_quat'])}")
            # print(f"Gripper qpos = {gripper_qpos_obs}")
            # print(f"state_vec = {element['observation/state']}")
            logging.debug("Requesting inference from server...")
            try:
                # --- Debug prints before sending to server ---
                # print("==== Sending element to OpenPI server ====")
                # print("Prompt:", element.get("prompt"))
                # print("Observation/image shape:", element["observation/image"].shape, "dtype:", element["observation/image"].dtype)
                # print("Observation/wrist_image shape:", element["observation/wrist_image"].shape, "dtype:", element["observation/wrist_image"].dtype)
                # print("Observation/state:", element["observation/state"])

                inf_start = time.time()
                inference_result = client.infer(element)
                logging.debug(f"Inference time: {time.time() - inf_start:.4f}s")

                # --- Debug prints after receiving server response ---
                # print("==== Received inference_result from OpenPI server ====")
                # print("Raw inference_result:", inference_result)

                if "actions" not in inference_result:
                    logging.error(f"Server response missing 'actions': {inference_result}")
                    break
                action_chunk = inference_result["actions"]

                # --- Debug print for action_chunk ---
                # print("Action chunk type:", type(action_chunk))
                # if isinstance(action_chunk, np.ndarray):
                #     print("Action chunk shape:", action_chunk.shape, "dtype:", action_chunk.dtype)
                #     print("Action chunk sample (first 1):", action_chunk[:1])

                if not isinstance(action_chunk, np.ndarray) or action_chunk.size == 0:
                    logging.warning("Policy server returned empty or invalid action chunk.")
                    # Send zero action as fallback
                    action_chunk = np.zeros((args.replan_steps, env.action_dim))
                    if env.robots[0].gripper.dof > 0: action_chunk[:, 6] = -1.0 # Keep gripper closed?

                # Ensure chunk shape is reasonable (at least N, action_dim)
                if len(action_chunk.shape) != 2 or action_chunk.shape[1] != env.action_dim:
                    logging.error(
                        f"Action chunk shape mismatch. Expected (~N, {env.action_dim}), Got: {action_chunk.shape}")
                    break

                action_plan.extend(action_chunk[: args.replan_steps])

            except Exception as e:
                logging.error(f"Failed to get inference from server: {e}")
                import traceback
                print("Exception occurred during inference:")
                traceback.print_exc()
                break # Exit loop

        # Get next action
        if action_plan:
             action = action_plan.popleft()
        else:
             logging.warning("Action plan empty after inference attempt. Sending zero action.")
             action = np.zeros(env.action_dim)
             if env.robots[0].gripper.dof > 0: action[6] = -1.0 # Keep gripper closed?

        action = action.tolist()
        action[3] = 0
        action[4] = 0
        action[5] = 0

        # --- Step Environment ---
        # try:
        #     # Ensure action is numpy array and correct shape before tolist()
        #     if not isinstance(action, np.ndarray): action = np.array(action) # Convert if needed
        #     if action.shape[0] != env.action_dim:
        #         logging.error(f"Action dimension mismatch before step. Policy: {action.shape[0]}, Env: {env.action_dim}")
        #         break
        obs, reward, done, info = env.step(action) # Must match env.action_space!
        total_reward += reward
        t += 1
        # except Exception as e:
        #     logging.error(f"Error during env.step at step {t}: {e}")
        #     break

        # Render step if requested
        if has_renderer:
            env.render()

        # Logging
        if t % 50 == 0:
            logging.info(
                f"Step: {t}/{args.horizon}, Reward: {reward:.3f},\
                Loop Time: {time.time() - loop_start_time:.4f}s"
            )
        else:
            logging.debug(
                f"Step: {t}/{args.horizon}, Reward: {reward:.3f}, \
                Loop Time: {time.time() - loop_start_time:.4f}s"
            )

    # --- End of Episode ---
    episode_duration = time.time() - start_episode_time
    logging.info(f"Episode finished after {t} steps. Duration: {episode_duration:.2f}s. Total reward: {total_reward:.3f}")

    # Save video
    if frames:
        logging.info(f"Saving video ({len(frames)} frames) to {video_full_path}...")
        try:
            imageio.mimwrite(str(video_full_path), frames, fps=env.control_freq)
            logging.info("Video saved.")
        except Exception as e:
            logging.error(f"Failed to save video: {e}")
    else:
        logging.warning("No frames collected for video.")

    env.close()
    logging.info("Environment closed.")


# def translate_12_state_to_8(obs_dict):
#     # --- 3. Extract/Construct Proprioceptive State ---
#     # Reduced form for proprioceptive state - 7 joint positions and 1
#     EXPECTED_STATE_DIM = 8
#     # --- Gripper QPOS interpretation (NEEDS USER INPUT FROM XML) ---
#     # Example: Assume first joint in qpos is main actuator
#     GRIPPER_QPOS_INDEX = 0
#     # Find these limits in your specific Kinova gripper XML file!
#     GRIPPER_JOINT_MIN = 0.0 # Example Placeholder
#     GRIPPER_JOINT_MAX = 0.8 # Example Placeholder (e.g., Robotiq 2F-85 range)
#     # ---
#     try:
#         # --- Define keys and expected dimensions (EXAMPLE for Panda) ---
#         # --- ADJUST KEYS AND DIMS FOR KINOVA3 ---
#         # print(obs_dict.keys())
#         joint_sin = obs_dict['robot0_joint_pos_sin']
#         joint_cos = obs_dict['robot0_joint_pos_cos']
#         # Calculate the 7 joint angles in radians
#         arm_joint_positions = np.arctan2(joint_sin, joint_cos)

#         # --- Get 1 Gripper State value by interpreting qpos ---
#         gripper_qpos = obs_dict.get('robot0_gripper_qpos')
#         kinova_gripper_state_norm = 0.0 # Default value

#         if gripper_qpos is not None and len(gripper_qpos) > GRIPPER_QPOS_INDEX:
#             current_joint_val = gripper_qpos[GRIPPER_QPOS_INDEX]
#             # Normalize based on known limits (USER MUST PROVIDE)
#             if GRIPPER_JOINT_MAX > GRIPPER_JOINT_MIN:
#                 kinova_gripper_state_norm = (current_joint_val - GRIPPER_JOINT_MIN) / (GRIPPER_JOINT_MAX - GRIPPER_JOINT_MIN)
#                 kinova_gripper_state_norm = np.clip(kinova_gripper_state_norm, 0.0, 1.0)
#             else:
#                 print(f"Warning: Invalid gripper joint limits [{GRIPPER_JOINT_MIN}, {GRIPPER_JOINT_MAX}].")
#         else:
#             print(f"Warning: 'robot0_gripper_qpos' not found or too short. Using 0.0 for gripper.")
#         kinova_gripper_state = np.array([kinova_gripper_state_norm], dtype=np.float32) # Shape (1,)
#         # # --- Concatenate into 8-element state vector ---
#         # proprio_state = np.concatenate([
#         #     arm_joint_positions,    # 7 elements
#         #     kinova_gripper_state    # 1 element
#         # ]).astype(np.float32)
#         # ---
#         return kinova_gripper_state

#     except (KeyError, ValueError, TypeError) as e:
#         print(f"Error constructing proprio state manually in step {step_index}: {e}. Using zeros.")
#         proprio_state = np.zeros(EXPECTED_STATE_DIM, dtype=np.float32)
#         return 0


def _quat2axisangle(quat):
    """
    Copied from robosuite: 
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a1
    6bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)."""
    x, y, z, w = quat
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s'
        )
    tyro.cli(run_robosuite_with_openpi)