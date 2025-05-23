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


ENV_NAME = "Hanoi"  #: Robosuite environment (e.g., Stack, Lift, PickPlace)
PROMPT = f"pick the blue cube from red cube"
NUM = "0"
VIDEO_NAME = "finetune16000_kinova3" + ENV_NAME + " " + PROMPT + NUM + ".mp4"  #: Filename for the output video

@dataclasses.dataclass
class Args:
    """Arguments for running Robosuite with OpenPI Websocket Policy"""
    # --- Server Connection ---
    host: str = "127.0.0.1" #: Hostname of the OpenPI policy server
    port: int = 8000 #: Port of the OpenPI policy server

    # --- Policy Interaction ---
    resize_size: int = 224 #: Target size for image resizing (must match model training)
    replan_steps: int = 5 #: Number of steps per action chunk from policy server

    # --- Robosuite Environment ---
    env_name: str = ENV_NAME 
    robots: str = "Kinova3" #: Robot model to use
    controller: str = "OSC_POSE" #: Robosuite controller name
    horizon: int = 500 #: Max steps per episode
    skip_steps: int = 20 #: Number of initial steps to skip (e.g., wait for objects to settle)

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
    seed: int = 7 #: Random seed


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

            # Rotate 180 degrees
            img = np.ascontiguousarray(img_obs[::-1, ::-1]) # Rotate
            wrist_img = np.ascontiguousarray(wrist_obs[::-1, ::-1]) # Rotate

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

        # --- Get Action from OpenPI Server ---
        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate(
                    (
                        obs['robot0_eef_pos'], 
                        _quat2axisangle(obs['robot0_eef_quat']), 
                        obs['robot0_gripper_qpos'],
                    )
                ),
                # "prompt": f"Complete the {args.env_name} task",
                "prompt": PROMPT,
            }
            logging.debug("Requesting inference from server...")
            try:
                inf_start = time.time()
                inference_result = client.infer(element)
                logging.debug(f"Inference time: {time.time() - inf_start:.4f}s")

                if "actions" not in inference_result:
                    logging.error(f"Server response missing 'actions': {inference_result}")
                    break
                action_chunk = inference_result["actions"]

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
                break # Exit loop

        # Get next action
        if action_plan:
             action = action_plan.popleft()
        else:
             logging.warning("Action plan empty after inference attempt. Sending zero action.")
             action = np.zeros(env.action_dim)
             if env.robots[0].gripper.dof > 0: action[6] = -1.0 # Keep gripper closed?

        # --- Step Environment ---
        try:
            # Ensure action is numpy array and correct shape before tolist()
            if not isinstance(action, np.ndarray): action = np.array(action) # Convert if needed
            if action.shape[0] != env.action_dim:
                logging.error(f"Action dimension mismatch before step. Policy: {action.shape[0]}, Env: {env.action_dim}")
                break
            obs, reward, done, info = env.step(action.tolist()) # Must match env.action_space!
            total_reward += reward
            t += 1
        except Exception as e:
            logging.error(f"Error during env.step at step {t}: {e}")
            break

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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s'
        )
    tyro.cli(run_robosuite_with_openpi)