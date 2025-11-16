"""
Main script for running Robosuite with OpenPI policy server.
"""
import collections
import dataclasses
import logging
import math
import pathlib
import time
import psutil
import subprocess
from typing import Dict, Any
from args import Args

import imageio
import numpy as np
import wandb
import tyro

# Assuming openpi_client is installed in the environment
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from hanoi_detectors import PandaHanoiDetector as SimpleHanoiDetector
from task_manager import TaskManager
from multi_config_hanoi_environmnet import MultiConfigHanoiEnvironment

# ------------------------------------------------------------------------------
# Constants and Configuration
# ------------------------------------------------------------------------------
# Task configuration
COLOR2OBJ = {
    "blue": "cube1", "red": "cube2", "green": "cube3", "yellow": "cube4"}
AREA2PEG = {"left": "peg1", "middle": "peg2", "right": "peg3"}

# Planning predicates and modes
PLANNING_PREDICATES = {
    "Hanoi": ['on', 'clear', 'grasped', 'smaller'],
    "Hanoi4x3": ['on', 'clear', 'grasped', 'smaller'],
    "KitchenEnv": ['on', 'clear', 'grasped', 'stove_on'],
    "NutAssembly": ['on', 'clear', 'grasped'],
}
PLANNING_MODE = {"Hanoi": 0, "Hanoi4x3": 0, "KitchenEnv": 1, "NutAssembly": 0}


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to axis-angle representation.
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


# ------------------------------------------------------------------------------
# System monitoring functions
# ------------------------------------------------------------------------------
def get_gpu_power_usage():
    """Get GPU power usage in watts."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            power_values = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            return sum(power_values) if power_values else 0.0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass
    return 0.0

def get_gpu_utilization():
    """Get GPU utilization percentage."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            util_values = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            return sum(util_values) / len(util_values) if util_values else 0.0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass
    return 0.0

def get_gpu_memory_usage():
    """Get GPU memory usage percentage."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            total_used = 0
            total_available = 0
            for line in lines:
                if ',' in line:
                    used, total = line.split(',')
                    total_used += float(used.strip())
                    total_available += float(total.strip())
            return (total_used / total_available * 100) if total_available > 0 else 0.0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass
    return 0.0

def get_cpu_power_usage():
    """Get CPU power usage estimation based on frequency and utilization."""
    try:
        # Get CPU frequency and utilization
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        if cpu_freq and cpu_freq.current > 0:
            # Rough estimation: higher frequency and utilization = higher power
            # This is a simplified model - actual power depends on many factors
            base_power = 15.0  # Base power in watts
            freq_factor = (cpu_freq.current / cpu_freq.max) if cpu_freq.max > 0 else 1.0
            util_factor = cpu_percent / 100.0
            
            estimated_power = base_power * freq_factor * (0.5 + 0.5 * util_factor)
            return estimated_power
    except Exception:
        pass
    return 0.0

def get_system_metrics():
    """Get comprehensive system metrics with timestamps."""
    try:
        # Get current timestamp
        current_time = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        # GPU metrics
        gpu_power = get_gpu_power_usage()
        gpu_util = get_gpu_utilization()
        gpu_memory = get_gpu_memory_usage()
        
        # CPU power estimation
        cpu_power = get_cpu_power_usage()
        
        return {
            'timestamp': current_time,
            'timestamp_iso': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)),
            'cpu_percent': cpu_percent,
            'cpu_freq_mhz': cpu_freq.current if cpu_freq else 0,
            'cpu_count': cpu_count,
            'cpu_power_watts': cpu_power,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'memory_total_gb': memory_total_gb,
            'disk_percent': disk_percent,
            'disk_used_gb': disk_used_gb,
            'disk_total_gb': disk_total_gb,
            'gpu_power_watts': gpu_power,
            'gpu_utilization_percent': gpu_util,
            'gpu_memory_percent': gpu_memory,
        }
    except Exception as e:
        logging.warning(f"Error getting system metrics: {e}")
        return {}

def calculate_energy_consumption(power_data_points):
    """
    Calculate total energy consumption from power data points with timestamps.
    
    Args:
        power_data_points: List of dicts with 'timestamp' and power values
        
    Returns:
        Dict with total energy consumption in watt-hours and joules
    """
    if len(power_data_points) < 2:
        return {'total_energy_wh': 0, 'total_energy_joules': 0, 'duration_seconds': 0}
    
    # Sort by timestamp
    sorted_points = sorted(power_data_points, key=lambda x: x['timestamp'])
    
    total_gpu_energy = 0
    total_cpu_energy = 0
    total_duration = 0
    
    for i in range(1, len(sorted_points)):
        prev = sorted_points[i-1]
        curr = sorted_points[i]
        
        # Time interval in hours
        time_interval_hours = (curr['timestamp'] - prev['timestamp']) / 3600.0
        time_interval_seconds = curr['timestamp'] - prev['timestamp']
        
        # Average power during this interval
        avg_gpu_power = (prev.get('gpu_power_watts', 0) + curr.get('gpu_power_watts', 0)) / 2
        avg_cpu_power = (prev.get('cpu_power_watts', 0) + curr.get('cpu_power_watts', 0)) / 2
        
        # Energy = Power × Time
        gpu_energy = avg_gpu_power * time_interval_hours
        cpu_energy = avg_cpu_power * time_interval_hours
        
        total_gpu_energy += gpu_energy
        total_cpu_energy += cpu_energy
        total_duration += time_interval_seconds
    
    total_energy_wh = total_gpu_energy + total_cpu_energy
    total_energy_joules = total_energy_wh * 3600  # Convert watt-hours to joules
    
    return {
        'total_energy_wh': total_energy_wh,
        'total_energy_joules': total_energy_joules,
        'gpu_energy_wh': total_gpu_energy,
        'cpu_energy_wh': total_cpu_energy,
        'duration_seconds': total_duration,
        'duration_hours': total_duration / 3600
    }

# --------------------------------------------------------------------------------------
# Observation preprocessing
# --------------------------------------------------------------------------------------
class ObservationPreprocessor:
    """Handles preprocessing of environment observations for OpenPI."""
    
    def __init__(self, resize_size: int, env_manager=None):
        self.resize_size = resize_size
        self.env_manager = env_manager
        
    def preprocess_observations(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Preprocess observations for OpenPI inference."""
        # Define required observation keys
        required_obs_keys = [
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            # "robot0_gripper_qpos",
            # 'robot0_joint_pos_cos',
            # 'robot0_joint_pos_sin',
            # "gripper0_left_inner_finger",
            # "gripper0_right_inner_finger"
        ]
        
        # Ensure obs is a dict (like in working examples)
        if not isinstance(obs, dict):
            obs = self.env_manager.env.env._get_observations()
        
        # Verify all required keys are present
        if not all(k in obs for k in required_obs_keys):
            missing_keys = [k for k in required_obs_keys if k not in obs]
            raise KeyError(f"Missing required observation keys: {missing_keys}. Available: {list(obs.keys())}")
        
        # Process images
        img_obs = obs["agentview_image"]
        wrist_obs = obs["robot0_eye_in_hand_image"]
        
        # Rotate 180 degrees
        img = np.ascontiguousarray(img_obs[::-1, ::-1], dtype=np.uint8)
        wrist_img = np.ascontiguousarray(wrist_obs[::-1, ::-1], dtype=np.uint8)
        
        # Resize and pad
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, self.resize_size, self.resize_size))
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, self.resize_size, self.resize_size))

        joint_cos = obs['robot0_joint_pos_cos']
        joint_sin = obs['robot0_joint_pos_sin']
        joint_state = np.arctan2(joint_sin, joint_cos).astype(np.float32)

        left_finger_pos = self.env_manager.env.sim.data.body_xpos[
            self.env_manager.env.sim.model.body_name2id("gripper0_left_inner_finger")
        ]
        right_finger_pos = self.env_manager.env.sim.data.body_xpos[
            self.env_manager.env.sim.model.body_name2id("gripper0_right_inner_finger")
        ]
        gripper_width = float(np.linalg.norm(left_finger_pos - right_finger_pos))
        
        # Use the gripper width calculated from actual finger positions in sim
        eef_gripper = np.array([gripper_width], dtype=np.float32)   

        # State is the concatenation of joint state and gripper opening
        state = np.concatenate((joint_state, eef_gripper)).astype(np.float32)
        
        # Process state
        # eef_pos = obs.get('robot0_eef_pos', np.zeros(3, dtype=np.float32))
        # eef_quat = obs.get('robot0_eef_quat', np.array([0., 0., 0., 1.], dtype=np.float32))
        # eef_gripper = obs.get('robot0_gripper_qpos', np.zeros(2, dtype=np.float32))
        
        # # Convert quaternion to axis angle
        # eef_axis_angle = _quat2axisangle(eef_quat)
        # eef_state = np.concatenate((eef_pos, eef_axis_angle, eef_gripper)).astype(np.float32)
        
        return {
            "image": img,
            "wrist_image": wrist_img,
            "state": state,
            "raw_agentview": obs["agentview_image"]  # For video recording
        }


# --------------------------------------------------------------------------------------
# Main execution function
# --------------------------------------------------------------------------------------
def main(args: Args) -> None:
    """Runs a Robosuite environment controlled by an OpenPI policy server."""
    logging.info(f"Running Robosuite env '{args.env_name}' with OpenPI server at {args.host}:{args.port}")
    logging.info(f"Rendering mode: {args.render_mode}")
    # logging.info(f"Multi-config settings: random_block_placement={args.random_block_placement}, 
    # random_block_selection={args.random_block_selection}")
    
    # Initialize W&B
    settings = wandb.Settings(_stats_sampling_interval=args.log_every_n_seconds)
    group_name = f"multi_config_hanoi_{time.strftime('%Y%m%d-%H%M%S')}"
    
    # Setup environment manager
    env_manager = MultiConfigHanoiEnvironment(args)
    env_manager.setup()

    # Setup task manager
    task_manager = TaskManager(env_manager.tasks, args.planner_guided, 
                              args.time_based_progression, args.task_timeout)

    # Setup observation preprocessor
    obs_preprocessor = ObservationPreprocessor(args.resize_size, env_manager)
    
    # Log the mode being used
    if args.planner_guided:
        logging.info(f"Running in PLANNER GUIDED MODE")
        if args.time_based_progression:
            logging.info(
                f"Subtasks will progress automatically after \
                {args.task_timeout} steps timeout"
            )
        else:
            logging.info(
                f"Subtasks will progress automatically as they are \
                achieved, but episode will terminate early if any subtask \
                takes longer than {args.task_timeout} steps"
            )
    else:
        logging.info("Running in END-TO-END MODE")
        logging.info(f"Using fixed prompt: {task_manager.single_prompt} \
            for all steps")
        if not args.time_based_progression:
            logging.info(f"Episode will terminate early if any task takes \
                longer than {args.task_timeout} steps")    
    
    # Setup websocket client
    logging.info(
        f"Connecting to OpenPI server at ws://{args.host}:{args.port}...")
    client = _websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port)
    
    # Run episodes
    for ep in range(args.episodes):
        logging.info(f"\n=========  EPISODE {ep+1}/{args.episodes}  =========")
        
        # Initialize W&B run for this episode
        run_name = args.generate_wandb_run_name(ep)
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=group_name,
            config=dataclasses.asdict(args),
            settings=settings,
        )
        
        # Setup video recording
        video_filename = args.generate_video_filename(ep)
        video_full_path = pathlib.Path(args.video_out_path) / video_filename
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
        frames = []
        
        # Setup full resolution video recording if enabled
        hd_frames = []
        hd_wrist_frames = []
        if args.save_full_res_video:
            full_res_filename = video_filename.replace('.mp4', '_hd.mp4')
            full_res_wrist_filename = video_filename.replace('.mp4', '_wrist_hd.mp4')
            full_res_video_path = pathlib.Path(args.video_out_path) / full_res_filename
            full_res_wrist_video_path = pathlib.Path(args.video_out_path) / full_res_wrist_filename
        
        # Reset environment
        try:
            obs = env_manager.reset()
            # Update task manager with new tasks and reset episode state
            task_manager.tasks = env_manager.tasks
            task_manager.reset_episode()
        except Exception as e:
            logging.error(f"Failed to reset environment: {e}")
            wandb_run.finish()
            continue
            
        # Episode variables
        action_plan = collections.deque()
        t = 0
        global_step = 0  # Initialize global step counter
        start_episode_time = time.time()
        task_manager.episode_start_step = t  # Initialize episode start step
        
        # Power monitoring data for energy calculation
        power_data_points = []
        
        # Debug episode start
        logging.info(f"Starting episode {ep+1} with {len(task_manager.tasks)} tasks")
        logging.info(f"Task manager state: current_task_idx={task_manager.current_task_idx}, \
            use_sequential_tasks={task_manager.use_sequential_tasks}")
        if task_manager.tasks:
            logging.info(f"First task: {task_manager.tasks[0]['prompt']}")
            if len(task_manager.tasks) > 6:
                logging.info(f"Task 7: {task_manager.tasks[6]['prompt']}")
        
        # Initial wandb logging
        wandb.log({"score": task_manager.episode_score}, step=global_step)
        
        # Log initial system metrics
        system_metrics = get_system_metrics()
        if system_metrics:
            wandb.log(system_metrics, step=global_step)
            power_data_points.append(system_metrics)
        
        logging.info("Starting episode...")
        
        # Main episode loop
        while t < args.horizon:
            loop_start_time = time.time()
            
            # Skip initial steps if needed
            if t < args.skip_steps:
                action = np.zeros(env_manager.env.action_dim)
                if env_manager.env.robots[0].gripper.dof > 0:
                    action[6] = -1.0  # Keep gripper closed
                step_result = env_manager.env.step(action)
                if len(step_result) == 5:
                    obs, reward, done, truncated, info = step_result
                else:
                    obs, reward, done, info = step_result
                t += 1
                if args.render_mode == 'human':
                    env_manager.env.render()
                continue
            
            # Preprocess observations
            try:
                processed_obs = obs_preprocessor.preprocess_observations(obs)
                frames.append(processed_obs["image"])
                
                # Capture full resolution frames for video recording if enabled
                if args.save_full_res_video:
                    full_res_frame = env_manager.env.sim.render(
                        width=args.full_res_width, 
                        height=args.full_res_height, 
                        camera_name="agentview"
                    )
                    full_res_wrist_frame = env_manager.env.sim.render(
                        width=args.full_res_width, 
                        height=args.full_res_height, 
                        camera_name="robot0_eye_in_hand"
                    )
                    hd_frames.append(full_res_frame)
                    hd_wrist_frames.append(full_res_wrist_frame)
                    
            except Exception as e:
                logging.error(f"Error during preprocessing at step {t}: {e}")
                break
            
            # Get action from OpenPI server if needed
            if not action_plan:
                element = {
                    "observation/image": processed_obs["image"],
                    "observation/wrist_image": processed_obs["wrist_image"],
                    "observation/state": processed_obs["state"],
                    "prompt": task_manager.get_current_prompt(),
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
                        action_chunk = np.zeros((args.replan_steps, env_manager.env.action_dim))
                        if env_manager.env.robots[0].gripper.dof > 0:
                            action_chunk[:, 6] = -1.0
                    
                    # Ensure chunk shape is reasonable
                    if len(action_chunk.shape) != 2 or action_chunk.shape[1] != env_manager.env.action_dim:
                        logging.error(f"Action chunk shape mismatch. Expected (~N, {env_manager.env.action_dim}), Got: {action_chunk.shape}")
                        break
                    
                    action_plan.extend(action_chunk[:args.replan_steps])
                    
                except Exception as e:
                    logging.error(f"Failed to get inference from server: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # Get next action
            if action_plan:
                action = action_plan.popleft()
            else:
                logging.warning("Action plan empty after inference attempt. Sending zero action.")
                action = np.zeros(env_manager.env.action_dim)
                if env_manager.env.robots[0].gripper.dof > 0:
                    action[6] = -1.0
            
            # Step environment
            try:
                if not isinstance(action, np.ndarray):
                    action = np.array(action)
                if action.shape[0] != env_manager.env.action_dim:
                    logging.error(f"Action dimension mismatch before step. Policy: {action.shape[0]}, Env: {env_manager.env.action_dim}")
                    break
                    
                action = action.tolist()
                action[3] = 0  # Zero out rotation components
                action[4] = 0
                action[5] = 0
                
                step_result = env_manager.env.step(action)
                if len(step_result) == 5:
                    obs, reward, done, truncated, info = step_result
                else:
                    obs, reward, done, info = step_result
                t += 1
                global_step += 1
                wandb.log({"reward": reward}, step=global_step)
                
            except Exception as e:
                logging.error(f"Error during env.step at step {t}: {e}")
                break
            
            # Check task completion
            if task_manager.check_task_completion(t):
                # Log task completion (always done regardless of mode)
                # Note: current_task_idx was already incremented in check_task_completion
                completed_task_idx = task_manager.current_task_idx - 1
                wandb.log(
                    {f"task_{completed_task_idx + 1}_completions": task_manager.task_totals[completed_task_idx]},
                    step=global_step,
                )
                wandb.log(
                    {f"task_{completed_task_idx + 1}_completed_step": t},
                    step=global_step,
                )
                wandb_run.summary[f"task{completed_task_idx + 1}_completed_step"] = t
                
                # Clear action plan to force replanning when task completes
                action_plan.clear()
                
                if task_manager.is_episode_complete():
                    # Log final score when episode completes
                    wandb.log({"score": task_manager.episode_score}, step=global_step)
                    logging.info(f"🎉 Episode completed! Final score: {task_manager.episode_score}")
                    break
            
            # Check task timeout (always check, behavior depends on time_based_progression setting)
            if task_manager.check_task_timeout(t):
                # Log task timeout
                wandb.log(
                    {f"task_{task_manager.current_task_idx}_timeout_step": t},
                    step=global_step,
                )
                wandb_run.summary[f"task{task_manager.current_task_idx}_timeout_step"] = t
                
                # Clear action plan to force replanning when task times out
                action_plan.clear()
                
                # Always break on timeout - either episode complete or early termination
                break
            
            # Update score and system metrics
            wandb.log({"score": task_manager.episode_score}, step=global_step)
            
            # Log system metrics every 10 steps to avoid too much overhead
            if t % 10 == 0:
                system_metrics = get_system_metrics()
                if system_metrics:
                    wandb.log(system_metrics, step=global_step)
                    # Store power data for energy calculation
                    power_data_points.append(system_metrics)
            
            # Render if requested
            if args.render_mode == 'human':
                env_manager.env.render()
            
            # Debug logging
            if t % 10 == 0 and hasattr(env_manager.detector_simple, "status"):
                logging.debug(f"Detector status @step {t}: {env_manager.detector_simple.status()}")
                if not task_manager.is_episode_complete() and task_manager.current_task_idx < len(env_manager.tasks):
                    current_task_done = env_manager.tasks[task_manager.current_task_idx]['done']()
                    logging.debug(f"Current task '{task_manager.get_current_prompt()}' predicate = {current_task_done}")
                    
                    # Additional debugging for task completion
                    if current_task_done:
                        logging.info(f"🎯 Task {task_manager.current_task_idx + 1} completion detected at step {t}: {env_manager.tasks[task_manager.current_task_idx]['prompt']}")
            
            # Regular logging
            if t % 50 == 0:
                if not task_manager.use_sequential_tasks:
                    # In single prompt mode, show progress information
                    completed_tasks = sum(task_manager.task_completed_this_episode)
                    total_tasks = len(task_manager.tasks) if task_manager.tasks else 0
                    current_task_num = task_manager.current_task_idx + 1 if task_manager.current_task_idx < total_tasks else total_tasks
                    steps_on_current_task = t - task_manager.task_start_step
                    logging.info(f"Step: {t}/{args.horizon}, Reward: {reward:.3f}, \
                        Completed: {completed_tasks}/{total_tasks}, Working on task {current_task_num}, \
                        Steps on current task: {steps_on_current_task}, Loop Time: {time.time() - loop_start_time:.4f}s")
                else:
                    logging.info(f"Step: {t}/{args.horizon}, Reward: {reward:.3f}, Loop Time: {time.time() - loop_start_time:.4f}s")
            else:
                logging.debug(f"Step: {t}/{args.horizon}, Reward: {reward:.3f}, Loop Time: {time.time() - loop_start_time:.4f}s")
        
        # End of episode
        episode_duration = time.time() - start_episode_time
        logging.info(f"Episode finished after {t} steps. Duration: {episode_duration:.2f}s. Score: {task_manager.episode_score}")
        
        # Final wandb logging for episode end
        wandb.log({"score": task_manager.episode_score}, step=global_step)
        wandb.log({"episode_duration": episode_duration}, step=global_step)
        wandb.log({"episode_steps": t}, step=global_step)
        
        # Log final system metrics
        system_metrics = get_system_metrics()
        if system_metrics:
            wandb.log(system_metrics, step=global_step)
            power_data_points.append(system_metrics)
        
        # Calculate and log energy consumption for this episode
        if len(power_data_points) >= 2:
            energy_stats = calculate_energy_consumption(power_data_points)
            logging.info(f"⚡ Episode energy consumption: {energy_stats['total_energy_wh']:.3f} Wh ({energy_stats['total_energy_joules']:.1f} J)")
            logging.info(f"   GPU: {energy_stats['gpu_energy_wh']:.3f} Wh, CPU: {energy_stats['cpu_energy_wh']:.3f} Wh")
            logging.info(f"   Duration: {energy_stats['duration_hours']:.3f} hours")
            
            # Log energy stats to wandb
            wandb.log({
                'episode_energy_wh': energy_stats['total_energy_wh'],
                'episode_energy_joules': energy_stats['total_energy_joules'],
                'episode_gpu_energy_wh': energy_stats['gpu_energy_wh'],
                'episode_cpu_energy_wh': energy_stats['cpu_energy_wh'],
                'episode_duration_hours': energy_stats['duration_hours']
            }, step=global_step)
        
        # Log completion summary (always useful regardless of mode)
        completion_summary = task_manager.get_completion_summary()
        logging.info(f"Task completion summary: {completion_summary['completed_tasks']}/{completion_summary['total_tasks']} tasks completed ({completion_summary['completion_rate']:.1f}%)")
        
        # Log progress information for single prompt mode
        if not task_manager.use_sequential_tasks:
            current_task_num = task_manager.current_task_idx + 1 if task_manager.current_task_idx < len(task_manager.tasks) else len(task_manager.tasks)
            steps_on_current_task = t - task_manager.task_start_step
            logging.info(f"Progress tracking: Episode started at step {task_manager.episode_start_step}, last progress at step {task_manager.last_progress_step}, current task: {current_task_num}, steps on current task: {steps_on_current_task}")
        
        # Save videos
        if frames:
            logging.info(f"Saving model observation video ({len(frames)} frames) to {video_full_path}...")
            try:
                imageio.mimwrite(str(video_full_path), frames, fps=env_manager.env.control_freq)
                logging.info("Model observation video saved.")
            except Exception as e:
                logging.error(f"Failed to save model observation video: {e}")
        else:
            logging.warning("No frames collected for model observation video.")
            
        # Save full resolution videos if enabled
        if args.save_full_res_video and hd_frames:
            logging.info(f"Saving full resolution agentview video ({len(hd_frames)} frames) to {full_res_video_path}...")
            try:
                # Flip frames vertically to fix upside-down issue (same as in dataset_making)
                flipped_frames = [np.flipud(frame) for frame in hd_frames]
                imageio.mimwrite(str(full_res_video_path), flipped_frames, fps=env_manager.env.control_freq, macro_block_size=None)
                logging.info("Full resolution agentview video saved.")
            except Exception as e:
                logging.error(f"Failed to save full resolution agentview video: {e}")
                
            logging.info(f"Saving full resolution wrist video ({len(hd_wrist_frames)} frames) to {full_res_wrist_video_path}...")
            try:
                # Flip frames vertically to fix upside-down issue
                flipped_wrist_frames = [np.flipud(frame) for frame in hd_wrist_frames]
                imageio.mimwrite(str(full_res_wrist_video_path), flipped_wrist_frames, fps=env_manager.env.control_freq, macro_block_size=None)
                logging.info("Full resolution wrist video saved.")
            except Exception as e:
                logging.error(f"Failed to save full resolution wrist video: {e}")
        elif args.save_full_res_video:
            logging.warning("Full resolution video recording enabled but no frames collected.")
        
        # W&B summary and finish
        for idx, count in enumerate(task_manager.task_totals, 1):
            wandb_run.summary[f"step{idx}_completions"] = count
        
        # Log overall completion metrics
        completion_summary = task_manager.get_completion_summary()
        wandb_run.summary["total_tasks_completed"] = completion_summary["completed_tasks"]
        wandb_run.summary["total_tasks_available"] = completion_summary["total_tasks"]
        wandb_run.summary["completion_rate_percent"] = completion_summary["completion_rate"]
        
        wandb_run.finish()
    
    # Cleanup
    env_manager.close()
    logging.info("All episodes completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)#, format='%(asctime)s %(levelname)-8s %(message)s')
    tyro.cli(main)
