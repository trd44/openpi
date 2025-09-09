"""
Script to use the openpi model in the Towers of Hanoi Robosuite environment.
"""
import collections
import dataclasses
import logging
import math
import pathlib
import os
import time
from typing import List, Dict, Any, Optional, Callable

import imageio
import numpy as np
import wandb
import tyro
import robosuite as suite
from robosuite import load_controller_config

# Assuming openpi_client is installed in the environment
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from hanoi_detectors import PandaHanoiDetector

# --------------------------------------------------------------------------------------
# Constants and Configuration
# --------------------------------------------------------------------------------------
# Task configuration
COLOR2OBJ = {"blue": "cube1", "red": "cube2", "green": "cube3", "yellow": "cube4"}
AREA2PEG = {"left": "peg1", "middle": "peg2", "right": "peg3"}

# --------------------------------------------------------------------------------------
# Simple, self‑contained predicate detector + task builder for long‑horizon execution
# --------------------------------------------------------------------------------------
class SimpleHanoiDetector:
    """
    Extremely lightweight state detector that relies only on MuJoCo body poses
    already present in the Robosuite simulation.  It lets us verify high‑level
    goals (block in area, block stacked, block lifted) without pulling in the
    much heavier PandaHanoiDetector / tasks.py dependencies.
    """
    XY_THRESH = 0.05           # how close XY coordinates must be to count as "aligned"
    Z_STACK_THRESH = 0.02      # minimum vertical gap to consider one cube on top of another
    LIFT_DELTA = 0.08          # accommodate peg height so "lifted" is stricter
    
    def __init__(self, env):
        import numpy as _np
        self.np = _np
        self.env = env

        # Mapping from human‑readable color / area names to MuJoCo body names
        self.cube_bodies = {
            "blue":  "cube1_main",
            "red":   "cube2_main",
            "green": "cube3_main",
        }
        self.peg_bodies = {
            "left":   "peg1_main",
            "middle": "peg2_main",
            "right":  "peg3_main",
        }

        # Cache peg XY centres
        self.peg_xy = {
            k: self._body_pos(v)[:2] for k, v in self.peg_bodies.items()
        }

        # Remember each cube's initial Z so we can later tell if it was lifted
        self.cube_base_z = {c: self._body_pos(b)[2]
                            for c, b in self.cube_bodies.items()}

    def status(self):
        """
        Return a lightweight dict summarising cube XY, Z and lifted flag
        for quick debugging prints.
        """
        info = {}
        for c in self.cube_bodies:
            pos = self._cube_pos(c)
            info[c] = {
                "xy": [float(pos[0]), float(pos[1])],
                "z":  float(pos[2]),
                "lifted": bool(self.lifted(c)),
                # distances to each area for convenience
                "d_left":  float(self.np.linalg.norm(pos[:2] - self.peg_xy["left"])),
                "d_mid":   float(self.np.linalg.norm(pos[:2] - self.peg_xy["middle"])),
                "d_right": float(self.np.linalg.norm(pos[:2] - self.peg_xy["right"])),
            }
        return info

    # ------------------------------------------------------------------ utilities
    def _body_pos(self, body_name):
        bid = self.env.sim.model.body_name2id(body_name)
        return self.env.sim.data.body_xpos[bid]

    def _cube_pos(self, color):
        return self._body_pos(self.cube_bodies[color])

    # ------------------------------------------------------------------ predicates
    def lifted(self, color):
        """Returns True if the cube is > LIFT_DELTA above its initial height."""
        return self._cube_pos(color)[2] > self.cube_base_z[color] + self.LIFT_DELTA

    def in_area(self, color, area):
        """Cube's XY very close to the peg XY."""
        cube_xy = self._cube_pos(color)[:2]
        return self.np.linalg.norm(cube_xy - self.peg_xy[area]) < self.XY_THRESH

    def stacked(self, top_color, bottom_color):
        """`top_color` cube sits on (and roughly centred over) `bottom_color` cube."""
        top = self._cube_pos(top_color)
        bot = self._cube_pos(bottom_color)
        xy_close = self.np.linalg.norm(top[:2] - bot[:2]) < self.XY_THRESH
        z_above  = (top[2] - bot[2]) > self.Z_STACK_THRESH
        return xy_close and z_above


# -----------------------------------------------------------------------------
# Task sequence builder using PandaHanoiDetector groundings
# -----------------------------------------------------------------------------
def build_task_sequence_g(
        det_ground: PandaHanoiDetector, 
        det_simple: SimpleHanoiDetector,
    ) -> List[Dict[str, Any]]:
    """
    Uses PandaHanoiDetector.get_groundings().  Each lambda re‑queries groundings
    so it always reflects the latest simulation state.
    """
    def g():
        # fresh snapshot every call
        return det_ground.get_groundings(as_dict=True,
                                         binary_to_float=False,
                                         return_distance=False)

    return [
        # 1  Pick blue
        {"prompt": "Pick the blue block.",
         "done": lambda: bool(g().get("grasped(cube1)", False)) and det_simple.lifted("blue")},

        # 2  Place blue on right peg
        {"prompt": "Place the blue block in the right area.",
         "done": lambda: bool(g().get("on(cube1,peg3)", False))},

        # 3  Pick red
        {"prompt": "Pick the red block.",
         "done": lambda: bool(g().get("grasped(cube2)", False)) and det_simple.lifted("red")},

        # 4  Place red on middle peg
        {"prompt": "Place the red block in the middle area.",
         "done": lambda: bool(g().get("on(cube2,peg2)", False))},

        # 5  Pick blue
        {"prompt": "Pick the blue block.",
         "done": lambda: bool(g().get("grasped(cube1)", False)) and det_simple.lifted("blue")},

        # 6  Stack blue on red
        {"prompt": "Place the blue block on top of the red block.",
         "done": lambda: bool(g().get("on(cube1,cube2)", False))},

        # 7  Pick green
        {"prompt": "Pick the green block.",
         "done": lambda: bool(g().get("grasped(cube3)", False)) and det_simple.lifted("green")},

        # 8  Place green on right peg
        {"prompt": "Place the green block in the right area.",
         "done": lambda: bool(g().get("on(cube3,peg3)", False))},

        # 9
        {"prompt": "Pick the blue block.",
         "done": lambda: bool(g().get("grasped(cube1)", False)) and det_simple.lifted("blue")},

        # 10  Move blue to left peg
        {"prompt": "Place the blue block in the left area.",
         "done": lambda: bool(g().get("on(cube1,peg1)", False))},

        # 11  Pick red
        {"prompt": "Pick the red block.",
         "done": lambda: bool(g().get("grasped(cube2)", False)) and det_simple.lifted("red")},

        # 12  Stack red on green
        {"prompt": "Place the red block on top of the green block.",
         "done": lambda: bool(g().get("on(cube2,cube3)", False))},

        # 13
        {"prompt": "Pick the blue block.",
         "done": lambda: bool(g().get("grasped(cube1)", False)) and det_simple.lifted("blue")},

        # 14  Stack blue on red
        {"prompt": "Place the blue block on top of the red block.",
         "done": lambda: bool(g().get("on(cube1,cube2)", False))},
    ]


# --------------------------------------------------------------------------------------
# Configuration dataclass
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    """Arguments for running Robosuite with OpenPI Websocket Policy"""
    # --- Server Connection ---
    host: str = "127.0.0.1"         # Hostname of the OpenPI policy server
    port: int = 8000                # Port of the OpenPI policy server

    # --- Policy Interaction ---
    resize_size: int = 224           # Target size for image resizing (must match model training)
    replan_steps: int = 50           # Number of steps per action chunk from policy server
    use_sequential_tasks: bool = True # If True, use sequential task prompts from build_task_sequence_g; if False, use single prompt "Play Towers of Hanoi."
    time_based_progression: bool = False # If True, advance to next task after task_timeout steps regardless of completion
    task_timeout: int = 500          # Number of steps to wait before timing out a task (only used if time_based_progression=True)

    # --- Robosuite Environment ---
    env_name: str = "Hanoi" 
    robots: str = "Panda"           # Robot model to use
    controller: str = "OSC_POSE"    # Robosuite controller name
    horizon: int = 100050             # Max steps per episode
    skip_steps: int = 50            # Number of initial steps to skip (e.g., wait for objects to settle)

    # --- Rendering & Video ---
    render_mode: str = "headless"    # Rendering mode: 'headless' (save video) or 'human' (live view via X11)
    video_out_path: str = "data/robosuite_videos" # Directory to save videos
    camera_names: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
        ) # Cameras for observation/video
    camera_height: int = 256 # Rendered camera height (before potential resize)
    camera_width: int = 256 # Rendered camera width (before potential resize)
    required_cameras: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
        ) # Required cameras for OpenPI preprocessing (will be added to camera_names if missing)

    # --- Misc ---
    seed: int = 3           #: Random seed
    episodes: int = 50      #: How many episodes to run back-to-back

    # --- Logging ---
    wandb_project: str = "TEST hanoi 300 subtasks" #: W&B project name
    log_every_n_seconds: float = 0.5 #: Logging interval for W&B settings
    
    def generate_video_filename(self, episode: int) -> str:
        """Generate video filename based on current arguments and episode number."""
        return f"{self.env_name}_seed{self.seed}_ep{episode+1}_{time.strftime('%H%M%S')}.mp4"
    
    def generate_wandb_run_name(self, episode: int) -> str:
        """Generate W&B run name based on current arguments and episode number."""
        return f"{self.env_name}_seed{self.seed}_ep{episode+1}_{time.strftime('%H%M%S')}"
    
    def get_required_cameras(self) -> List[str]:
        """Get the list of required cameras for OpenPI preprocessing."""
        return self.required_cameras


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# Environment setup and management
# --------------------------------------------------------------------------------------
class HanoiEnvironment:
    """Manages the Robosuite Hanoi environment setup and operations."""
    
    def __init__(self, args: Args):
        self.args = args
        self.env = None
        self.detector_simple = None
        self.detector_ground = None
        self.tasks = None
        
    def setup(self) -> None:
        """Initialize the environment and detectors."""
        np.random.seed(self.args.seed)
        
        # Load controller configuration
        controller_config = load_controller_config(default_controller=self.args.controller)
        
        # Setup rendering
        has_renderer = (self.args.render_mode == 'human')
        has_offscreen = True  # Need offscreen renderer for observations even if rendering on-screen
        
        # Verify required cameras
        for cam in self.args.get_required_cameras():
            if cam not in self.args.camera_names:
                logging.warning(f"Required camera '{cam}' not in requested camera_names. Adding it.")
                self.args.camera_names.append(cam)
        
        # Create environment
        self.env = suite.make(
            env_name=self.args.env_name,
            robots=self.args.robots,
            controller_configs=controller_config,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen,
            control_freq=20,  # Robosuite default
            horizon=self.args.horizon,
            use_object_obs=True,  # Get state observations
            use_camera_obs=True,  # Get camera observations
            camera_names=self.args.camera_names,
            camera_heights=self.args.camera_height,
            camera_widths=self.args.camera_width,
            render_camera="agentview" if has_renderer else None,
            ignore_done=True,  # Let horizon end the episode
            hard_reset=False,  # Faster resets can sometimes be unstable, switch if needed
            random_block_placement=False,
            random_block_selection=False,
        )
        
        # Seed environment
        try:
            self.env.seed(self.args.seed)
            logging.info(f"Environment seeded with {self.args.seed} using env.seed()")
        except AttributeError:
            logging.warning("env.seed() method not found. Seeding might rely on env.reset(seed=...) instead.")
        except Exception as e:
            logging.error(f"Error calling env.seed(): {e}")
        
        # Initialize detectors
        self.detector_simple = SimpleHanoiDetector(self.env)
        self.detector_ground = PandaHanoiDetector(self.env)
        self.tasks = build_task_sequence_g(self.detector_ground, self.detector_simple)
        
    def reset(self):
        """Reset the environment and reinitialize detectors."""
        try:
            obs = self.env.reset()
            # Reinitialize detectors after reset
            self.detector_simple = SimpleHanoiDetector(self.env)
            self.detector_ground = PandaHanoiDetector(self.env)
            self.tasks = build_task_sequence_g(self.detector_ground, self.detector_simple)
            return obs
        except Exception as e:
            logging.error(f"Failed to reset environment: {e}")
            if "unexpected keyword argument 'seed'" in str(e):
                logging.error("Environment reset failed likely because it expects env.seed() instead of env.reset(seed=...).")
            raise
    
    def close(self):
        """Close the environment."""
        if self.env:
            self.env.close()
            logging.info("Environment closed.")


# -----------------------------------------------------------------------------
# Task management
# -----------------------------------------------------------------------------
class TaskManager:
    """Manages task progression and completion tracking."""
    
    def __init__(self, tasks: List[Dict[str, Any]], use_sequential_tasks: bool = False, 
                 time_based_progression: bool = False, task_timeout: int = 200):
        self.tasks = tasks
        self.use_sequential_tasks = use_sequential_tasks
        self.time_based_progression = time_based_progression
        self.task_timeout = task_timeout
        self.current_task_idx = 0
        self.current_prompt = tasks[0]["prompt"] if tasks and use_sequential_tasks else "Play Towers of Hanoi."
        self.task_totals = [0] * len(self.tasks) if tasks else []
        self.episode_score = 0
        self.task_completed_this_episode = [False] * len(self.tasks) if tasks else []
        self.task_start_step = 0  # Track when current task started
        
    def reset_episode(self):
        """Reset episode-specific counters."""
        if self.tasks:  # Always reset task tracking if tasks exist
            self.task_totals = [0] * len(self.tasks)
            self.task_completed_this_episode = [False] * len(self.tasks)
        if self.use_sequential_tasks:
            self.current_task_idx = 0
            self.current_prompt = self.tasks[0]["prompt"] if self.tasks else ""
        self.episode_score = 0
        self.task_start_step = 0
        
    def check_task_timeout(self, step: int) -> bool:
        """Check if current task has timed out and either advance or terminate based on mode."""
        if not self.tasks:
            return False
        
        if self.current_task_idx >= len(self.tasks):
            return False
        
        # Check if we've exceeded the timeout for the current task
        steps_on_current_task = step - self.task_start_step
        if steps_on_current_task >= self.task_timeout:
            if self.time_based_progression:
                # In time-based progression mode, advance to next task
                logging.info(f"⏰ Task {self.current_task_idx + 1}/{len(self.tasks)} timed out after {steps_on_current_task} steps: {self.tasks[self.current_task_idx]['prompt']}")
                
                # Advance to next task
                self.current_task_idx += 1
                if self.current_task_idx >= len(self.tasks):
                    logging.info("🎉 All tasks completed (via timeout) – ending episode early.")
                    return True
                    
                if self.use_sequential_tasks:
                    # In sequential mode, change the prompt
                    self.current_prompt = self.tasks[self.current_task_idx]["prompt"]
                    logging.info(f"⏭️ Starting next task {self.current_task_idx + 1}/{len(self.tasks)}: {self.current_prompt}")
                else:
                    # In single prompt mode, keep the same prompt but log task progression
                    logging.info(f"📊 Task {self.current_task_idx} timed out in single prompt mode - continuing with same prompt")
                
                # Reset task start step for the new task
                self.task_start_step = step
                return True
            else:
                # In completion-only mode, terminate the episode early
                logging.info(f"⏰ Task {self.current_task_idx + 1}/{len(self.tasks)} timed out after {steps_on_current_task} steps: {self.tasks[self.current_task_idx]['prompt']}")
                logging.info("🛑 Episode terminated early due to task timeout (time_based_progression=False)")
                return True
        
        return False

    def check_task_completion(self, step: int) -> bool:
        """Check if current task is completed and advance if so."""
        if not self.tasks:
            return False
        
        task_completed = False
        
        # Always check completion in sequential order, regardless of mode
        if (self.current_task_idx < len(self.tasks) and 
            self.tasks[self.current_task_idx]["done"]() and 
            not self.task_completed_this_episode[self.current_task_idx]):
            
            # Mark task as completed for this episode
            self.task_completed_this_episode[self.current_task_idx] = True
            self.task_totals[self.current_task_idx] += 1
            self.episode_score += 1
            
            logging.info(f"✅ Completed task {self.current_task_idx + 1}/{len(self.tasks)} on step {step} : {self.tasks[self.current_task_idx]['prompt']}")
            
            # Always advance to next task to maintain sequential order
            self.current_task_idx += 1
            if self.current_task_idx >= len(self.tasks):
                logging.info("🎉 All tasks completed – ending episode early.")
                return True
                
            if self.use_sequential_tasks:
                # In sequential mode, change the prompt
                self.current_prompt = self.tasks[self.current_task_idx]["prompt"]
                logging.info(f"⏭️ Starting next task {self.current_task_idx + 1}/{len(self.tasks)}: {self.current_prompt}")
            else:
                # In single prompt mode, keep the same prompt but log task progression
                logging.info(f"📊 Task {self.current_task_idx} completed in single prompt mode - continuing with same prompt")
            
            # Reset task start step for the new task
            self.task_start_step = step
            task_completed = True
            
        return task_completed
    
    def get_current_prompt(self) -> str:
        """Get the current task prompt."""
        return self.current_prompt
    
    def is_episode_complete(self) -> bool:
        """Check if all tasks are completed."""
        if not self.use_sequential_tasks:
            return False  # Never complete in single prompt mode
        return self.current_task_idx >= len(self.tasks)
    
    def should_clear_action_plan(self) -> bool:
        """Determine if action plan should be cleared (only in sequential mode when task completes)."""
        return self.use_sequential_tasks and self.is_episode_complete()
    
    def get_completion_summary(self) -> Dict[str, Any]:
        """Get a summary of task completion status for logging/monitoring."""
        if not self.tasks:
            return {"total_tasks": 0, "completed_tasks": 0, "completion_rate": 0.0}
        
        completed = sum(self.task_totals)
        total = len(self.tasks)
        completion_rate = (completed / total) * 100 if total > 0 else 0.0
        
        return {
            "total_tasks": total,
            "completed_tasks": completed,
            "completion_rate": completion_rate,
            "task_details": {f"task_{i+1}": count for i, count in enumerate(self.task_totals)}
        }


# --------------------------------------------------------------------------------------
# Observation preprocessing
# --------------------------------------------------------------------------------------
class ObservationPreprocessor:
    """Handles preprocessing of environment observations for OpenPI."""
    
    def __init__(self, resize_size: int):
        self.resize_size = resize_size
        
    def preprocess_observations(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Preprocess observations for OpenPI inference."""
        # Define required observation keys
        required_obs_keys = [
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos"
        ]
        
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
        
        # Process state
        eef_pos = obs.get('robot0_eef_pos', np.zeros(3, dtype=np.float32))
        eef_quat = obs.get('robot0_eef_quat', np.array([0., 0., 0., 1.], dtype=np.float32))
        eef_gripper = obs.get('robot0_gripper_qpos', np.zeros(2, dtype=np.float32))
        
        # Convert quaternion to axis angle
        eef_axis_angle = _quat2axisangle(eef_quat)
        eef_state = np.concatenate((eef_pos, eef_axis_angle, eef_gripper)).astype(np.float32)
        
        return {
            "image": img,
            "wrist_image": wrist_img,
            "state": eef_state,
            "raw_agentview": obs["agentview_image"]  # For video recording
        }


# --------------------------------------------------------------------------------------
# Main execution function
# --------------------------------------------------------------------------------------
def run_robosuite_with_openpi(args: Args) -> None:
    """Runs a Robosuite environment controlled by an OpenPI policy server."""
    logging.info(f"Running Robosuite env '{args.env_name}' with OpenPI server at {args.host}:{args.port}")
    logging.info(f"Rendering mode: {args.render_mode}")
    
    # Initialize W&B
    settings = wandb.Settings(_stats_sampling_interval=args.log_every_n_seconds)
    group_name = f"inference_experiments_{time.strftime('%Y%m%d-%H%M%S')}"
    
    # Setup environment
    env_manager = HanoiEnvironment(args)
    env_manager.setup()
    
    # Setup task manager
    task_manager = TaskManager(env_manager.tasks, args.use_sequential_tasks, 
                              args.time_based_progression, args.task_timeout)
    
    # Log the mode being used
    if args.use_sequential_tasks:
        logging.info(f"Running in SEQUENTIAL TASK MODE with {len(env_manager.tasks)} tasks")
        if args.time_based_progression:
            logging.info(f"Tasks will progress automatically as goals are achieved OR after {args.task_timeout} steps timeout")
        else:
            logging.info(f"Tasks will progress automatically as goals are achieved, but episode will terminate early if any task takes longer than {args.task_timeout} steps")
    else:
        logging.info("Running in SINGLE PROMPT MODE")
        logging.info("Using fixed prompt: 'Play Towers of Hanoi.' for all steps")
        if not args.time_based_progression:
            logging.info(f"Episode will terminate early if any task takes longer than {args.task_timeout} steps")
    
    # Setup observation preprocessor
    obs_preprocessor = ObservationPreprocessor(args.resize_size)
    
    # Setup websocket client
    logging.info(f"Connecting to OpenPI server at ws://{args.host}:{args.port}...")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    
    # Run episodes
    for ep in range(args.episodes):
        logging.info(f"\n==========  EPISODE {ep+1}/{args.episodes}  ==========")
        
        # Initialize W&B run for this episode
        run_name = args.generate_wandb_run_name(ep)
        wandb_run = wandb.init(
            project="openpi_hanoi_300_inference_rgb",
            name=run_name,
            group=group_name,
            config=dataclasses.asdict(args),
            settings=settings,
        )
        
        # Reset episode state
        task_manager.reset_episode()
        global_step = 0
        wandb.log({"score": task_manager.episode_score}, step=global_step)
        
        # Setup video recording
        video_filename = args.generate_video_filename(ep)
        video_full_path = pathlib.Path(args.video_out_path) / video_filename
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
        frames = []
        
        # Reset environment
        try:
            obs = env_manager.reset()
        except Exception as e:
            logging.error(f"Failed to reset environment: {e}")
            wandb_run.finish()
            continue
            
        # Episode variables
        action_plan = collections.deque()
        t = 0
        start_episode_time = time.time()
        logging.info("Starting episode...")
        
        # Main episode loop
        while t < args.horizon:
            loop_start_time = time.time()
            
            # Skip initial steps if needed
            if t < args.skip_steps:
                action = np.zeros(env_manager.env.action_dim)
                if env_manager.env.robots[0].gripper.dof > 0:
                    action[6] = -1.0  # Keep gripper closed
                obs, reward, done, info = env_manager.env.step(action)
                t += 1
                if args.render_mode == 'human':
                    env_manager.env.render()
                continue
            
            # Preprocess observations
            try:
                processed_obs = obs_preprocessor.preprocess_observations(obs)
                frames.append(processed_obs["image"])
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
                
                obs, reward, done, info = env_manager.env.step(action)
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
                
                # Clear action plan to force replanning (only in sequential mode)
                if task_manager.use_sequential_tasks:
                    action_plan.clear()
                
                if task_manager.is_episode_complete():
                    break
            
            # Check task timeout (always check, behavior depends on time_based_progression setting)
            if task_manager.check_task_timeout(t):
                # Log task timeout
                wandb.log(
                    {f"task_{task_manager.current_task_idx}_timeout_step": t},
                    step=global_step,
                )
                wandb_run.summary[f"task{task_manager.current_task_idx}_timeout_step"] = t
                
                # Clear action plan to force replanning (only in sequential mode)
                if task_manager.use_sequential_tasks:
                    action_plan.clear()
                
                # Always break on timeout - either episode complete or early termination
                break
            
            # Update score
            wandb.log({"score": task_manager.episode_score}, step=global_step)
            
            # Render if requested
            if args.render_mode == 'human':
                env_manager.env.render()
            
            # Debug logging
            if t % 10 == 0 and hasattr(env_manager.detector_simple, "status"):
                logging.debug(f"Detector status @step {t}: {env_manager.detector_simple.status()}")
                if not task_manager.is_episode_complete():
                    logging.debug(f"Current task '{task_manager.get_current_prompt()}' predicate = {env_manager.tasks[task_manager.current_task_idx]['done']()}")
            
            # Regular logging
            if t % 50 == 0:
                logging.info(f"Step: {t}/{args.horizon}, Reward: {reward:.3f}, Loop Time: {time.time() - loop_start_time:.4f}s")
            else:
                logging.debug(f"Step: {t}/{args.horizon}, Reward: {reward:.3f}, Loop Time: {time.time() - loop_start_time:.4f}s")
        
        # End of episode
        episode_duration = time.time() - start_episode_time
        logging.info(f"Episode finished after {t} steps. Duration: {episode_duration:.2f}s. Score: {task_manager.episode_score}")
        
        # Log completion summary (always useful regardless of mode)
        completion_summary = task_manager.get_completion_summary()
        logging.info(f"Task completion summary: {completion_summary['completed_tasks']}/{completion_summary['total_tasks']} tasks completed ({completion_summary['completion_rate']:.1f}%)")
        
        # Save video
        if frames:
            logging.info(f"Saving video ({len(frames)} frames) to {video_full_path}...")
            try:
                imageio.mimwrite(str(video_full_path), frames, fps=env_manager.env.control_freq)
                logging.info("Video saved.")
            except Exception as e:
                logging.error(f"Failed to save video: {e}")
        else:
            logging.warning("No frames collected for video.")
        
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
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s'
    )
    tyro.cli(run_robosuite_with_openpi)