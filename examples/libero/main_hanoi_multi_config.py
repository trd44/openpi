"""
Multi-configuration Hanoi script that can handle all block configurations similar to the dataset making module.
This script combines the planning capabilities from dataset_making with the task sequence management from main_hanoi.py.
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

# Import from dataset_making module
import sys
sys.path.append('/app')
from dataset_making.record_demos import RecordDemos
from dataset_making.panda_hanoi_detector import PandaHanoiDetector
from planning.planner import add_predicates_to_pddl, call_planner

# Assuming openpi_client is installed in the environment
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from hanoi_detectors import PandaHanoiDetector as SimpleHanoiDetector

# --------------------------------------------------------------------------------------
# Constants and Configuration
# --------------------------------------------------------------------------------------
# Task configuration
COLOR2OBJ = {"blue": "cube1", "red": "cube2", "green": "cube3", "yellow": "cube4"}
AREA2PEG = {"left": "peg1", "middle": "peg2", "right": "peg3"}

# Planning predicates and modes
PLANNING_PREDICATES = {
    "Hanoi": ['on', 'clear', 'grasped', 'smaller'],
    "KitchenEnv": ['on', 'clear', 'grasped', 'stove_on'],
    "NutAssembly": ['on', 'clear', 'grasped'],
}
PLANNING_MODE = {"Hanoi": 0, "KitchenEnv": 1, "NutAssembly": 0}

# --------------------------------------------------------------------------------------
# Configuration dataclass
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    """Arguments for running Robosuite with OpenPI Websocket Policy and multi-config support"""
    # --- Server Connection ---
    host: str = "127.0.0.1"         # Hostname of the OpenPI policy server
    port: int = 8000                # Port of the OpenPI policy server

    # --- Policy Interaction ---
    resize_size: int = 224           # Target size for image resizing (must match model training)
    replan_steps: int = 50           # Number of steps per action chunk from policy server
    use_sequential_tasks: bool = True # If True, use sequential task prompts; if False, use single prompt
    time_based_progression: bool = False # If True, advance to next task after task_timeout steps regardless of completion
    task_timeout: int = 500          # Number of steps to wait before timing out a task

    # --- Robosuite Environment ---
    env_name: str = "Hanoi" 
    env: str = "Hanoi"                # Environment name for RecordDemos compatibility
    robots: str = "Panda"           # Robot model to use
    controller: str = "OSC_POSE"    # Robosuite controller name
    horizon: int = 100050             # Max steps per episode
    skip_steps: int = 50            # Number of initial steps to skip (e.g., wait for objects to settle)

    # --- Multi-configuration support ---
    random_block_placement: bool = True # Place blocks on pegs randomly according to Towers of Hanoi rules
    random_block_selection: bool = True  # Randomly select 3 out of 4 blocks
    cube_init_pos_noise_std: float = 0.01  # Std dev for XY jitter of initial tower position

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
        ) # Required cameras for OpenPI preprocessing

    # --- Misc ---
    seed: int = 3           #: Random seed
    episodes: int = 50      #: How many episodes to run back-to-back

    # --- Logging ---
    wandb_project: str = "openpi_hanoi_300_inference_rgby39000_random_config" #: W&B project name
    log_every_n_seconds: float = 0.5 #: Logging interval for W&B settings
    
    def generate_video_filename(self, episode: int) -> str:
        """Generate video filename based on current arguments and episode number."""
        return f"{self.env_name}_multi_config_seed{self.seed}_ep{episode+1}_{time.strftime('%H%M%S')}.mp4"
    
    def generate_wandb_run_name(self, episode: int) -> str:
        """Generate W&B run name based on current arguments and episode number."""
        return f"{self.env_name}_multi_config_seed{self.seed}_ep{episode+1}_{time.strftime('%H%M%S')}"
    
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
class MultiConfigHanoiEnvironment:
    """Manages the Robosuite Hanoi environment setup with multi-configuration support."""
    
    def __init__(self, args: Args):
        self.args = args
        self.env = None
        self.detector_simple = None
        self.detector_ground = None
        self.tasks = None
        self.recorder = None
        
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
        
        # Create environment with multi-config support
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
            random_block_placement=self.args.random_block_placement,
            random_block_selection=self.args.random_block_selection,
            cube_init_pos_noise_std=self.args.cube_init_pos_noise_std
        )
        
        logging.info(f"Environment created with random_block_placement={self.args.random_block_placement}, random_block_selection={self.args.random_block_selection}")
        
        # Monkey patch to fix robosuite bug with random_block_selection and random_block_placement
        # The issue is that both methods try to access cubes that don't exist when these flags are True.
        # We need to patch both methods to respect current_block_config.
        
        # Patch place_block_tower for random_block_selection
        original_place_block_tower = self.env.place_block_tower
        
        def patched_place_block_tower():
            """Patched version that respects current_block_config when random_block_selection=True"""
            try:
                # Check if we have random_block_selection and current_block_config is set
                if hasattr(self.env, 'random_block_selection') and self.env.random_block_selection:
                    if hasattr(self.env, 'current_block_config') and self.env.current_block_config:
                        available_cubes = set(self.env.current_block_config)
                        logging.info(f"Available cubes in current_block_config: {available_cubes}")
                        
                        # Update placement initializers to only use available cubes
                        if hasattr(self.env, 'large_block_placement_initializer'):
                            large_cube = self.env.current_block_config[2] if len(self.env.current_block_config) > 2 else None
                            if large_cube and hasattr(self.env, large_cube):
                                self.env.large_block_placement_initializer.mujoco_objects = [getattr(self.env, large_cube)]
                                logging.info(f"Updated large_block_placement_initializer to use {large_cube}")
                        
                        if hasattr(self.env, 'medium_block_placement_initializer'):
                            medium_cube = self.env.current_block_config[1] if len(self.env.current_block_config) > 1 else None
                            if medium_cube and hasattr(self.env, medium_cube):
                                self.env.medium_block_placement_initializer.mujoco_objects = [getattr(self.env, medium_cube)]
                                logging.info(f"Updated medium_block_placement_initializer to use {medium_cube}")
                        
                        if hasattr(self.env, 'small_block_placement_initializer'):
                            small_cube = self.env.current_block_config[0] if len(self.env.current_block_config) > 0 else None
                            if small_cube and hasattr(self.env, small_cube):
                                self.env.small_block_placement_initializer.mujoco_objects = [getattr(self.env, small_cube)]
                                logging.info(f"Updated small_block_placement_initializer to use {small_cube}")
                
                # Call original method
                return original_place_block_tower()
                
            except Exception as e:
                logging.error(f"Error in patched place_block_tower: {e}")
                import traceback
                logging.error(f"Full traceback: {traceback.format_exc()}")
                # Fallback to original method
                return original_place_block_tower()
        
        # Patch place_blocks_randomly for random_block_placement
        original_place_blocks_randomly = self.env.place_blocks_randomly
        
        def patched_place_blocks_randomly():
            """Completely rewritten version that respects current_block_config when random_block_placement=True"""
            try:
                # Check if we have random_block_placement and current_block_config is set
                if hasattr(self.env, 'random_block_placement') and self.env.random_block_placement:
                    if hasattr(self.env, 'current_block_config') and self.env.current_block_config:
                        available_cubes = set(self.env.current_block_config)
                        logging.info(f"Available cubes in current_block_config for placement: {available_cubes}")
                        
                        # Set the large, medium, small variables to use available cubes
                        available_cubes_list = list(available_cubes)
                        
                        # Sort by cube number to maintain consistent ordering
                        available_cubes_list.sort(key=lambda x: int(x.replace('cube', '')))
                        
                        logging.info(f"Available cubes (sorted): {available_cubes_list}")
                        
                        # Set large, medium, small based on available cubes
                        if len(available_cubes_list) >= 3:
                            self.env.large = available_cubes_list[2]  # largest cube
                            self.env.medium = available_cubes_list[1]  # medium cube
                            self.env.small = available_cubes_list[0]   # smallest cube
                        elif len(available_cubes_list) == 2:
                            self.env.large = available_cubes_list[1]
                            self.env.medium = available_cubes_list[0]
                            self.env.small = available_cubes_list[0]
                        elif len(available_cubes_list) == 1:
                            self.env.large = available_cubes_list[0]
                            self.env.medium = available_cubes_list[0]
                            self.env.small = available_cubes_list[0]
                        
                        logging.info(f"Set large={self.env.large}, medium={self.env.medium}, small={self.env.small}")
                        
                        # CRITICAL FIX: Update the placement initializers to use the correct mujoco objects
                        # Map cube names to mujoco objects
                        cube_name_to_obj = {
                            'cube1': self.env.cube1,
                            'cube2': self.env.cube2, 
                            'cube3': self.env.cube3,
                            'cube4': self.env.cube4
                        }
                        
                        # Update placement initializers with correct mujoco objects
                        if self.env.large in cube_name_to_obj:
                            self.env.large_block_placement_initializer.mujoco_objects = [cube_name_to_obj[self.env.large]]
                            logging.info(f"Updated large_block_placement_initializer to use {self.env.large}")
                        
                        if self.env.medium in cube_name_to_obj:
                            self.env.medium_block_placement_initializer.mujoco_objects = [cube_name_to_obj[self.env.medium]]
                            logging.info(f"Updated medium_block_placement_initializer to use {self.env.medium}")
                        
                        if self.env.small in cube_name_to_obj:
                            self.env.small_block_placement_initializer.mujoco_objects = [cube_name_to_obj[self.env.small]]
                            logging.info(f"Updated small_block_placement_initializer to use {self.env.small}")
                        
                        # Now call the original method with the corrected variables
                        return original_place_blocks_randomly()
                
                # Call original method
                return original_place_blocks_randomly()
                
            except Exception as e:
                logging.error(f"Error in patched place_blocks_randomly: {e}")
                import traceback
                logging.error(f"Full traceback: {traceback.format_exc()}")
                # Fallback to original method
                return original_place_blocks_randomly()
        
        # Apply the patches
        self.env.place_block_tower = patched_place_block_tower
        self.env.place_blocks_randomly = patched_place_blocks_randomly
        logging.info("Applied monkey patches for robosuite random_block_selection and random_block_placement bugs")
        
        # Add GymWrapper for compatibility with RecordDemos (after patching)
        # Add GymWrapper for compatibility with RecordDemos (like in working examples)
        from robosuite.wrappers import GymWrapper
        self.env = GymWrapper(self.env, proprio_obs=True)
        
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
        
        # Setup PDDL path
        self.pddl_path = os.path.join('/app/planning', 'PDDL', self.args.env_name.lower())
        if not self.pddl_path.endswith(os.sep):
            self.pddl_path += os.sep
        
        # Create recorder for planning
        self.recorder = RecordDemos(
            self.env,
            vision_based=True,  # Use vision-based observations
            detector=self.detector_ground,
            pddl_path=self.pddl_path,
            args=self.args,
            render=self.args.render_mode == 'human',
            randomize=True,
            noise_std_factor=0.03
        )
        
    def reset(self):
        """Reset the environment and generate a new plan."""
        try:
            # Reset the environment
            logging.info("Resetting environment...")
            obs = self.env.reset()
            logging.info("Environment reset successful")
            
            # Reinitialize detectors after reset
            logging.info("Initializing detectors...")
            self.detector_simple = SimpleHanoiDetector(self.env)
            logging.info("Simple detector initialized")
            
            self.detector_ground = PandaHanoiDetector(self.env)
            logging.info(f"Ground detector initialized with objects: {self.detector_ground.objects}")
            
            # Generate a new plan using the recorder
            logging.info("Generating plan...")
            self.recorder.reset()
            logging.info("Plan generation successful")
            
            # Build task sequence based on the generated plan
            logging.info("Building task sequence...")
            self.tasks = self._build_task_sequence_from_plan()
            logging.info(f"Task sequence built with {len(self.tasks)} tasks")
            
            return obs
        except Exception as e:
            logging.error(f"Failed to reset environment: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            if "unexpected keyword argument 'seed'" in str(e):
                logging.error("Environment reset failed likely because it expects env.seed() instead of env.reset(seed=...).")
            raise
    
    def _generate_plan(self) -> List[str]:
        """Generate a plan using the PDDL planner."""
        try:
            # Detect initial state
            state = self.detector_ground.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            
            # Include all relevant predicates
            init_predicates = {}
            for predicate, value in state.items():
                if predicate.split('(')[0] in PLANNING_PREDICATES[self.args.env_name]:
                    init_predicates[predicate] = value
            
            # Filter predicates to only include those involving active cubes
            active_cubes = []
            for cube in self.detector_ground.objects:
                is_active = False
                for pred, value in init_predicates.items():
                    if pred.startswith('on(') and value:
                        parts = pred.split('(')[1].split(',')
                        obj1 = parts[0]
                        obj2 = parts[1].rstrip(')')
                        if obj1 == cube or obj2 == cube:
                            is_active = True
                            break
                if is_active:
                    active_cubes.append(cube)
            
            # Filter predicates to only include those involving active cubes or pegs
            filtered_predicates = {}
            for predicate, value in init_predicates.items():
                if value:
                    if predicate.startswith('smaller('):
                        parts = predicate.split('(')[1].split(',')
                        obj1 = parts[0]
                        obj2 = parts[1].rstrip(')')
                        if (obj1 in active_cubes or obj1 in self.detector_ground.object_areas) and \
                           (obj2 in active_cubes or obj2 in self.detector_ground.object_areas):
                            filtered_predicates[predicate] = value
                    else:
                        predicate_involves_active_cube = False
                        for cube in active_cubes:
                            if cube in predicate:
                                predicate_involves_active_cube = True
                                break
                        
                        if not predicate_involves_active_cube:
                            for peg in self.detector_ground.object_areas:
                                if peg in predicate:
                                    predicate_involves_active_cube = True
                                    break
                        
                        if predicate_involves_active_cube:
                            filtered_predicates[predicate] = value
            
            detected_objects = {
                'cubes': active_cubes,
                'pegs': self.detector_ground.object_areas
            }
            
            add_predicates_to_pddl(self.pddl_path, filtered_predicates, detected_objects=detected_objects)
            
            # Generate plan
            plan, _ = call_planner(self.pddl_path, problem="problem_save.pddl", mode=PLANNING_MODE[self.args.env_name])
            
            print(f"Generated plan: {plan}")
            return plan if plan else []
            
        except Exception as e:
            logging.error(f"Failed to generate plan: {e}")
            return []
    
    def _build_task_sequence_from_plan(self) -> List[Dict[str, Any]]:
        """Build task sequence from the generated plan."""
        try:
            if not hasattr(self.recorder, 'plan') or not self.recorder.plan:
                logging.warning("No plan available, using default task sequence")
                return self._get_default_task_sequence()
            
            logging.info(f"Building task sequence from plan: {self.recorder.plan}")
            tasks = []
            for i, op_str in enumerate(self.recorder.plan):
                # Convert PDDL operator to natural language
                natural_instruction = self._convert_plan_to_natural_language(op_str)
                
                # Create task with completion check
                task = {
                    "prompt": natural_instruction,
                    "done": self._create_completion_check(op_str, i)
                }
                tasks.append(task)
            
            return tasks
        except Exception as e:
            logging.error(f"Error building task sequence from plan: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            logging.warning("Falling back to default task sequence")
            return self._get_default_task_sequence()
    
    def _convert_plan_to_natural_language(self, op_str: str) -> str:
        """Convert PDDL plan to natural language commands."""
        colors = {"cube1": "blue block", "cube2": "red block", "cube3": "green block", "cube4": "yellow block"}
        areas = {"peg1": "left area", "peg2": "middle area", "peg3": "right area"}
        op = op_str.lower().split()
        if not op: return ""
        if op[0] == "pick":
            block = colors.get(op[1], op[1])
            if len(op) > 2:
                # PICK CUBE1 CUBE2 means pick cube1 from cube2
                from_obj = colors.get(op[2], op[2])
                return f"Pick the {block}."
            else:
                return f"Pick up the {block}."
        if op[0] == "place":
            block = colors.get(op[1], op[1])
            # Place target can be area or another block
            if op[2].startswith("cube"):
                target = colors.get(op[2], op[2])
                return f"Place the {block} on top of the {target}."
            else:
                area = areas.get(op[2], op[2])
                return f"Place the {block} in the {area}."
        return op_str  # fallback
    
    def _create_completion_check(self, op_str: str, step_idx: int) -> Callable:
        """Create a completion check function for a given operation."""
        try:
            op = op_str.lower().split()
            
            if op[0] == "pick":
                cube = op[1]
                def check_pick():
                    try:
                        return (bool(self.detector_ground.get_groundings(as_dict=True, binary_to_float=False, return_distance=False).get(f"grasped({cube})", False)) and 
                                self.detector_simple.grasped(cube))
                    except Exception as e:
                        logging.error(f"Error in pick check for {cube}: {e}")
                        return False
                return check_pick
                
            elif op[0] == "place":
                cube = op[1]
                target = op[2]
                def check_place():
                    try:
                        return bool(self.detector_ground.get_groundings(as_dict=True, binary_to_float=False, return_distance=False).get(f"on({cube},{target})", False))
                    except Exception as e:
                        logging.error(f"Error in place check for {cube} on {target}: {e}")
                        return False
                return check_place
            
            else:
                # Default: always return False for unknown operations
                return lambda: False
        except Exception as e:
            logging.error(f"Error creating completion check for {op_str}: {e}")
            return lambda: False
    
    def _get_default_task_sequence(self) -> List[Dict[str, Any]]:
        """Get default task sequence when no plan is available."""
        try:
            def g():
                try:
                    return self.detector_ground.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
                except Exception as e:
                    logging.error(f"Error getting groundings: {e}")
                    return {}
            
            # Get available cubes dynamically
            available_cubes = []
            for cube in ['cube1', 'cube2', 'cube3', 'cube4']:
                try:
                    if cube in self.detector_ground.objects:
                        available_cubes.append(cube)
                except Exception as e:
                    logging.error(f"Error checking cube {cube}: {e}")
                    continue
            
            logging.info(f"Available cubes: {available_cubes}")
            
            if len(available_cubes) < 1:
                # Fallback to a very simple task if no cubes available
                return [
                    {"prompt": "Move the robot arm.", "done": lambda: True},
                ]
            
            if len(available_cubes) < 2:
                # Fallback to a simple task if not enough cubes
                return [
                    {"prompt": "Pick up a block.", "done": lambda: any(g().get(f"grasped({cube})", False) for cube in available_cubes)},
                    {"prompt": "Place the block in the right area.", "done": lambda: any(g().get(f"on({cube},peg3)", False) for cube in available_cubes)},
                ]
            
            # Create tasks for available cubes
            tasks = []
            colors = {"cube1": "blue", "cube2": "red", "cube3": "green", "cube4": "yellow"}
            pegs = ["peg1", "peg2", "peg3"]
            
            for i, cube in enumerate(available_cubes):
                color = colors.get(cube, cube)
                target_peg = pegs[i % len(pegs)]  # Distribute across pegs
                
                tasks.append({
                    "prompt": f"Pick the {color} block.", 
                    "done": lambda c=cube: bool(g().get(f"grasped({c})", False)) and self.detector_simple.grasped(colors.get(c, c))
                })
                tasks.append({
                    "prompt": f"Place the {color} block in the {target_peg} area.", 
                    "done": lambda c=cube, p=target_peg: bool(g().get(f"on({c},{p})", False))
                })
            
            return tasks
        except Exception as e:
            logging.error(f"Error creating default task sequence: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            # Return a very simple fallback
            return [
                {"prompt": "Move the robot arm.", "done": lambda: True},
            ]
    
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
            "robot0_gripper_qpos"
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
    """Runs a Robosuite environment controlled by an OpenPI policy server with multi-config support."""
    logging.info(f"Running Robosuite env '{args.env_name}' with OpenPI server at {args.host}:{args.port}")
    logging.info(f"Rendering mode: {args.render_mode}")
    logging.info(f"Multi-config settings: random_block_placement={args.random_block_placement}, random_block_selection={args.random_block_selection}")
    
    # Initialize W&B
    settings = wandb.Settings(_stats_sampling_interval=args.log_every_n_seconds)
    group_name = f"multi_config_hanoi_{time.strftime('%Y%m%d-%H%M%S')}"
    
    # Setup environment
    env_manager = MultiConfigHanoiEnvironment(args)
    env_manager.setup()
    
    # Setup task manager
    task_manager = TaskManager(env_manager.tasks, args.use_sequential_tasks, 
                              args.time_based_progression, args.task_timeout)
    
    # Log the mode being used
    if args.use_sequential_tasks:
        logging.info(f"Running in SEQUENTIAL TASK MODE with dynamic task generation")
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
    obs_preprocessor = ObservationPreprocessor(args.resize_size, env_manager)
    
    # Setup websocket client
    logging.info(f"Connecting to OpenPI server at ws://{args.host}:{args.port}...")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    
    # Run episodes
    for ep in range(args.episodes):
        logging.info(f"\n==========  EPISODE {ep+1}/{args.episodes}  ==========")
        
        # Initialize W&B run for this episode
        run_name = args.generate_wandb_run_name(ep)
        wandb_run = wandb.init(
            project=args.wandb_project,
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
            # Update task manager with new tasks
            task_manager.tasks = env_manager.tasks
            task_manager.reset_episode()
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
