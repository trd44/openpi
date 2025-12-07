import dataclasses
from typing import List
import time

# --------------------------------------------------------------------------------------
# Configuration dataclass
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    """Arguments for running Robosuite with OpenPI Websocket Policy and multi-config support"""
    
    # --- Server Connection ---
    host: str = "127.0.0.1"         # Hostname of the OpenPI policy server
    port: int = 8000                # Port of the OpenPI policy server

    # --- planner ---
    planner:str = "pddl"            # Planner to use: 'pddl' or 'gpt-5'

    # --- Policy Interaction ---
    resize_size: int = 224           # Target size for image resizing (must match model training)
    replan_steps: int = 10           # Number of steps per action chunk from policy server
    planner_guided: bool =False      # If True, use subtask prompts; if False, use end-to-end task prompts
    time_based_progression: bool = False # If True, advance to next task after task_timeout steps regardless of completion
    task_timeout: int = 500          # Number of steps to wait before timing out a task

    # --- Robosuite Environment ---
    env_name: str = "Hanoi" 
    env: str = "Hanoi"                # Environment name for RecordDemos compatibility
    robots: str = "Kinova3"           # Robot model to use
    controller: str = "OSC_POSE"    # Robosuite controller name
    horizon: int = 7050             # Max steps per episode
    skip_steps: int = 50            # Number of initial steps to skip (e.g., wait for objects to settle)

    # --- Multi-configuration support ---
    random_block_placement: bool = False # Place blocks on pegs randomly according to Towers of Hanoi rules
    random_block_selection: bool = False  # Randomly select 3 out of 4 blocks
    cube_init_pos_noise_std: float = 0.01  # Std dev for XY jitter of initial tower position

    # --- Rendering & Video ---
    render_mode:    str = "headless"                   # Rendering mode: 'headless' (save video) or 'human' (live view)
    video_out_path: str = "data/robosuite_videos"   # Directory to save videos
    camera_height:  int = 256        # Rendered camera height (before potential resize)
    camera_width:   int = 256         # Rendered camera width (before potential resize)
    camera_names:   List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
        ) # Cameras for observation/video
    verbose: bool = False
    noise_std: float = 0.0
    noisy_fraction: float = 0.0
    
    # --- Full Resolution Video Recording ---
    save_full_res_video: bool = True    # Save full resolution videos
    full_res_height: int = 480          # Full resolution video height
    full_res_width: int = 640           # Full resolution video width
    required_cameras: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
        ) # Required cameras for OpenPI preprocessing

    # --- Misc ---
    seed: int = 3           # Random seed
    episodes: int = 10      # How many episodes to run back-to-back

    # --- Logging ---
    wandb_project: str = "TEST_Kinova3_PI05_Hanoi_50_EE_E2E"   # W&B project name
    log_every_n_seconds: float = 0.5                              # Logging interval for W&B settings
    