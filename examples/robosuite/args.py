"""
Args for running Robosuite with OpenPI Websocket Policy and multi-config support
"""
import dataclasses
from typing import List

@dataclasses.dataclass
class Args:
    """Arguments for running Robosuite with OpenPI Websocket Policy and multi-config support"""

    # --- Experiment Settings ---
    wandb_project_prefix: str = "TEST_Kinova3_Journal_Paper"   # W&B project name
    episodes: int = 2                                          # How many episodes to run
    seed: int = 3                                              # Random seed
    log_every_n_seconds: float = 0.5                           # Logging interval for W&B settings

    # --- Robosuite Environment ---
    env: str = "AssemblyLineSorting"             # Robotsuite environment name
    end_to_end_prompt: str = "Sort the cubes on the assembly line by color."
    robots: str = "Kinova3"              # Robot model to use
    controller: str = "OSC_POSE"         # Robosuite controller name
    peg_xy_jitter: float = 0.0225        # Hanoi specific; changes tower spawn location
    cube_placement_noise: float = 0.025  # All other environments; Uniform noise in meters to add to cube x and y positions during spawn
    settle_steps: int = 50               # Number of steps to wait for objects to settle
    horizon: int = 9050                  # Max steps per episode (includes settle_steps) 9050 = 5 minutes of video at 30fps with 50 settle steps

    # --- Subtask Guidance ---
    task_timeout: int = 900         # Number of steps to wait before timing out a task (600 = 20 seconds at 30 fps)
    planner_guided: bool = False    # True for planner-guided; False for end-to-end
    planner:str = "pddl"            # Planner to use: 'pddl' or 'gpt-5'
    # time_based_progression: bool = False # If True, advance to next task after task_timeout steps regardless of completion
    # Not advised, was experimental for not having a planner and didn't work well.

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
    save_full_res_video: bool = False    # Save full resolution videos
    full_res_height: int = 480          # Full resolution video height
    full_res_width: int = 640           # Full resolution video width
    required_cameras: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
        ) # Required cameras for OpenPI preprocessing
    
    # --- OpenPi Server Connection --- (Don't change)
    host: str = "127.0.0.1"         # Hostname of the OpenPI policy server
    port: int = 8000                # Port of the OpenPI policy server

    # --- Policy Interaction --- (Don't change)
    resize_size: int = 224          # Target size for image resizing (must match model training)
    replan_steps: int = 50          # Number of steps per action chunk from policy server

    # --- Multi-configuration support --- Currently unused
    # random_block_placement: bool = False # Place blocks on pegs randomly according to Towers of Hanoi rules
    # random_block_selection: bool = False  # Randomly select 3 out of 4 blocks
    # cube_init_pos_noise_std: float = 0.01  # Std dev for XY jitter of initial tower position
