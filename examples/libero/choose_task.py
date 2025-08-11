import collections
import dataclasses # Imported to match your list, not actively used for config in this version
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv # Using this as the env type
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro # Imported to match your list, but not used for CLI parsing in this version

# --- User Configuration ---
# Details for connecting to your running policy model server
MODEL_SERVER_HOST = "0.0.0.0"
MODEL_SERVER_PORT = 8000

# Image preprocessing
RESIZE_SIZE = 224 # Size to resize observation images to

# Policy interaction
REPLAN_STEPS = 5 # How many steps of an action plan to execute before replanning

# Task Selection: Choose which LIBERO task to run
# Available task suites: "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"
TARGET_TASK_SUITE_NAME = "libero_10"

# To find a TARGET_TASK_ID for the suite above:
# You can run the following snippet in a separate Python interpreter or temporarily in this script:
# """
# from libero.libero import benchmark
# suite_name_to_inspect = "libero_10" # Change this to the suite you're interested in
# task_suite_obj = benchmark.get_benchmark_dict()[suite_name_to_inspect]()
# print(f"Tasks in suite: {suite_name_to_inspect}")
# for i, task_obj in enumerate(task_suite_obj.tasks):
#     print(f"  ID: {i}, Name: {task_obj.name}, Description: {task_obj.language}")
# """
TARGET_TASK_ID = 1 # Index of the task within the TARGET_TASK_SUITE_NAME (e.g., 0 for the first task)

# How many different initial states (trials) of this specific task to attempt.
# If NUM_TRIALS_FOR_SELECTED_TASK is more than available initial states for the task,
# it will run for the number of available initial states.
NUM_TRIALS_FOR_SELECTED_TASK = 20

# Simulation parameters
NUM_STEPS_WAIT = 10  # Number of steps to wait for objects to stabilize in sim after reset/action
# MAX_STEPS_PER_EPISODE will be derived based on the TARGET_TASK_SUITE_NAME later in the script.

# Output and Reproducibility
VIDEO_OUT_PATH = "data/libero/videos/single_task_runs"  # Path to save videos
SEED = 0  # Random Seed (for reproducibility)
# --- End User Configuration ---


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # Resolution used for rendering environment observations


def _get_libero_env_for_task(
    task_object, # The specific task object from LIBERO benchmark (type Any)
    resolution: int,
    seed_val: int,
): # Returns a tuple (OffScreenRenderEnv, str)
    """
    Initializes and returns the LIBERO environment for a given task object.
    """
    task_description_out = task_object.language
    # Ensure bddl_file_name is a string path
    bddl_file_path_str = str(pathlib.Path(get_libero_path("bddl_files")) / task_object.problem_folder / task_object.bddl_file)
    
    env_args = {"bddl_file_name": bddl_file_path_str, "camera_heights": resolution, "camera_widths": resolution}
    
    logging.info(f"Initializing environment for task: {task_object.name} using BDDL: {bddl_file_path_str}")
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed_val)
    return env, task_description_out


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion to an axis-angle representation.
    """
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def run_episode(
    env: OffScreenRenderEnv, # Environment instance
    task_description: str,
    policy_client: _websocket_client_policy.WebsocketClientPolicy,
    max_steps_for_episode: int,
    initial_state_data: dict, # Specific initial state for this trial (type Dict)
    video_filename_prefix: str,
    # Pass relevant config values directly
    current_replan_steps: int,
    current_resize_size: int,
    current_num_steps_wait: int,
    current_video_out_path: str,
    save_only_failures: bool = False,
): # Returns a tuple (bool, int)
    """
    Runs a single episode in the environment.
    """
    logging.info(f"\nStarting new trial. Task: {task_description}")
    
    current_obs_dict = env.reset() # General reset
    logging.info("Setting specific initial state for this trial.")
    current_obs_dict = env.set_init_state(initial_state_data) # Apply specific initial state

    action_plan = collections.deque()
    replay_images = []
    t = 0
    done = False

    # Total steps include wait time and actual interaction time
    effective_max_steps = max_steps_for_episode + current_num_steps_wait

    while t < effective_max_steps:
        try:
            # Wait for stabilization
            if t < current_num_steps_wait:
                current_obs_dict, reward, _, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                raw_img_for_video = np.ascontiguousarray(current_obs_dict["agentview_image"][::-1, ::-1])
                processed_img_for_video = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(raw_img_for_video, current_resize_size, current_resize_size)
                )
                replay_images.append(processed_img_for_video)
                continue

            # Preprocess images
            img_agent_view_raw = np.ascontiguousarray(current_obs_dict["agentview_image"][::-1, ::-1])
            img_wrist_view_raw = np.ascontiguousarray(current_obs_dict["robot0_eye_in_hand_image"][::-1, ::-1])

            img_agent_view_processed = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_agent_view_raw, current_resize_size, current_resize_size)
            )
            img_wrist_view_processed = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_wrist_view_raw, current_resize_size, current_resize_size)
            )
            replay_images.append(img_agent_view_processed)

            # Get action from policy
            if not action_plan:
                policy_obs = {
                    "observation/image": img_agent_view_processed,
                    "observation/wrist_image": img_wrist_view_processed,
                    "observation/state": np.concatenate(
                        (
                            current_obs_dict["robot0_eef_pos"],
                            _quat2axisangle(current_obs_dict["robot0_eef_quat"]),
                            current_obs_dict["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": "pick up the butter and put it in the basket",
                    # "prompt": str(task_description),
                }
                action_chunk = policy_client.infer(policy_obs)["actions"]
                assert (
                    len(action_chunk) >= current_replan_steps
                ), f"Policy should predict at least {current_replan_steps} steps, but got {len(action_chunk)}."
                action_plan.extend(action_chunk[:current_replan_steps])

            action_to_execute = action_plan.popleft()
            current_obs_dict, reward, step_done, info = env.step(action_to_execute.tolist())
            
            if step_done: # Task success condition met
                logging.info(f"Episode SUCCEEDED at step {t - current_num_steps_wait} (interaction step)!")
                done = True
                # Capture one last frame
                raw_img_final = np.ascontiguousarray(current_obs_dict["agentview_image"][::-1, ::-1])
                processed_img_final = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(raw_img_final, current_resize_size, current_resize_size)
                )
                replay_images.append(processed_img_final)
                break
            
            t += 1
            if t >= effective_max_steps:
                 logging.info(f"Episode TIMED OUT after {t - current_num_steps_wait} interaction steps (max_steps: {max_steps_for_episode}).")
                 break

        except Exception as e:
            logging.error(f"Caught exception during episode: {e}", exc_info=True)
            break 

    # Save video
    video_suffix = "success" if done else "failure"
    video_file_path = pathlib.Path(current_video_out_path) / f"{video_filename_prefix}_{video_suffix}.mp4"
    if save_only_failures and done:
        logging.info("Episode succeeded and save_only_failures=True, so not saving video.")
        return done, (t - current_num_steps_wait if t >= current_num_steps_wait else 0)
    try:
        imageio.mimwrite(video_file_path, [np.asarray(frame) for frame in replay_images], fps=10)
        logging.info(f"Saved video to: {video_file_path}")
    except Exception as e:
        logging.error(f"Failed to save video: {e}")

    return done, (t - current_num_steps_wait if t >= current_num_steps_wait else 0) # Return interaction steps


def evaluate_selected_task():
    # Setup
    np.random.seed(SEED)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    pathlib.Path(VIDEO_OUT_PATH).mkdir(parents=True, exist_ok=True)

    # Connect to policy server
    try:
        client = _websocket_client_policy.WebsocketClientPolicy(MODEL_SERVER_HOST, MODEL_SERVER_PORT)
        logging.info(f"Attempting to connect to policy server at {MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")
        # Optional: Add a ping or simple test inference here if your client supports it
    except Exception as e:
        logging.error(f"Failed to initialize policy client: {e}. Ensure the policy server is running.")
        return

    # Load task suite and specific task
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        if TARGET_TASK_SUITE_NAME not in benchmark_dict:
            logging.error(f"Unknown task suite: {TARGET_TASK_SUITE_NAME}. Available: {list(benchmark_dict.keys())}")
            return
        task_suite = benchmark_dict[TARGET_TASK_SUITE_NAME]()
        
        if not (0 <= TARGET_TASK_ID < task_suite.n_tasks):
            logging.error(f"TARGET_TASK_ID {TARGET_TASK_ID} is out of range for suite {TARGET_TASK_SUITE_NAME} (0-{task_suite.n_tasks-1}).")
            return
        
        selected_task_object = task_suite.get_task(TARGET_TASK_ID)
        logging.info(f"Selected Task: {selected_task_object.name} (ID: {TARGET_TASK_ID}) from suite {TARGET_TASK_SUITE_NAME}")

    except Exception as e:
        logging.error(f"Failed to load task suite or task: {e}. Ensure LIBERO is correctly installed.")
        return

    # Get initial states for the selected task
    try:
        initial_states_for_task = task_suite.get_task_init_states(TARGET_TASK_ID)
        # MODIFIED CHECK: Robustly check if initial_states_for_task is None or an empty sequence (list or numpy array)
        # This avoids the "truth value of an array is ambiguous" error if initial_states_for_task is a non-empty numpy array.
        if initial_states_for_task is None or \
           (hasattr(initial_states_for_task, '__len__') and len(initial_states_for_task) == 0):
            logging.warning(f"No initial states found for task '{selected_task_object.name}'. "
                            "This is likely due to the missing datasets path. "
                            "Please ensure '/app/third_party/libero/libero/datasets' exists and is populated. "
                            "Cannot run trials.")
            return

    except Exception as e:
        logging.error(f"Error getting initial states for task '{selected_task_object.name}': {e}")
        return

    # Determine max_steps for episodes based on the task suite
    if TARGET_TASK_SUITE_NAME == "libero_spatial": max_steps_episode = 220
    elif TARGET_TASK_SUITE_NAME == "libero_object": max_steps_episode = 280
    elif TARGET_TASK_SUITE_NAME == "libero_goal": max_steps_episode = 300
    elif TARGET_TASK_SUITE_NAME == "libero_10": max_steps_episode = 520
    elif TARGET_TASK_SUITE_NAME == "libero_90": max_steps_episode = 400
    else:
        logging.warning(f"Max steps not explicitly defined for task suite {TARGET_TASK_SUITE_NAME}. Using a default of 300.")
        max_steps_episode = 300
    
    logging.info(f"Max interaction steps per episode set to: {max_steps_episode}")

    total_episodes_run = 0
    total_successes = 0
    
    num_available_initial_states = len(initial_states_for_task)
    actual_num_trials_to_run = min(NUM_TRIALS_FOR_SELECTED_TASK, num_available_initial_states)

    if NUM_TRIALS_FOR_SELECTED_TASK > num_available_initial_states:
        logging.warning(
            f"Requested {NUM_TRIALS_FOR_SELECTED_TASK} trials, but only {num_available_initial_states} "
            f"initial states are available for task '{selected_task_object.name}'. "
            f"Running {num_available_initial_states} trials."
        )
    
    if actual_num_trials_to_run == 0:
        logging.info(f"No trials to run for task '{selected_task_object.name}'. Exiting.")
        return

    logging.info(f"Will run {actual_num_trials_to_run} trial(s) for task '{selected_task_object.name}'.")

    env, task_description = _get_libero_env_for_task(
        selected_task_object, LIBERO_ENV_RESOLUTION, SEED
    )

    for trial_idx in tqdm.tqdm(range(actual_num_trials_to_run), desc=f"Trials for '{selected_task_object.name[:30]}'"):
        current_initial_state_data = initial_states_for_task[trial_idx]
        
        task_name_sanitized = selected_task_object.name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        video_prefix = f"BUTTER_task_{TARGET_TASK_ID}_{task_name_sanitized}_trial{trial_idx+1}"
        
        success, steps_taken = run_episode(
            env, task_description, client, max_steps_episode,
            initial_state_data=current_initial_state_data,
            video_filename_prefix=video_prefix,
            current_replan_steps=REPLAN_STEPS,
            current_resize_size=RESIZE_SIZE,
            current_num_steps_wait=NUM_STEPS_WAIT,
            current_video_out_path=VIDEO_OUT_PATH,
            save_only_failures=True, # Change to True to save only failure videos
        )
        
        total_episodes_run += 1
        if success:
            total_successes += 1
        
        logging.info(f"Trial {trial_idx+1}/{actual_num_trials_to_run}: Success = {success}, Interaction Steps = {steps_taken}")
        if total_episodes_run > 0:
            current_sr = total_successes / total_episodes_run * 100
            logging.info(f"Cumulative Success Rate for this task: {current_sr:.1f}% ({total_successes}/{total_episodes_run})")

    logging.info("-" * 30)
    if total_episodes_run > 0:
        final_sr = total_successes / total_episodes_run * 100
        logging.info(f"FINAL RESULTS for Task '{selected_task_object.name}':")
        logging.info(f"  Success Rate: {final_sr:.1f}% ({total_successes}/{total_episodes_run})")
        logging.info(f"  Total trials run: {total_episodes_run}")
    else:
        logging.info(f"No trials were completed for task '{selected_task_object.name}'.")
    logging.info("-" * 30)
    logging.info("Evaluation finished.")


if __name__ == "__main__":
    # Tyro is imported to match your list, but we are not using it for CLI parsing here.
    # We directly call the main function.
    evaluate_selected_task()
    # all_benchmark_suites = benchmark.get_benchmark_dict()

    # print("Listing all available LIBERO tasks by suite:\n")

    # for suite_name, benchmark_constructor in all_benchmark_suites.items():
    #     print(f"--- SUITE: {suite_name} ---")
    #     try:
    #         # Instantiate the benchmark suite
    #         task_suite_obj = benchmark_constructor()
            
    #         # The 'tasks' attribute of the suite object is a list of task objects
    #         if hasattr(task_suite_obj, 'tasks') and task_suite_obj.tasks:
    #             for i, task_obj in enumerate(task_suite_obj.tasks):
    #                 # Each task_obj typically has attributes like 'name', 'language', etc.
    #                 task_id_in_suite = i # The ID used in your script (TARGET_TASK_ID)
    #                 task_name = getattr(task_obj, 'name', 'N/A')
    #                 task_language = getattr(task_obj, 'language', 'N/A')
                    
    #                 print(f"  ID: {task_id_in_suite:<3} | Name: {task_name:<70} | Language: {task_language}")
    #         else:
    #             print(f"  No tasks found or 'tasks' attribute missing for suite: {suite_name}")
    #         print("\n") # Add a newline for better separation between suites

    #     except Exception as e:
    #         print(f"  Could not load or inspect tasks for suite {suite_name}. Error: {e}\n")

