"""
Main script for running Robosuite with OpenPI policy server.
"""
import collections
import dataclasses
import logging
import os
import pathlib
import time
from args import Args

import imageio
import numpy as np
import wandb
import tyro

from openpi_client import websocket_client_policy as _websocket_client_policy

from env_manager import EnvManager
from obs_processor import ObsProcessor
from task_manager import TaskManager
from utils import (
    get_system_metrics,
    calculate_energy_consumption,
    generate_video_filename
)

def _resolve_wandb_mode() -> str:
    """
    Resolve the W&B mode to avoid interactive login prompts in headless / docker runs.

    - If WANDB_MODE is explicitly set, respect it.
    - Otherwise, default to "online".
    """
    explicit = os.getenv("WANDB_MODE")
    if explicit:
        return explicit

    return "online"


def _validate_wandb_config(wandb_mode: str) -> None:
    """
    Fail fast instead of blocking on W&B's interactive login prompt.

    We do NOT auto-disable W&B. If you want to run without W&B, explicitly set
    WANDB_MODE=disabled (or WANDB_MODE=offline).
    """
    mode = (wandb_mode or "").strip().lower()
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"Unsupported WANDB_MODE={wandb_mode!r}. Expected one of: online, offline, disabled.")

    if mode == "online":
        api_key = os.getenv("WANDB_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "WANDB_API_KEY is not set, but WANDB_MODE resolves to 'online'.\n"
                "Set WANDB_API_KEY to enable W&B logging, or explicitly set WANDB_MODE=offline/disabled.\n"
                "This script refuses to block on W&B's interactive login prompt."
            )


def main(args: Args) -> None:
    """Runs a Robosuite environment controlled by an OpenPI policy server."""
    logging.info(f"Running Robosuite env '{args.env}' with OpenPI server at {args.host}:{args.port}")
    logging.info(f"Rendering mode: {args.render_mode}")
    
    # Initialize W&B
    wandb_mode = _resolve_wandb_mode()
    _validate_wandb_config(wandb_mode)
    settings = wandb.Settings(_stats_sampling_interval=args.log_every_n_seconds)
    wandb_proj_name = f"{args.wandb_project_prefix}_{args.env}"
    wandb_group_name = f"{args.env}_{args.episodes}eps_seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    
    # Setup environment manager
    env_manager = EnvManager(args)
    env_manager.setup()

    # Setup task manager
    task_manager = TaskManager(args, env_manager.tasks)

    # Setup observation preprocessor
    obs_processor = ObsProcessor(args.resize_size, env_manager)
    
    # Log the mode being used
    if args.planner_guided:
        logging.info(f"Running in PLANNER GUIDED MODE")
        logging.info(
            f"Subtasks will progress automatically as they are \
            achieved, but episode will terminate early if any subtask \
            takes longer than {args.task_timeout} steps"
        )
    else:
        logging.info("Running in END-TO-END MODE")
        logging.info(f"Using fixed prompt: {task_manager.single_prompt} for all steps")
        logging.info(f"Episode will terminate early if any task takes longer than {args.task_timeout} steps")    
    
    # Setup websocket client
    logging.debug(f"Connecting to OpenPI server at ws://{args.host}:{args.port}...")
    client = _websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port)
    
    # Run episodes
    for ep in range(args.episodes):
        logging.info(f"\n=========  EPISODE {ep+1}/{args.episodes}  =========")

        wandb_run_name = f"{args.env}_{ep+1}ep_{time.strftime('%Y%m%d-%H%M%S')}"
        
        # Initialize W&B run for this episode
        wandb_run = wandb.init(
            project=wandb_proj_name,
            name=wandb_run_name,
            group=wandb_group_name,
            config=dataclasses.asdict(args),
            settings=settings,
            mode=wandb_mode,
        )
        
        # Setup video recording
        video_filename = generate_video_filename(args, ep)
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
            if t < args.settle_steps:
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
                processed_obs = obs_processor.preprocess_observations(obs)
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
                # print("Prompt:")
                # print(task_manager.get_current_prompt())
                # print("State:")
                # print(processed_obs["state"])
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

                # print(f"Action: {action}")
                
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
            
            # Check task timeout
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
            video_full_path.parent.mkdir(parents=True, exist_ok=True)
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
            full_res_video_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving full resolution agentview video ({len(hd_frames)} frames) to {full_res_video_path}...")
            try:
                # Flip frames vertically to fix upside-down issue (same as in dataset_making)
                flipped_frames = [np.flipud(frame) for frame in hd_frames]
                imageio.mimwrite(str(full_res_video_path), flipped_frames, fps=env_manager.env.control_freq, macro_block_size=None)
                logging.info("Full resolution agentview video saved.")
            except Exception as e:
                logging.error(f"Failed to save full resolution agentview video: {e}")
                
            full_res_wrist_video_path.parent.mkdir(parents=True, exist_ok=True)
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
