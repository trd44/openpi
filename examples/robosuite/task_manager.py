# -----------------------------------------------------------------------------
# Task management
# -----------------------------------------------------------------------------
from typing import List, Dict, Any
import logging

class TaskManager:
    """Manages task progression and completion tracking."""
    
    def __init__(self, tasks: List[Dict[str, Any]], use_sequential_tasks: bool = False, 
                 time_based_progression: bool = False, task_timeout: int = 200):
        self.single_prompt = "Stack the cubes from largest to smallest on the platform."
        self.tasks = tasks
        self.use_sequential_tasks = use_sequential_tasks
        self.time_based_progression = time_based_progression
        self.task_timeout = task_timeout
        self.current_task_idx = 0
        self.current_prompt = tasks[0]["prompt"] if tasks and use_sequential_tasks else self.single_prompt
        self.task_totals = [0] * len(self.tasks) if tasks else []
        self.episode_score = 0
        self.task_completed_this_episode = [False] * len(self.tasks) if tasks else []
        self.task_start_step = 0  # Track when current task started
        self.episode_start_step = 0  # Track when episode started for overall progress tracking
        self.last_progress_step = 0  # Track when any task was last completed
        
    def reset_episode(self):
        """Reset episode-specific counters."""
        if self.tasks:  # Always reset task tracking if tasks exist
            self.task_totals = [0] * len(self.tasks)
            self.task_completed_this_episode = [False] * len(self.tasks)
        
        # Always reset current task index to 0 for both modes
        self.current_task_idx = 0
        
        if self.use_sequential_tasks:
            self.current_prompt = self.tasks[0]["prompt"] if self.tasks else ""
        else:
            self.current_prompt = self.single_prompt
            
        self.episode_score = 0
        self.task_start_step = 0
        self.episode_start_step = 0
        self.last_progress_step = 0
        
    def check_task_timeout(self, step: int) -> bool:
        """Check if current task has timed out and either advance or terminate based on mode."""
        if not self.tasks:
            return False
        
        if self.use_sequential_tasks:
            # Sequential mode: check timeout for current task
            if self.current_task_idx >= len(self.tasks):
                return False
            
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
                        
                    # In sequential mode, change the prompt
                    self.current_prompt = self.tasks[self.current_task_idx]["prompt"]
                    logging.info(f"⏭️ Starting next task {self.current_task_idx + 1}/{len(self.tasks)}: {self.current_prompt}")
                    
                    # Reset task start step for the new task
                    self.task_start_step = step
                    return True
                else:
                    # In completion-only mode, terminate the episode early
                    logging.info(f"⏰ Task {self.current_task_idx + 1}/{len(self.tasks)} timed out after {steps_on_current_task} steps: {self.tasks[self.current_task_idx]['prompt']}")
                    logging.info("🛑 Episode terminated early due to task timeout (time_based_progression=False)")
                    return True
        else:
            # Single prompt mode: check timeout for current sequential task
            if self.current_task_idx >= len(self.tasks):
                return False
            
            steps_on_current_task = step - self.task_start_step
            
            # Debug logging for timeout
            if step % 1000 == 0:  # Log every 1000 steps
                logging.info(f"Timeout check: step={step}, current_task_idx={self.current_task_idx}, task_start_step={self.task_start_step}, steps_on_current_task={steps_on_current_task}, timeout={self.task_timeout}")
            
            if steps_on_current_task >= self.task_timeout:
                logging.info(f"⏰ Task {self.current_task_idx + 1}/{len(self.tasks)} timed out after {steps_on_current_task} steps: {self.tasks[self.current_task_idx]['prompt']}")
                logging.info("🛑 Episode terminated early due to task timeout in single prompt mode")
                return True
        
        return False

    def check_task_completion(self, step: int) -> bool:
        """Check if current task is completed and advance if so."""
        if not self.tasks:
            return False
        
        task_completed = False
        
        if self.use_sequential_tasks:
            # Sequential mode: check completion in sequential order
            if (self.current_task_idx < len(self.tasks) and 
                self.tasks[self.current_task_idx]["done"]() and 
                not self.task_completed_this_episode[self.current_task_idx]):
                
                # Mark task as completed for this episode
                self.task_completed_this_episode[self.current_task_idx] = True
                self.task_totals[self.current_task_idx] += 1
                self.episode_score += 1
                
                logging.info(f"✅ Completed task {self.current_task_idx + 1}/{len(self.tasks)} on step {step} : {self.tasks[self.current_task_idx]['prompt']}")
                logging.info(f"📊 Score updated: {self.episode_score}/{len(self.tasks)} tasks completed")
                
                # Advance to next task
                self.current_task_idx += 1
                if self.current_task_idx >= len(self.tasks):
                    logging.info("🎉 All tasks completed – ending episode early.")
                    return True
                    
                # In sequential mode, change the prompt
                self.current_prompt = self.tasks[self.current_task_idx]["prompt"]
                logging.info(f"⏭️ Starting next task {self.current_task_idx + 1}/{len(self.tasks)}: {self.current_prompt}")
                
                # Reset task start step for the new task
                self.task_start_step = step
                task_completed = True
        else:
            # Single prompt mode: check tasks in sequential order (can't complete task 2 before task 1)
            if (self.current_task_idx < len(self.tasks) and 
                self.tasks[self.current_task_idx]["done"]() and 
                not self.task_completed_this_episode[self.current_task_idx]):
                
                # Mark task as completed for this episode
                self.task_completed_this_episode[self.current_task_idx] = True
                self.task_totals[self.current_task_idx] += 1
                self.episode_score += 1
                
                logging.info(f"✅ Completed task {self.current_task_idx + 1}/{len(self.tasks)} on step {step} : {self.tasks[self.current_task_idx]['prompt']}")
                logging.info(f"📊 Score updated: {self.episode_score}/{len(self.tasks)} tasks completed")
                
                # Update progress tracking
                self.last_progress_step = step
                task_completed = True
                
                # Advance to next task (sequential order maintained)
                self.current_task_idx += 1
                if self.current_task_idx >= len(self.tasks):
                    logging.info("🎉 All tasks completed – ending episode early.")
                    return True
                else:
                    # Reset task start step for the new task
                    self.task_start_step = step
                    logging.info(f"📊 Task {self.current_task_idx} completed in single prompt mode - continuing with same prompt for next task")
                    logging.info(f"Task progression: current_task_idx={self.current_task_idx}, task_start_step={self.task_start_step}")
            
        return task_completed
    
    def get_current_prompt(self) -> str:
        """Get the current task prompt."""
        return self.current_prompt
    
    def is_episode_complete(self) -> bool:
        """Check if all tasks are completed."""
        if not self.tasks:
            return False
        # Episode is complete when all tasks have been completed
        return all(self.task_completed_this_episode)
    
    def should_clear_action_plan(self) -> bool:
        """Determine if action plan should be cleared when episode completes."""
        return self.is_episode_complete()
    
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
            "task_details": {f"task_{i+1}": count for i, count in enumerate(self.task_totals)},
            "episode_progress": {
                "episode_start_step": self.episode_start_step,
                "last_progress_step": self.last_progress_step,
                "steps_since_progress": 0  # Will be calculated by caller
            }
        }
