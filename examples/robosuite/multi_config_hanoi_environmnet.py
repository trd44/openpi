# Environment setup and management
# --------------------------------------------------------------------------------------

import logging
import numpy as np
from typing import List, Dict, Any, Callable
import os
import sys
sys.path.append('/app')
from dataset_making.record_demos import RecordDemos
from dataset_making.panda_hanoi_detector import PandaHanoiDetector
from planning.planner import add_predicates_to_pddl, call_planner


import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.detector import (
    HanoiDetector,
    KitchenDetector, 
    NutAssemblyDetector, 
    CubeSortingDetector,
    HeightStackingDetector,
    AssemblyLineSortingDetector,
    PatternReplicationDetector
)

from args import Args

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
        for cam in self.args.required_cameras:
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
            # random_block_placement=self.args.random_block_placement,
            # random_block_selection=self.args.random_block_selection,
            # cube_init_pos_noise_std=self.args.cube_init_pos_noise_std
        )
        
        logging.info(f"Environment created with random_block_placement={self.args.random_block_placement}, random_block_selection={self.args.random_block_selection}")
        
        # Monkey patch to fix robosuite bug with random_block_selection and random_block_placement
        # The issue is that both methods try to access cubes that don't exist when these flags are True.
        # We need to patch both methods to respect current_block_config.
        
        # Check if place_block_tower method exists before trying to patch it
        if hasattr(self.env, 'place_block_tower'):
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
            
            # Apply the patch
            self.env.place_block_tower = patched_place_block_tower
            logging.info("Applied place_block_tower patch")
        else:
            logging.warning("place_block_tower method not found on environment. Skipping patch.")
        
        # Check if place_blocks_randomly method exists before trying to patch it
        if hasattr(self.env, 'place_blocks_randomly'):
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
            
            # Apply the patch
            self.env.place_blocks_randomly = patched_place_blocks_randomly
            logging.info("Applied place_blocks_randomly patch")
        else:
            logging.warning("place_blocks_randomly method not found on environment. Skipping patch.")
        
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
        self.detector_simple = HeightStackingDetector(self.env)
        self.detector_ground = HeightStackingDetector(self.env)
        
        # Setup PDDL path
        # Use 'hanoi' for PDDL path since Hanoi4x3 uses the same PDDL as Hanoi
        pddl_env_name = 'hanoi' if self.args.env_name.lower() == 'hanoi4x3' else self.args.env_name.lower()
        self.pddl_path = os.path.join('/app/planning', 'PDDL', pddl_env_name)
        # uncoment the line below if running without docker
        # self.pddl_path = os.path.join('/home/hrilab/Documents/.vlas/cycliclxm-slim/CyclicLxM/planning/', 'PDDL', self.args.env_name.lower())
        if not self.pddl_path.endswith(os.sep):
            self.pddl_path += os.sep
        
        # Create recorder for planning
        self.recorder = RecordDemos(
            args=self.args,
            env=self.env,
            # vision_based=True,  # Use vision-based observations
            detector=self.detector_ground,
            pddl_path=self.pddl_path
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
            self.detector_simple = HeightStackingDetector(self.env)
            logging.info("Simple detector initialized")
            
            self.detector_ground = HeightStackingDetector(self.env)
            logging.info(f"Ground detector initialized with objects: {self.detector_ground.objects}")
            
            # Generate a new plan using the recorder
            logging.info("Generating plan...")
            self.recorder.reset(successes=0)
            logging.info("Plan generation successful")
            
            # Build task sequence based on the generated plan
            logging.info("Building task sequence...")
            self.tasks = self._build_task_sequence_from_plan()
            logging.info(f"Task sequence built with {len(self.tasks)} tasks")
            
            # Log the task sequence for debugging
            for i, task in enumerate(self.tasks):
                logging.info(f"Task {i+1}: {task['prompt']}")
            
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
                        # Use ground detector as primary check, with simple detector as backup
                        ground_grasped = bool(self.detector_ground.get_groundings(as_dict=True, binary_to_float=False, return_distance=False).get(f"grasped({cube})", False))
                        simple_grasped = self.detector_simple.grasped(cube)
                        
                        # Log the individual detector results for debugging
                        if ground_grasped or simple_grasped:
                            logging.debug(f"Pick check for {cube}: ground_grasped={ground_grasped}, simple_grasped={simple_grasped}")
                        
                        # Return True if either detector confirms grasping
                        return ground_grasped or simple_grasped
                    except Exception as e:
                        logging.error(f"Error in pick check for {cube}: {e}")
                        return False
                return check_pick
                
            elif op[0] == "place":
                cube = op[1]
                target = op[2]
                def check_place():
                    try:
                        # Check if cube is placed on target using ground detector
                        ground_placed = bool(self.detector_ground.get_groundings(as_dict=True, binary_to_float=False, return_distance=False).get(f"on({cube},{target})", False))
                        
                        # Additional check: ensure cube is not grasped anymore (has been released)
                        not_grasped = not bool(self.detector_ground.get_groundings(as_dict=True, binary_to_float=False, return_distance=False).get(f"grasped({cube})", False))
                        
                        # Log the individual detector results for debugging
                        if ground_placed or not_grasped:
                            logging.debug(f"Place check for {cube} on {target}: ground_placed={ground_placed}, not_grasped={not_grasped}")
                        
                        # Return True if cube is placed on target and not grasped
                        return ground_placed and not_grasped
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