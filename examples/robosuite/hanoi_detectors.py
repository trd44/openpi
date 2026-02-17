"""
Task-specific detector predicates for Panda Hanoi environments in robosuite.
"""
# -----------------------------------------------------------------------------
# File: openpi/examples/robosuite/hanoi_detectors.py
# Purpose: Predicate checks used by planning state extraction and task scoring.
# -----------------------------------------------------------------------------
import numpy as np
from robosuite.utils.detector import HanoiDetector as _BaseHanoiDetector

class PandaHanoiDetector(_BaseHanoiDetector):
    """Panda-specific detector implementation for Hanoi predicates."""

    def __init__(self, env):
        """Initialize geometry mappings and peg reference positions."""
        super().__init__(env)
        robot_model_name = getattr(env.robots[0].robot_model, 'model_type', '')
        self.is_panda = 'Panda' in robot_model_name
        
        gripper_model = self.env.robots[0].gripper
        self.left_tip = gripper_model.important_geoms.get("left_fingerpad") 
        self.right_tip = gripper_model.important_geoms.get("right_fingerpad")

        # PDDL names
        self.objects = ['cube1', 'cube2', 'cube3'] 
        self.pegs_pddl_names = ['peg1', 'peg2', 'peg3'] 

        # PDDL name -> MuJoCo body name
        self.object_id = {
            'cube1': 'cube1_main',
            'cube2': 'cube2_main',
            'cube3': 'cube3_main',
            'peg1':  'peg1_main',
            'peg2':  'peg2_main',
            'peg3':  'peg3_main',
            'gripper': 'gripper0_eef'
        }

        if hasattr(env, 'pegs_xy_center'):
            self.peg_target_positions = {
                'peg1': np.array(env.pegs_xy_center[0]),
                'peg2': np.array(env.pegs_xy_center[1]),
                'peg3': np.array(env.pegs_xy_center[2]),
            }
        else:
            self.peg_target_positions = {}
            print("Warning: env.pegs_xy_center not found in PandaHanoiDetector init.")

    def open(self, gripper_pddl_name, return_distance=False):
        """Check whether the gripper is open, or return opening distance."""
        j1 = float(self.env.sim.data.get_joint_qpos('gripper0_finger_joint1'))
        j2 = float(self.env.sim.data.get_joint_qpos('gripper0_finger_joint2'))
        qpos_diff = abs(j1 - j2)
        
        if return_distance:
            return qpos_diff
        return qpos_diff > 0.075  # Threshold for "open"

    def grasped(self, obj_pddl_name, return_distance=False):
        """Check whether either finger pad is in contact with the target object."""
        target_mjcf_body_name = self.object_id.get(obj_pddl_name)
        if not target_mjcf_body_name: return False
        
        body_id = self.env.sim.model.body_name2id(target_mjcf_body_name)
        if body_id == -1: return False

        model = self.env.sim.model
        geom_ids = [i for i, b_id in enumerate(model.geom_bodyid) if b_id == body_id]
        obj_mjcf_geom_names = [model.geom_id2name(g_id) for g_id in geom_ids if model.geom_id2name(g_id) is not None]

        if self.left_tip is None or self.right_tip is None:
            return False

        left_contact  = self.env.check_contact(self.left_tip,  obj_mjcf_geom_names)
        right_contact = self.env.check_contact(self.right_tip, obj_mjcf_geom_names)
        is_grasped = bool(left_contact or right_contact)

        if return_distance:
            return 0.0 if is_grasped else 1.0 
        return is_grasped

    def over(self, gripper_pddl_name, obj_pddl_name, return_distance=False):
        """Check XY alignment between gripper and a cube or peg target."""
        gripper_mjcf_body_name = self.object_id.get(gripper_pddl_name)
        if not gripper_mjcf_body_name:
            return False if not return_distance else np.inf
        
        gripper_body_id = self.env.sim.model.body_name2id(gripper_mjcf_body_name)
        if gripper_body_id == -1:
            return False if not return_distance else np.inf
        gripper_pos = self.env.sim.data.body_xpos[gripper_body_id]

        # Get target object/peg position
        target_pos_full = None
        if obj_pddl_name in self.peg_target_positions:
            target_pos_full = self.peg_target_positions[obj_pddl_name]
        else:
            target_mjcf_body_name = self.object_id.get(obj_pddl_name)
            if not target_mjcf_body_name:
                print(f"Detector.over: Target PDDL name '{obj_pddl_name}' not in object_id map.")
                return False if not return_distance else np.inf
            
            target_body_id = self.env.sim.model.body_name2id(target_mjcf_body_name)
            if target_body_id == -1: return False if not return_distance else np.inf
            target_pos_full = self.env.sim.data.body_xpos[target_body_id]

        dist_xy = np.linalg.norm(gripper_pos[:2] - target_pos_full[:2])

        if return_distance:
            return dist_xy

        OVER_THRESHOLD = 0.0005 
        is_condition_met = dist_xy < OVER_THRESHOLD
        
        return is_condition_met

    def on(self, obj1, obj2):
        """Check if obj1 is on top of obj2 (either another cube or a peg)."""
        try:
            # Get obj1 position
            obj1_body_name = self.object_id.get(obj1)
            if not obj1_body_name:
                return False
            obj1_body_id = self.env.sim.model.body_name2id(obj1_body_name)
            obj1_pos = self.env.sim.data.body_xpos[obj1_body_id]
            
            # Get obj2 position
            if obj2 in self.peg_target_positions:
                # obj2 is a peg, use predefined position
                obj2_pos = self.peg_target_positions[obj2]
            else:
                # obj2 is another cube
                obj2_body_name = self.object_id.get(obj2)
                if not obj2_body_name:
                    return False
                obj2_body_id = self.env.sim.model.body_name2id(obj2_body_name)
                obj2_pos = self.env.sim.data.body_xpos[obj2_body_id]
            
            # Check if obj1 is on top of obj2
            dist_xy = np.linalg.norm(obj1_pos[:2] - obj2_pos[:2])
            dist_z = obj1_pos[2] - obj2_pos[2]
            
            # obj1 should be above obj2 and close in XY
            return bool(dist_xy < 0.03 and dist_z > 0.001 and dist_z < 0.055)
            
        except (ValueError, KeyError):
            return False

    def get_groundings(self, as_dict=False, binary_to_float=False, return_distance=False):
        """Build a grounding vector/dict for grasped, over, open, and on predicates."""
        # Start from base detector groundings when available.
        try:
            groundings = super().get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        except AttributeError:
            groundings = {}

        pddl_gripper = "gripper"
        all_manipulable_objects = self.objects
        all_placement_targets = self.pegs_pddl_names + self.objects

        for obj_name in all_manipulable_objects:
            val = self.grasped(obj_name, return_distance=return_distance)
            if not return_distance and binary_to_float: val = float(val)
            groundings[f'grasped({obj_name})'] = val

        for target_name in all_placement_targets:
            val = self.over(pddl_gripper, target_name, return_distance=return_distance)
            if not return_distance and binary_to_float: val = float(val)
            groundings[f'over({pddl_gripper},{target_name})'] = val
        
        val = self.open(pddl_gripper, return_distance=return_distance)
        if not return_distance and binary_to_float: val = float(val)
        groundings[f'open_gripper({pddl_gripper})'] = val

        # Evaluate stacking predicates for cube-on-peg and cube-on-cube.
        all_placement_targets = self.pegs_pddl_names + self.objects
        for obj_name in all_manipulable_objects:
            for target_name in all_placement_targets:
                if obj_name != target_name:
                    val = self.on(obj_name, target_name)
                    if not return_distance and binary_to_float: val = float(val)
                    groundings[f'on({obj_name},{target_name})'] = val

        # Keep keys stable if callers convert dict output to arrays.
        groundings = {str(k): v for k, v in groundings.items()}
        
        return dict(sorted(groundings.items())) if as_dict else np.array([v for k, v in sorted(groundings.items())])
