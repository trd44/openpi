# In your dataset_making/panda_hanoi_detector.py

import numpy as np
from robosuite.utils.detector import HanoiDetector as _BaseHanoiDetector # Keep this inheritance

class PandaHanoiDetector(_BaseHanoiDetector):
    def __init__(self, env):
        super().__init__(env) # Call the base class's __init__
        robot_model_name = getattr(env.robots[0].robot_model, 'model_type', '') # Corrected attribute
        self.is_panda = 'Panda' in robot_model_name # Corrected variable name
        
        gripper_model = self.env.robots[0].gripper
        # Ensure important_geoms are correctly fetched; these names are standard in Robosuite
        self.left_tip = gripper_model.important_geoms.get("left_fingerpad") 
        self.right_tip = gripper_model.important_geoms.get("right_fingerpad")

        self.objects = ['cube1', 'cube2', 'cube3'] # PDDL names for cubes
        self.pegs_pddl_names = ['peg1', 'peg2', 'peg3'] # PDDL names for pegs

        # This map is crucial: PDDL name -> MuJoCo body name
        self.object_id = {
            'cube1': 'cube1_main', # Ensure these _main suffixes match your XML body names
            'cube2': 'cube2_main',
            'cube3': 'cube3_main',
            'peg1':  'peg1_main', # Ensure these are actual body names for pegs in your env XML
            'peg2':  'peg2_main',
            'peg3':  'peg3_main',
            # It's good practice to also map the gripper's PDDL name if used in predicates
            'gripper': 'gripper0_eef' # Assuming 'gripper' is the PDDL name
        }

        # Store peg center positions from the environment if available
        # These might be more stable than querying sim.data.body_xpos if pegs are visual/static
        # The log showed peg3 target tgt = [0.1, 0.27, 0.798]
        # Your Hanoi env code had self.pegs_xy_center
        if hasattr(env, 'pegs_xy_center'): # Check if env has this attribute
            self.peg_target_positions = {
                'peg1': np.array(env.pegs_xy_center[0]),
                'peg2': np.array(env.pegs_xy_center[1]),
                'peg3': np.array(env.pegs_xy_center[2]),
            }
        else:
            self.peg_target_positions = {} # Fallback or populate from sim later
            print("Warning: env.pegs_xy_center not found in PandaHanoiDetector init.")


    def open(self, gripper_pddl_name, return_distance=False): # gripper_pddl_name likely "gripper"
        # Your existing open logic seems fine for Panda based on qpos
        j1 = float(self.env.sim.data.get_joint_qpos('gripper0_finger_joint1'))
        j2 = float(self.env.sim.data.get_joint_qpos('gripper0_finger_joint2'))
        qpos_diff = abs(j1 - j2)
        
        if return_distance:
            return qpos_diff
        return qpos_diff > 0.075 # Threshold for "open"

    def grasped(self, obj_pddl_name, return_distance=False):
        # Your existing grasped logic
        target_mjcf_body_name = self.object_id.get(obj_pddl_name)
        if not target_mjcf_body_name: return False
        
        body_id = self.env.sim.model.body_name2id(target_mjcf_body_name)
        if body_id == -1: return False

        model = self.env.sim.model
        geom_ids = [i for i, b_id in enumerate(model.geom_bodyid) if b_id == body_id]
        obj_mjcf_geom_names = [model.geom_id2name(g_id) for g_id in geom_ids if model.geom_id2name(g_id) is not None]

        # Ensure left_tip and right_tip are valid geom names
        if self.left_tip is None or self.right_tip is None:
            # print("Warning: Gripper fingerpad geoms not found for grasped check.")
            return False

        left_contact  = self.env.check_contact(self.left_tip,  obj_mjcf_geom_names)
        right_contact = self.env.check_contact(self.right_tip, obj_mjcf_geom_names)
        is_grasped = bool(left_contact or right_contact)

        if return_distance:
            # For distance, perhaps return 0.0 if grasped, 1.0 if not, or a more continuous metric if available
            return 0.0 if is_grasped else 1.0 
        return is_grasped

    def over(self, gripper_pddl_name, obj_pddl_name, return_distance=False):
        gripper_mjcf_body_name = self.object_id.get(gripper_pddl_name) # Should be 'gripper0_eef'
        if not gripper_mjcf_body_name:
            # print(f"Detector.over: Gripper PDDL name '{gripper_pddl_name}' not in object_id map.")
            return False if not return_distance else np.inf
        
        gripper_body_id = self.env.sim.model.body_name2id(gripper_mjcf_body_name)
        if gripper_body_id == -1: return False if not return_distance else np.inf # Should not happen
        gripper_pos = self.env.sim.data.body_xpos[gripper_body_id]

        # Get target object/peg position
        target_pos_full = None
        if obj_pddl_name in self.peg_target_positions: # Use predefined positions for pegs if available
            target_pos_full = self.peg_target_positions[obj_pddl_name]
            # print(f"Using PREDEFINED peg target for {obj_pddl_name}: {target_pos_full[:2]}")
        else: # For other objects like cubes, get current simulation position
            target_mjcf_body_name = self.object_id.get(obj_pddl_name)
            if not target_mjcf_body_name:
                print(f"Detector.over: Target PDDL name '{obj_pddl_name}' not in object_id map.")
                return False if not return_distance else np.inf
            
            target_body_id = self.env.sim.model.body_name2id(target_mjcf_body_name)
            if target_body_id == -1: return False if not return_distance else np.inf
            target_pos_full = self.env.sim.data.body_xpos[target_body_id]
            # print(f"Using SIMULATED target pos for {obj_pddl_name}: {target_pos_full[:2]}")

        dist_xy = np.linalg.norm(gripper_pos[:2] - target_pos_full[:2])

        if return_distance:
            return dist_xy

        # --- ADJUST THIS THRESHOLD ---
        OVER_THRESHOLD = 0.0005 
        is_condition_met = dist_xy < OVER_THRESHOLD
        
        # print(f"  PandaHanoiDetector.over('{gripper_pddl_name}', '{obj_pddl_name}'): dist_xy={dist_xy:.4f}, thresh={OVER_THRESHOLD}, met={is_condition_met}")
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
        # Start with groundings from the base class, if it provides any useful ones
        # Or initialize an empty dict if _BaseHanoiDetector.get_groundings is not suitable/available
        try:
            groundings = super().get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        except AttributeError: # If base class doesn't have it or it's not suitable
            groundings = {}

        # PDDL entities
        pddl_gripper = "gripper" # The PDDL name for your gripper
        all_manipulable_objects = self.objects # Cubes
        all_placement_targets = self.pegs_pddl_names + self.objects # Pegs + Cubes (for stacking)

        # Grasped predicates
        for obj_name in all_manipulable_objects:
            val = self.grasped(obj_name, return_distance=return_distance)
            if not return_distance and binary_to_float: val = float(val)
            groundings[f'grasped({obj_name})'] = val
            # For compatibility with some planners, also maybe:
            # groundings[f'holding({pddl_gripper},{obj_name})'] = val


        # Over predicates
        for target_name in all_placement_targets: # Cubes and Pegs
            val = self.over(pddl_gripper, target_name, return_distance=return_distance)
            if not return_distance and binary_to_float: val = float(val)
            groundings[f'over({pddl_gripper},{target_name})'] = val
        
        # Open gripper predicate
        val = self.open(pddl_gripper, return_distance=return_distance)
        if not return_distance and binary_to_float: val = float(val)
        groundings[f'open_gripper({pddl_gripper})'] = val # Or just open() or handempty()

        # On predicates - check if objects are placed on pegs or stacked on other objects
        all_placement_targets = self.pegs_pddl_names + self.objects  # Pegs + Cubes (for stacking)
        for obj_name in all_manipulable_objects:
            for target_name in all_placement_targets:
                if obj_name != target_name:  # Don't check if object is on itself
                    val = self.on(obj_name, target_name)
                    if not return_distance and binary_to_float: val = float(val)
                    groundings[f'on({obj_name},{target_name})'] = val

        # Ensure all keys are strings if converting to array later
        groundings = {str(k): v for k, v in groundings.items()}
        
        return dict(sorted(groundings.items())) if as_dict else np.array([v for k, v in sorted(groundings.items())])

# class PandaHanoiDetector(_BaseHanoiDetector):
#     """HanoiDetector subclass that supports both Kinova and Panda grippers."""
#     def __init__(self, env):
#         super().__init__(env)
#         # Detect if the robot in this env is a Panda
#         robot_model = getattr(env.robots[0].robot_model, 'model_type', '')
#         self.is_panda = 'Panda' in robot_model
#         gripper_model = self.env.robots[0].gripper
#         self.left_tip  = gripper_model.important_geoms.get("left_fingerpad")
#         self.right_tip = gripper_model.important_geoms.get("right_fingerpad")

#         self.objects = ['cube1', 'cube2', 'cube3']
#         self.object_id = {
#             'cube1': 'cube1_main',
#             'cube2': 'cube2_main',
#             'cube3': 'cube3_main',
#             'peg1':  'peg1_main',
#             'peg2':  'peg2_main',
#             'peg3':  'peg3_main',
#         }


#     def open(self, gripper, return_distance=False):
#         """
#         Override open(): Panda uses joint qpos, Kinova uses built-in logic.
#         """
#         j1 = float(self.env.sim.data.get_joint_qpos('gripper0_finger_joint1'))
#         j2 = float(self.env.sim.data.get_joint_qpos('gripper0_finger_joint2'))
#         qpos =  abs(j1 - j2)         
#         # print("qpos = ", qpos)
#         return qpos > 0.075  
#         # if self.is_panda:
#         #     # Panda: read finger joint position
#         #     j1 = float(self.env.sim.data.get_joint_qpos('gripper0_finger_joint1'))
#         #     j2 = float(self.env.sim.data.get_joint_qpos('gripper0_finger_joint2'))
#         #     qpos =  abs(j1 - j2)         
#         #     print("qpos = ", qpos)   
#         #     if return_distance:
#         #         return qpos
#         #     return qpos > 0.01
#         # else:
#         #     # Kinova: fallback to base class implementation
#         #     return super().open(gripper, return_distance)

#     def grasped(self, obj, return_distance=False):
#         """
#         Returns True if either the left or right fingertip is in contact
#         with the specified object.
#         """
#         # Determine the body name and id for the object
#         body_name = self.object_id[obj]
#         body_id = self.env.sim.model.body_name2id(body_name)

#         # Gather all geom names belonging to this body
#         model = self.env.sim.model
#         geom_ids = [i for i, b in enumerate(model.geom_bodyid) if b == body_id]
#         obj_geoms = [model.geom_id2name(g) for g in geom_ids]

#         # Check for contact between each fingertip and the object's geoms
#         left_contact  = self.env.check_contact(self.left_tip,  obj_geoms)
#         right_contact = self.env.check_contact(self.right_tip, obj_geoms)
#         is_grasped = bool(left_contact or right_contact)

#         if return_distance:
#             return float(is_grasped)
#         return is_grasped

# # import numpy as np

# # class HanoiDetector:
# #     def __init__(self, env):
# #         self.env = env
# #          # Determine robot type from the first robot's model
# #         robot_model = getattr(env, 'robots', [None])[0]
# #         model_type = getattr(robot_model, 'robot_model', None)
# #         model_name = getattr(model_type, 'model_type', '') if model_type else ''
# #         if 'Kinova' in model_name:
# #             self.robot = 'kinova'
# #         elif 'Panda' in model_name:
# #             self.robot = 'panda'
# #         else:
# #             self.robot = 'unknown'

# #         self.objects = ['cube1', 'cube2', 'cube3']
# #         self.object_id = {'cube1': 'cube1_main', 'cube2': 'cube2_main', 'cube3': 'cube3_main', 'peg1': 'peg1_main', 'peg2': 'peg2_main', 'peg3': 'peg3_main'}
# #         self.object_areas = ['peg1', 'peg2', 'peg3']
# #         self.area_pos = {'peg1': self.env.pegs_xy_center[0], 'peg2': self.env.pegs_xy_center[1], 'peg3': self.env.pegs_xy_center[2]}
# #         self.grippers_areas = ['pick', 'drop', 'activate', 'lightswitch']
# #         self.grippers = ['gripper']
# #         self.area_size = self.env.peg_radius
# #         self.max_distance = 10 #max distance for the robotic arm in meters

# #     def at(self, obj, area, return_distance=False):
# #         obj_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj]]
# #         dist = np.linalg.norm(obj_pos - self.area_pos[area])
# #         if return_distance:
# #             return dist
# #         else:
# #             return bool(dist < self.area_size)

# #     def select_object(self, obj_name):
# #         """
# #         Selects the object tuple (object, object name) from self.objects using one of the strings ["milk", "bread", "cereal", "can"],
# #         ignoring the caps from the first letter in the self.obj_names.
# #         """
# #         obj_name = obj_name.lower()
# #         for obj, name in zip(self.env.objects, self.env.obj_names):
# #             if name.startswith(obj_name):
# #                 return obj
# #         return None

# #     def grasped(self, obj):
# #         """
# #         Return True if gripper is grasping the given object.
# #         """
# #         if self.robot == 'panda':
# #             # Panda: check contact via gripper.important_geoms
# #             gripper = self.env.robots[0].gripper
# #             obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
# #             obj_geoms = gripper.important_geoms
# #             # geom names may vary; assume left/right fingerpads
# #             g_left = gripper.important_geoms.get('left_fingerpad')
# #             g_right= gripper.important_geoms.get('right_fingerpad')
# #             # check contacts
# #             if g_left and g_right:
# #                 return self.env.check_contact(g_left, [obj_body]) and \
# #                        self.env.check_contact(g_right, [obj_body])
# #             return False
# #         else:
# #             # Kinova: original collision-based logic
# #             gripper = self.env.robots[0].gripper
# #             object_geoms = gripper.contact_geoms
# #             o_geoms = object_geoms if isinstance(object_geoms, (list, tuple)) else [object_geoms]
# #             # gripper pads
# #             g_geoms = [
# #                 gripper.important_geoms['left_inner_finger'],
# #                 gripper.important_geoms['right_inner_finger']
# #             ]
# #             # ensure both sides in contact
# #             for g in g_geoms:
# #                 if not self.env.check_contact(g, o_geoms):
# #                     return False
# #             return True

#     def over(self, gripper, obj, return_distance=False):
#         obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
#         if gripper == 'gripper':
#             gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
#             print("obj = ", obj)
#             print("self.object_areas = ", self.object_areas)
#             if obj in self.object_areas:
#                 obj_pos = self.area_pos[obj]
#             else:
#                 obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
#             dist_xy = np.linalg.norm(gripper_pos[:-1] - obj_pos[:-1])
#             if return_distance:
#                 return dist_xy
#             else:
#                 return bool(dist_xy < 0.004)#bool(dist_xy < 0.02)#bool(dist_xy < 0.004)#return bool(dist_xy < 0.004)
    
#     def at_grab_level(self, gripper, obj, return_distance=False):
#         obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
#         if gripper == 'gripper':
#             gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
#             obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
#             dist_z = np.linalg.norm(gripper_pos[2] - obj_pos[2])
#             if return_distance:
#                 return dist_z
#             else:
#                 return bool(dist_z < 0.005)
    
#     def on(self, obj1, obj2):
#         obj1_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj1]]
#         if obj2 in self.object_areas:
#             obj2_pos = self.area_pos[obj2]
#         else:
#             obj2_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj2]]
#         dist_xyz = np.linalg.norm(obj1_pos - obj2_pos)
#         dist_xy = np.linalg.norm(obj1_pos[:-1] - obj2_pos[:-1])
#         dist_z = np.linalg.norm(obj1_pos[2] - obj2_pos[2])
#         return bool(dist_xy < 0.03 and obj1_pos[2] > obj2_pos[2]+0.001 and dist_z < 0.055)
    
#     def clear(self, obj):
#         for other_obj in self.objects:
#             if other_obj != obj:
#                 if self.on(other_obj, obj):
#                     return False
#         return True
    
#     def open(self, gripper, return_distance=False):
#         """
#         Return True if gripper is open. For Kinova uses inner finger bodies; for Panda uses joint qpos.
#         """
#         if self.robot == 'panda':
#             # Panda: use joint position of the first finger joint
#             qpos = float(self.env.sim.data.get_joint_qpos('robot0_finger_joint1'))
#             if return_distance:
#                 return qpos
#             # threshold (tune as needed)
#             return bool(qpos > 0.03)
#         else:
#             # Kinova or default: use body distances
#             left = self.env.sim.model.body_name2id('gripper0_left_inner_finger')
#             right = self.env.sim.model.body_name2id('gripper0_right_inner_finger')
#             aper = np.linalg.norm(
#                 self.env.sim.data.body_xpos[left] - self.env.sim.data.body_xpos[right]
#             )
#             if return_distance:
#                 return aper
#             return bool(aper > 0.13)
    
#     def picked_up(self, obj, return_distance=False):
#         active_obj = self.select_object(obj)
#         z_target = self.env.table_offset[2] + 0.35
#         object_z_loc = self.env.sim.data.body_xpos[self.env.obj_body_id[active_obj.name]][2]
#         z_dist = z_target - object_z_loc
#         if return_distance:
#             return z_dist
#         else:
#             return bool(z_dist < 0.15)

#     def get_groundings(self, as_dict=False, binary_to_float=False, return_distance=False):
            
#             groundings = {}
    
#             # Check if the gripper is grasping each object
#             for obj in self.objects:
#                 groundings[f'grasped({obj})'] = self.grasped(obj)
#                 # grasped_value = self.grasped(obj)
#                 # if binary_to_float:
#                 #     grasped_value = float(grasped_value)
#                 # groundings[f'grasped({obj})'] = grasped_value

#             # Check if the gripper is over each object
#             for gripper in ['gripper']:
#                 for obj in self.objects+self.object_areas:
#                     over_value = self.over(gripper, obj, return_distance=return_distance)
#                     if return_distance:
#                         over_value = over_value / self.max_distance
#                     if binary_to_float:
#                         over_value = float(over_value)
#                     groundings[f'over({gripper},{obj})'] = over_value

#             # Check if the gripper is at the same height as each object
#             for gripper in ['gripper']:
#                 for obj in self.objects:
#                     at_grab_level_value = self.at_grab_level(gripper, obj, return_distance=return_distance)
#                     if return_distance:
#                         at_grab_level_value = at_grab_level_value / self.max_distance
#                     if binary_to_float:
#                         at_grab_level_value = float(at_grab_level_value)
#                     groundings[f'at_grab_level({gripper},{obj})'] = at_grab_level_value

#             # Check if each object is on another object
#             for obj1 in self.objects:
#                 for obj2 in self.objects+self.object_areas:
#                     if obj1 != obj2:
#                         on_value = self.on(obj1, obj2)
#                         if binary_to_float:
#                             on_value = float(on_value)
#                         groundings[f'on({obj1},{obj2})'] = on_value

#             # Check if each object is clear
#             for obj in self.objects+self.object_areas:
#                 clear_value = self.clear(obj)
#                 if binary_to_float:
#                     clear_value = float(clear_value)
#                 groundings[f'clear({obj})'] = clear_value
            
#             # Check if the gripper is open
#             gripper_open_value = self.open('gripper', return_distance=return_distance)
#             if binary_to_float:
#                 gripper_open_value = float(gripper_open_value)
#             groundings['open_gripper(gripper)'] = gripper_open_value

#             # Check if an object has been picked up
#             for obj in self.objects:
#                 picked_up_value = self.picked_up(obj, return_distance=return_distance)
#                 if return_distance:
#                     picked_up_value = picked_up_value / self.max_distance
#                 if binary_to_float:
#                     picked_up_value = float(picked_up_value)
#                 groundings[f'picked_up({obj})'] = picked_up_value

#             return dict(sorted(groundings.items())) if as_dict else np.asarray([v for k, v in sorted(groundings.items())])
    
#     def dict_to_array(self, groundings):
#         return np.asarray([v for k, v in sorted(groundings.items())])
