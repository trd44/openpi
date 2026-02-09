# --------------------------------------------------------------------------------------
# Observation preprocessing
# --------------------------------------------------------------------------------------
import numpy as np
import math

from typing import Dict, Any

from openpi_client import image_tools

class ObsProcessor:
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
            # "robot0_gripper_qpos",
            # 'robot0_joint_pos_cos',
            # 'robot0_joint_pos_sin',
            # "gripper0_left_inner_finger",
            # "gripper0_right_inner_finger"
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

        joint_cos = obs['robot0_joint_pos_cos']
        joint_sin = obs['robot0_joint_pos_sin']
        joint_state = np.arctan2(joint_sin, joint_cos).astype(np.float32)

        # left_finger_pos = self.env_manager.env.sim.data.body_xpos[
        #     self.env_manager.env.sim.model.body_name2id("gripper0_left_inner_finger")
        # ]
        # right_finger_pos = self.env_manager.env.sim.data.body_xpos[
        #     self.env_manager.env.sim.model.body_name2id("gripper0_right_inner_finger")
        # ]
        # gripper_width = float(np.linalg.norm(left_finger_pos - right_finger_pos))
        eef_pos = obs.get('robot0_eef_pos', np.zeros(3, dtype=np.float32))
        eef_quat = obs.get('robot0_eef_quat', np.array([0., 0., 0., 1.], dtype=np.float32))
        eef_axis_angle = _quat2axisangle(eef_quat)


        j1 = self.env_manager.env.sim.data.get_joint_qpos("gripper0_finger_joint1")
        j2 = self.env_manager.env.sim.data.get_joint_qpos("gripper0_finger_joint2")
        eef_gripper = np.array([j1,j2], dtype=np.float32)    

        eef_state = np.concatenate((eef_pos, eef_axis_angle, eef_gripper)).astype(np.float32)

        # State is the concatenation of joint state and gripper opening
        state = np.concatenate((joint_state, eef_gripper)).astype(np.float32)
        
        return {
            "image": img,
            "wrist_image": wrist_img,
            "state": eef_state,
            "raw_agentview": obs["agentview_image"]  # For video recording
        }
    
def _quat2axisangle(quat):
    """
    Copied from robosuite: 
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
