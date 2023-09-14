from holodex.robot.allegro_kdl import AllegroKDL as AllegroKDLBase
import numpy as np


class AllegroKDL(AllegroKDLBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_fingertip_coords(self, joint_positions):
        index_coords = self.finger_forward_kinematics('index', joint_positions[:4])[0]
        middle_coords = self.finger_forward_kinematics('middle', joint_positions[4:8])[0]
        ring_coords = self.finger_forward_kinematics('ring', joint_positions[8:12])[0]
        thumb_coords = self.finger_forward_kinematics('thumb', joint_positions[12:16])[0]

        finger_tip_coords = np.hstack([index_coords, middle_coords, ring_coords, thumb_coords])
        return np.array(finger_tip_coords)
