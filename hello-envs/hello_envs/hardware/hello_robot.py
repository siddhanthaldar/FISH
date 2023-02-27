import stretch_body.robot
import numpy as np
import PyKDL
import rospy
#from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R
import math
import time
import random
from utils import euler_to_quat, urdf_joint_to_kdl_joint, urdf_pose_to_kdl_frame, urdf_inertial_to_kdl_rbi, kdl_tree_from_urdf_model



 
joint_list = ["joint_fake","joint_lift","joint_arm_l3","joint_arm_l2","joint_arm_l1" ,"joint_arm_l0","joint_wrist_yaw","joint_wrist_pitch","joint_wrist_roll"]
STRETCH_GRIPPER = 40
LOWER_BOUND = 0

OVERRIDE_STATES = {}

class HelloRobot:
    
    def home(self, lift_pos):
        self.CURRENT_STATE = STRETCH_GRIPPER
        self.robot.lift.move_to(lift_pos)
        self.robot.end_of_arm.move_to('stretch_gripper',self.CURRENT_STATE)
        while self.robot.get_status()['arm']['pos']>0.041 or self.robot.get_status()['arm']['pos']<0.039:
            self.robot.arm.move_to(0.04)
            self.robot.push_command()
        self.robot.end_of_arm.move_to('wrist_yaw', 0.00)
        PITCH_VAL = -0.0
        self.robot.end_of_arm.move_to('wrist_pitch', PITCH_VAL)
        #NOTE: below code is to fix the pitch drift issue in current hello-robot. Remove it if there is no pitch drift issue
        OVERRIDE_STATES['wrist_pitch'] = PITCH_VAL
        self.robot.end_of_arm.move_to('wrist_roll', 0.0)
        self.base_motion = 0
        
        self.robot.push_command()

    def setup_kdl(self, urdf_file = '/home/hello-robot/robot-files/stretch_nobase_raised.urdf'):
        self.joints = {'joint_fake':0}
        robot_model = URDF.from_xml_file(urdf_file)
        kdl_tree = kdl_tree_from_urdf_model(robot_model)
        self.arm_chain = kdl_tree.getChain('base_link', 'link_raised_gripper')
        self.joint_array = PyKDL.JntArray(self.arm_chain.getNrOfJoints())

        # Forward kinematics
        self.fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self.arm_chain)
        # Inverse Kinematics
        self.ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self.arm_chain)
        self.ik_p_kdl = PyKDL.ChainIkSolverPos_NR(self.arm_chain, self.fk_p_kdl, self.ik_v_kdl) 


    def __init__(self, lift_pos = 0.844):
        #Initializing ROS node
        try:
            rospy.init_node('hello_robot_node')
        except:
            pass
        #setting up the robot
        
        self.robot = stretch_body.robot.Robot()
        self.robot.startup()

        # Initializing the robot
        self.base_x = self.robot.base.status['x']
        self.base_y = self.robot.base.status['y']

        self.robot.push_command()

        self.home(lift_pos=lift_pos)
    
        time.sleep(2)

        # Constraining the robots movement
        self.clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

        # Joint dictionary for Kinematics
        self.setup_kdl()

    def updateJoints(self):
        #Update the joint state values in 'self.joints' using hellorobot api calls
        print('x, y:', self.robot.base.status['x'], self.robot.base.status['y'])
        origin_dist = math.sqrt((self.base_y - self.robot.base.status['y'])**2+(self.base_x - self.robot.base.status['x'])**2)
        print('orig_dist:', origin_dist)
        self.joints['joint_fake'] = origin_dist
        
        self.joints['joint_lift'] = self.robot.lift.status['pos']
        
        armPos = self.robot.arm.status['pos']
        self.joints['joint_arm_l3'] = armPos / 4.0
        self.joints['joint_arm_l2'] = armPos / 4.0
        self.joints['joint_arm_l1'] = armPos / 4.0
        self.joints['joint_arm_l0'] = armPos / 4.0
        
        self.joints['joint_wrist_yaw'] = self.robot.end_of_arm.status['wrist_yaw']['pos']
        self.joints['joint_wrist_roll'] = self.robot.end_of_arm.status['wrist_roll']['pos']
        self.joints['joint_wrist_pitch'] = OVERRIDE_STATES.get('wrist_pitch', self.robot.end_of_arm.status['wrist_pitch']['pos'])

        self.joints['joint_stretch_gripper'] = self.robot.end_of_arm.status['stretch_gripper']['pos']  


    def move(self, joints, gripper):
        # update the robot joints to the new values from 'joints'
        joints['joint_fake'] = self.clamp(joints['joint_fake'], 0.0002, 0.20)
        self.base_motion += joints['joint_fake']-self.joints['joint_fake']
        self.robot.base.translate_by(joints['joint_fake']-self.joints['joint_fake'], 5)
        self.robot.arm.move_to(joints['joint_arm_l3'] + 
                            joints['joint_arm_l2'] + 
                            joints['joint_arm_l1'] + 
                            joints['joint_arm_l0'])
        
        #yaw, pitch, roll limits 
        self.robot.end_of_arm.move_to('wrist_yaw', self.clamp(joints['joint_wrist_yaw'], -0.4, 1.7))
        self.robot.end_of_arm.move_to('wrist_pitch', self.clamp(joints['joint_wrist_pitch'], -0.8, 0.2))
        #NOTE: belwo code is to fix the pitch drift issue in current hello-robot. Remove it if there is no pitch drift issue
        OVERRIDE_STATES['wrist_pitch'] = joints['joint_wrist_pitch']
        self.robot.end_of_arm.move_to('wrist_roll', self.clamp(joints['joint_wrist_roll'], -1.53, 1.53))
        self.CURRENT_STATE  = (STRETCH_GRIPPER-LOWER_BOUND)*gripper[0]
        self.robot.end_of_arm.move_to('stretch_gripper', self.CURRENT_STATE)
        if self.CURRENT_STATE<5.0:
            self.robot.end_of_arm.move_to('stretch_gripper', -20)

        self.robot.push_command()

        #sleeping to make sure all the joints are updated correctly (remove if not necessary)
        time.sleep(.7)
    
    def move_to_pose(self, translation_tensor, rotational_tensor, gripper):
        
        global CURRENT_STATE 
        translation = [translation_tensor[0], translation_tensor[1], translation_tensor[2]]
        rotation = rotational_tensor
        
        gripper_close = False
        
        if gripper_close: 
            print("sleeping")
            time.sleep(0.5)
        # move logic
        self.updateJoints()
        for joint_index in range(self.joint_array.rows()):
            self.joint_array[joint_index] = self.joints[joint_list[joint_index]]
        curr_pose = PyKDL.Frame()
        goal_pose = PyKDL.Frame()
        del_pose = PyKDL.Frame()
        self.fk_p_kdl.JntToCart(self.joint_array, curr_pose)

        curr_rot = R.from_quat(curr_pose.M.GetQuaternion()).as_dcm()
        rot_matrix = R.from_euler('xyz', rotation, degrees=False).as_dcm()


        del_rot = PyKDL.Rotation(PyKDL.Vector(rot_matrix[0][0], rot_matrix[1][0], rot_matrix[2][0]),
                                  PyKDL.Vector(rot_matrix[0][1], rot_matrix[1][1], rot_matrix[2][1]),
                                  PyKDL.Vector(rot_matrix[0][2], rot_matrix[1][2], rot_matrix[2][2]))
        del_trans = PyKDL.Vector(translation[0], translation[1], translation[2])
        del_pose.M = del_rot
        del_pose.p = del_trans
        goal_pose_new = curr_pose*del_pose
        
        seed_array = PyKDL.JntArray(self.arm_chain.getNrOfJoints())
        self.ik_p_kdl.CartToJnt(seed_array, goal_pose_new, self.joint_array)
        ik_joints = {}
        for joint_index in range(self.joint_array.rows()):
            ik_joints[joint_list[joint_index]] = self.joint_array[joint_index]

        test_pose = PyKDL.Frame()
        self.fk_p_kdl.JntToCart(self.joint_array, test_pose)

        self.move(ik_joints, gripper)

        self.robot.push_command()

        self.updateJoints()
        for joint_index in range(self.joint_array.rows()):
            self.joint_array[joint_index] = self.joints[joint_list[joint_index]]