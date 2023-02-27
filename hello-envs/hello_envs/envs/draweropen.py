from hello_envs.envs import hello_env

import time
import gym
from gym import spaces
from copy import deepcopy as copy
import cv2 
import numpy as np

class DrawerOpenEnv(hello_env.HelloEnv):
    def __init__(self):
        # Reset limits
        val = -0.105
        self.reset_x_lims = [val, val]
        self.camera_angle = 15
        # Observation limits
        self.MIN_Z = -0.28
        self.MAX_Z = 0
        self.MIN_X = -0.14
        self.MAX_X = 0

        self.GLOBAL_XYZ = [0,0,0]
        super(DrawerOpenEnv, self).__init__()
        self.action_space = spaces.Box(low = np.array([-1]*4,dtype=np.float32), 
									   high = np.array([1]*4,dtype=np.float32),
									   dtype = np.float32)

        self.amax = np.array([0.02045020263405146, 0.003979210501773373, 0.062499330841923086, 0.9947434663772583])
        self.amin = np.array([-0.022169565804717052, -0.008110477391640981, -0.05382667709200192, 0.002753613982349634])

        self.diff = self.amax - self.amin

    def reset(self):
        x = input("Retracting the arm... Press Enter to continue")
        if self.GLOBAL_XYZ[2] != 0:
            self._publish_instruction([0, 0, 0, 0.3])
            time.sleep(1)
            self._publish_instruction([0.2*0.26, 0, 0.2*0.966, 0.3])
            time.sleep(1)
            self.GLOBAL_XYZ[2] += 0.2
        for i in range(3):
            self._publish_home()
            time.sleep(1)
        obs = {}
       
        x = input("Press Enter to continue... after reseting env")
        if x == 'c':
            x = input("Press Enter to continue... after reseting robot")
            for _ in range(3):
                self._publish_home()
                time.sleep(1)
                 
            self.GLOBAL_XYZ = [0,0,0]
        
        new_x = np.random.uniform(self.reset_x_lims[0], self.reset_x_lims[1])

        reset_y_action = np.array([(new_x - self.GLOBAL_XYZ[0])*0.966, 0, -(new_x - self.GLOBAL_XYZ[0])*0.26, 1]) 
        self._publish_instruction(reset_y_action)
        time.sleep(4)

        self.GLOBAL_XYZ = [new_x,0,0]
        print("New X:", new_x)
        obs['feature'] = None
        obs['pixels'] = self.get_obs()
        return obs
        

    def step(self, action):
            
            action = action * self.diff + self.amin

            X_VECTOR = np.array([np.cos(np.deg2rad(self.camera_angle)), 0, -np.sin(np.deg2rad(self.camera_angle))])
            dot_action = np.dot(action[:3], X_VECTOR)
            
            clipped_dot = np.clip(dot_action, self.MIN_X - self.GLOBAL_XYZ[0], self.MAX_X - self.GLOBAL_XYZ[0])
            del_dot = dot_action - clipped_dot
            
            self.GLOBAL_XYZ[0] += clipped_dot
            
            while self.prev_uid == self.uid:
                time.sleep(0.3)
            self.prev_uid = copy(self.uid)

            dot_action = np.dot(action[:3], np.array([0, 0.966, 0.26]))
            dot_action2 = np.dot(action[:3], np.array([0, 0.26, -0.966]))

            action[:3] = action[:3] + 0.35*dot_action2*np.array([0, 0.26, -0.966])
 
            self._publish_instruction(action)
            obs = {}
            obs['feature'] = None
            obs['pixels'] = self.get_obs()
            return obs, 0, False, {'is_success': False} #obs, reward, done, infos

    