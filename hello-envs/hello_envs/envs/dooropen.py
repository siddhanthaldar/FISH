from hello_envs.envs import hello_env

import time
import gym
from gym import spaces
from copy import deepcopy as copy
import cv2 
import numpy as np

class DoorOpenEnv(hello_env.HelloEnv):
    def __init__(self):
        # Reset limits
        self.reset_y_lims = [-0.095, -0.095]
        
        # Observation limits
        self.MIN_Z = -0.28
        self.MAX_Z = 0
        self.MAX_Y = 0
        self.MIN_Y = -0.11

        self.GLOBAL_XYZ = [0,0,0]
        super(DoorOpenEnv, self).__init__()
        self.action_space = spaces.Box(low = np.array([-1]*4,dtype=np.float32), 
									   high = np.array([1]*4,dtype=np.float32),
									   dtype = np.float32)

        self.amax = np.array([0.016178075877069577, 0.0057429156439002945, 0.03521448644975311, 0.9984233379364014])
        self.amin = np.array([-0.01679583449215663, -0.04048381295911532, -0.04069626674294205, 0.0017488393932580948])

        self.diff = self.amax - self.amin

    def reset(self):
        x = input("Retracting the arm... Press Enter to continue")
        if self.GLOBAL_XYZ[2] != 0:
            self._publish_instruction([0, 0, 0, 0.3])
            time.sleep(1)
            self._publish_instruction([0.2*0.26, 0, .2, 0.3])
            time.sleep(1)
            self.GLOBAL_XYZ[2] += 0.2
            self.GLOBAL_XYZ[0] += 0.2*0.26
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
        
        new_y = np.random.uniform(self.reset_y_lims[0], self.reset_y_lims[1])


        reset_y_action = np.array([0, new_y - self.GLOBAL_XYZ[1], 0, 1])
        self._publish_instruction(reset_y_action)
        time.sleep(4)

        self.GLOBAL_XYZ = [0,new_y,0]
        print("New Y:", new_y)
        obs['feature'] = None
        obs['pixels'] = self.get_obs()
        return obs
        

    def step(self, action):
            
           
            action = action * self.diff + self.amin

            # clipping action using the limits
            action[1] = np.clip(action[1], self.MIN_Y - self.GLOBAL_XYZ[1], self.MAX_Y - self.GLOBAL_XYZ[1])
            action[2] = np.clip(action[2], self.MIN_Z - self.GLOBAL_XYZ[2], self.MAX_Z - self.GLOBAL_XYZ[2])
            self.GLOBAL_XYZ[1] += action[1]
            self.GLOBAL_XYZ[2] += action[2]
            
            while self.prev_uid == self.uid:
                time.sleep(0.3)
            self.prev_uid = copy(self.uid)
            action[0] = 0.26*action[2]     #action transform to cancel noisy vertical motions
            self._publish_instruction(action)
            obs = {}
            obs['feature'] = None
            obs['pixels'] = self.get_obs()
            return obs, 0, False, {'is_success': False} #obs, reward, done, infos

    