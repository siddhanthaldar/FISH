from hello_envs.envs import hello_env

import time
import gym
from gym import spaces
from copy import deepcopy as copy
import cv2 
import numpy as np
from std_msgs.msg import Float64MultiArray

class LightSwitchEnv(hello_env.HelloEnv):
    def __init__(self):
        # Reset limits
        self.reset_y_lims = [-0.11, 0.0]
        
        # Observation limits
        self.MIN_Z = -0.28
        self.MAX_Z = 0
        self.MAX_Y = 0
        self.MIN_Y = -0.14

        self.GLOBAL_XYZ = [0,0,0]
        super(LightSwitchEnv, self).__init__()
        self.action_space = spaces.Box(low = np.array([-1]*3,dtype=np.float32), 
									   high = np.array([1]*3,dtype=np.float32),
									   dtype = np.float32)


        self.amax = np.array([0.004809487924420208, 0.006209186588403828, 0.03592271251894191])
        self.amin = np.array([-0.011282012505537179, -0.010504665051958198, -0.060072801876953845])

        self.diff = self.amax - self.amin
    
    def _publish_home(self):
            home_publisher_list = Float64MultiArray()
            home_publisher_list.layout.data_offset = self.uid
            home_publisher_list.data = [0]
            self.home_publisher.publish(home_publisher_list)

    def reset(self):
        x = input("Retracting the arm... Press Enter to continue")
        if self.GLOBAL_XYZ[2] != 0:
            self._publish_instruction([0, 0, 0, 0.0])
            time.sleep(1)
            self._publish_instruction([0.2*0.26, 0, .2, 0.0])
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


        reset_y_action = np.array([0, new_y - self.GLOBAL_XYZ[1], 0, 0])
        self._publish_instruction(reset_y_action)
        time.sleep(4)

        self.GLOBAL_XYZ = [0,new_y,0]
        print("New Y:", new_y)
        obs['feature'] = None
        obs['pixels'] = self.get_obs()
        return obs
        

    def step(self, action):
            
            
            action = action * self.diff + self.amin
            
            action[1] = np.clip(action[1], self.MIN_Y - self.GLOBAL_XYZ[1], self.MAX_Y - self.GLOBAL_XYZ[1])
            action[2] = np.clip(action[2], self.MIN_Z - self.GLOBAL_XYZ[2], self.MAX_Z - self.GLOBAL_XYZ[2])
            self.GLOBAL_XYZ[1] += action[1]
            self.GLOBAL_XYZ[2] += action[2]
            
            
            while self.prev_uid == self.uid:
                time.sleep(0.3)
            self.prev_uid = copy(self.uid)
            action[0] = 0.26*action[2]   #action transform to cancel noisy vertical motions

            action = np.append(action, 0)    
            self._publish_instruction(action)
            obs = {}
            obs['feature'] = None
            obs['pixels'] = self.get_obs()
            return obs, 0, False, {'is_success': False} #obs, reward, done, infos
