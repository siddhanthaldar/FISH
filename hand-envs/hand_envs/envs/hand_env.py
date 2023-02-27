'''
gym like wrapper for Sridhar's Hand
'''
import gym
from gym import spaces
import cv2 
import numpy as np
import time

from holobot_api import DeployAPI
from holobot.robot.allegro.allegro_kdl import AllegroKDL


class HandEnv(gym.Env):
        def __init__(self, robot_name, host_address = "172.24.71.211", camera_num = 0):
            print(camera_num, "CAMERA_NUM")
            self.width = 224
            self.height = 224
            self.robot_name = robot_name
            self.hand = DeployAPI(host_address, required_data = {"rgb_idxs": [camera_num], "depth_idxs": []})
            self.observation_space = spaces.Box(low = np.array([0,0],dtype=np.float32), high = np.array([255,255],dtype=np.float32), dtype = np.float32)
            self.home_state = np.array([0,  .6597462e+00, .6, .6,
                                        0,  .6, .4, .5,
                                        0,  .6, .4, .5,
                                        -3.5524368e-05,  2.5796890e-04,  2.0717978e-02,  2.9551983e-04])
            self._robot = AllegroKDL()

        def init_hand(self): #makes the hand flat 
            while True: 
                try:
                    self.hand.send_robot_action({self.robot_name: self.home_state})
                    break 
                except:
                    pass

        def step(self, action):
            action *= (1.0/5) 
            try: 
                converted_action = self._robot.get_joint_state_from_coord(action[0:3], action[3:6], action[6:9], action[9:12], self.hand.get_robot_state()['allegro']['position'])
                self.hand.send_robot_action({self.robot_name: converted_action})
            except:
                print("IK error")
            
            obs = {}
            obs['features'] = self.hand.get_robot_state()
            obs['pixels'] = self._crop_resize(self.hand.get_rgb_images()[0][0], self.width, self.height)
            obs['depth'] = self.hand.get_depth_images()

            return obs, 0, False, {'is_success': False} #obs, reward, done, infos


        def render(self, mode='rbg_array', width=0, height=0):
            return self.hand.get_rgb_images()[0][0]

        def _crop_resize(self, img, width, height):
            img = img[:420, 627:1045]
            img = cv2.resize(img, (width, height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    
        def reset(self): 
            self.init_hand()
            obs = {}
            obs['features'] = self.hand.get_robot_state()
            obs['pixels'] = self._crop_resize(self.hand.get_rgb_images()[0][0], self.width, self.height)
            obs['depth'] = self.hand.get_depth_images()
            return obs


        def get_reward():
            pass