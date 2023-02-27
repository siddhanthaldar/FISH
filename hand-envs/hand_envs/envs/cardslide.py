import time
import gym
from gym import spaces
from hand_envs.envs import hand_env

import cv2
import numpy as np

class CardSlideEnv(hand_env.HandEnv):
	def __init__(self, robot_name, host_address = "172.24.71.211", camera_num = 0):
		hand_env.HandEnv.__init__(
			self,
			robot_name = robot_name,
			host_address = host_address,
			camera_num = camera_num
		)
		self.width = 84
		self.height = 84
		self.home_state = np.array([-0.2240,  0.7161,  1.6306,  0.4943, -0.0916,  0.6534,  1.7042,  0.6300,
									-0.0434,  0.4351,  1.1561,  0.9907,  0.7882,  1.8104,  1.1842,  0.2748])

		
		self.robot_name = robot_name
		self.action_space = spaces.Box(low = np.array([-1]*12,dtype=np.float32), 
									   high = np.array([1]*12,dtype=np.float32),
									   dtype = np.float32)

	def init_hand(self): #makes the hand flat 
			while True: 
				try:
					self.hand.send_robot_action({self.robot_name: self.home_state})
					break 
				except:
					pass

	def _crop_resize(self, img, width, height):
			img = cv2.resize(img, (width, height))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			return img

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


	def reset(self): 
		self.init_hand()
		time.sleep(2)

		obs = {}
		obs['features'] = self.hand.get_robot_state()
		obs['pixels'] = self._crop_resize(self.hand.get_rgb_images()[0][0], self.width, self.height)
		obs['depth'] = self.hand.get_depth_images()
		return obs