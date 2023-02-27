import time
import gym
from gym import spaces
from hand_envs.envs import hand_env

import cv2
import numpy as np

class BottleSpinEnv(hand_env.HandEnv):
	def __init__(self, robot_name, host_address = "172.24.71.211", camera_num = 0):
		hand_env.HandEnv.__init__(
			self,
			robot_name = robot_name,
			host_address = host_address,
			camera_num = camera_num
		)
		self.robot_name = robot_name
		self.width = 224
		self.height = 224
		self.min_action = -5
		self.max_action = 5

		self.home_state = np.array([-0.06855812,  0.71875733,  1.2080745 ,  0.49285522, -0.04829068,
									1.4122896 ,  0.86096025,  0.5427761 , -0.05214972,  1.4942853 ,
									0.6980798 ,  0.72770816,  1.3649694 ,  1.276296 ,  1.239179  ,
									0.5427955 ]) 

		self.action_space = spaces.Box(low = np.array([-1]*12,dtype=np.float32), 
									   high = np.array([1]*12,dtype=np.float32),
									   dtype = np.float32)

	def _crop_resize(self, img, width, height):
			img = img[275:385, 675:810]
			img = cv2.resize(img, (width, height))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			return img

	def init_hand(self):
			while True: 
				try:
					self.hand.send_robot_action({self.robot_name: self.home_state})
					break 
				except:
					pass

	def step(self, action):
			action = np.clip(action, self.min_action, self.max_action)
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
		self.hand.send_robot_action({self.robot_name:  self.home_state})
		time.sleep(1)

		obs = {}
		obs['features'] = self.hand.get_robot_state()
		obs['pixels'] = self._crop_resize(self.hand.get_rgb_images()[0][0], self.width, self.height)
		obs['depth'] = self.hand.get_depth_images()
		return obs