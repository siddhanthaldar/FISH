import time
import gym
from gym import spaces
from hand_envs.envs import hand_env

import cv2
import numpy as np

class CubeFlipEnv(hand_env.HandEnv):
	def __init__(self, robot_name, host_address = "172.24.71.211", camera_num = 0):
		hand_env.HandEnv.__init__(
			self,
			robot_name = robot_name,
			host_address = host_address,
			camera_num = camera_num
		)
		# For further back cube 
		self.fingers_forward_1 = np.array([0,  .3, .3, .3,
										0,  2.5, 1, 1.5,
										0,  1.5, .6, .6,
										 -3.5524368e-05,  2.5796890e-04,  2.0717978e-02,  2.9551983e-04])
		self.fingers_forward_2 = np.array([0,  .3, .3, .3,
										0,  1.5, .6, .6,
										0,  1.5, .6, .6,
										 -3.5524368e-05,  2.5796890e-04,  2.0717978e-02,  2.9551983e-04])
		self.fingers_forward_3 = np.array([0,  .3, .3, .3,
										0,  1.5, .6, .6,
										0,  1.5, .6, .6,
										-3.5524368e-05,  2.5796890e-04,  2.0717978e-02,  2.9551983e-04])

		self.fingers_forward = np.array([0,  2, .7, .6,
										0,  2, .6, .5,
										0,  2, .6, .5,
										-3.5524368e-05,  2.5796890e-04,  2.0717978e-02,  2.9551983e-04])

		self.action_space = spaces.Box(low = np.array([-1]*12,dtype=np.float32), 
									   high = np.array([1]*12,dtype=np.float32),
									   dtype = np.float32)

	def _crop_resize(self, img, width, height):
			img = img[150:500, 280:800]
			img = cv2.resize(img, (width, height))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			return img
	

	def reset(self): 
		self.init_hand()
		time.sleep(.5)

		obs = {}
		obs['features'] = self.hand.get_robot_state()
		obs['pixels'] = self._crop_resize(self.hand.get_rgb_images()[0][0], self.width, self.height)
		obs['depth'] = self.hand.get_depth_images()
		return obs