'''
gym like wrapper for Hello Robot
'''
import gym
from gym import spaces
import cv2 
import numpy as np
import time
import rospy
import numpy as np
import torch

from sensor_msgs.msg import Image as Image_msg
from std_msgs.msg import Float64MultiArray, Int64
from cv_bridge import CvBridge, CvBridgeError

from copy import deepcopy as copy
import signal
import sys

TRAJECTORY_DIRECTORY = '/data/anant/trajectory_images/'


IMAGE_SUBSCRIBER_TOPIC = '/gopro_image'
DEPTH_SUBSCRIBER_TOPIC = '/depth_image'

TRANSLATIONAL_PUBLISHER_TOPIC = '/translation_tensor'
ROTATIONAL_PUBLISHER_TOPIC = '/rotational_tensor'
GRIPPER_PUBLISHER_TOPIC = '/gripper_tensor'
HOME_PUBLISHER_TOPIC = '/home_tensor'

PING_TOPIC = 'run_model_ping'
import time

class HelloEnv(gym.Env):
        def __init__(self):
             # Initializing a rosnode        
            rospy.init_node('image_subscriber')

            # Getting images from the rostopic
            self.image = None
            self.bridge = CvBridge()

            # Subscriber for images
            rospy.Subscriber(IMAGE_SUBSCRIBER_TOPIC, Image_msg, self._callback_image, queue_size=1)
            rospy.Subscriber(PING_TOPIC, Int64, self._callback_ping, queue_size=1)
            

            self.uid = -1
            self.prev_uid = -1
            time.sleep(1)
            x = input("Start Robot Code")
            # Publishers for the evaluated tensors
            self.translational_publisher = rospy.Publisher(TRANSLATIONAL_PUBLISHER_TOPIC, Float64MultiArray, queue_size=1)
            self.rotational_publisher = rospy.Publisher(ROTATIONAL_PUBLISHER_TOPIC, Float64MultiArray, queue_size=1)
            self.gripper_publisher = rospy.Publisher(GRIPPER_PUBLISHER_TOPIC, Float64MultiArray, queue_size=1)
            self.home_publisher = rospy.Publisher(HOME_PUBLISHER_TOPIC, Float64MultiArray, queue_size=1)

            self.width = 84
            self.height = 84

        def _callback_image(self, data):
            try:
                self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)
        
        def _publish_home(self):
            home_publisher_list = Float64MultiArray()
            home_publisher_list.layout.data_offset = self.uid
            home_publisher_list.data = [1]
            self.home_publisher.publish(home_publisher_list)
        
        def _publish_instruction(self, instruction):

            translation_tensor = instruction[:3]
            rotation_tensor = [0, 0, 0]
            gripper_tensor = instruction[3:]
    
            translation_publisher_list = Float64MultiArray()
            translation_publisher_list.layout.data_offset = self.uid
            translation_publisher_list.data = translation_tensor

            rotation_publisher_list = Float64MultiArray()
            rotation_publisher_list.layout.data_offset = self.uid
            rotation_publisher_list.data = rotation_tensor

            gripper_publisher_list = Float64MultiArray()
            gripper_publisher_list.layout.data_offset = self.uid
            gripper_publisher_list.data = gripper_tensor

            self.translational_publisher.publish(translation_publisher_list)
            self.rotational_publisher.publish(rotation_publisher_list)
            self.gripper_publisher.publish(gripper_publisher_list)

        def _callback_ping(self, data):
            self.uid = int(data.data)
            print('Received uid {}'.format(self.uid))              

        def signal_handler(self, signal, frame):
            self.visualize_trajectory(5)
            print("Interrupted! Exiting...")
            sys.exit()

        def get_obs(self):
            return self._crop_resize(self.image)

        def _crop_resize(self, img):
            img = cv2.resize(img, (self.width, self.height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        
        def step(self, action):
            while self.prev_uid == self.uid:
                time.sleep(0.3)
            self.prev_uid = copy(self.uid)

            self._publish_instruction(action)
            obs = {}
            obs['feature'] = None
            obs['pixels'] = self.get_obs()

            return obs, 0, False, {'is_success': False} #obs, reward, done, infos

        def reset(self):
            self._publish_home()
            obs = {}
            obs['feature'] = None
            obs['pixels'] = self.get_obs()
            return obs

        def render(self, mode='rbg_array', width=0, height=0):
            return self.image
      