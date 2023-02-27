from tensor_subscriber import TensorSubscriber
from hello_robot import HelloRobot
import rospy
from std_msgs.msg import Int64
import random
import pickle
import argparse

PING_TOPIC_NAME = 'run_model_ping'

parser = argparse.ArgumentParser()
parser.add_argument('--height', type=float, default=0.5, help='position of robot lift')
args = parser.parse_args()
params = vars(args)

def act():

    rospy.init_node('Acting_node')
    


    publisher = rospy.Publisher(PING_TOPIC_NAME, Int64, queue_size=1)
    
    tensor_sub_object = TensorSubscriber()
    rate = rospy.Rate(5)
    hello_robot = HelloRobot(lift_pos = params['height'])    

    while(True):
        # x = input()
        uid = random.randint(0,30000)
        publisher.publish(Int64(uid))
        print('published', uid)
        waiting = True
        while (waiting):
            
                
            if ((tensor_sub_object.tr_data_offset == uid) and (tensor_sub_object.rot_data_offset == uid) and (tensor_sub_object.gr_data_offset == uid)) or (tensor_sub_object.home_data_offset==uid):
                waiting = False
            rate.sleep()
        
        if tensor_sub_object.home_data_offset==uid:
            hello_robot.home(params['height'])
            rate.sleep()
            continue

        hello_robot.move_to_pose(tensor_sub_object.translation_tensor, tensor_sub_object.rotational_tensor, tensor_sub_object.gripper_tensor)
        rate.sleep()
        

if __name__ == '__main__':
    act()

