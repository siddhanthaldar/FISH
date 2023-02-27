import numpy as np
import pickle as pkl
from collections import deque
from pathlib import Path

demo_folder = Path("/path/FISH/expert_demos/robotgym/RobotInsertPeg-v1")

file = "expert_demos"
frame_stack = 3

# read the data
f = open(demo_folder / (file+".pkl"), "rb")
images, states, actions, rewards = pkl.load(f)

# stack the frames
demos = list()
for demo in images:
    stacked_demo = list()
    stacked_img = deque(maxlen=frame_stack)
    for frame in demo:
        stacked_img.append(frame)
        while len(stacked_img) < frame_stack:
            stacked_img.append(frame)
        stacked_demo.append(np.concatenate(stacked_img))
    demos.append(np.array(stacked_demo))
demos = np.array(demos)

# save the data
f = open(demo_folder / (file+"_stacked_"+str(frame_stack)+".pkl"), "wb")
pkl.dump([demos, states, actions, rewards], f)
