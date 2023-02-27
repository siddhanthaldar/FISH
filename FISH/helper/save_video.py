import sys
sys.path.append('../')

import cv2
import numpy as np
from pathlib import Path
import pickle as pkl

from video import TrainVideoRecorder

name = "_night3_25_scaled"
DEMO_PATH = Path(f"/path/FISH/expert_demos/robotgym/RobotFlipBagel-v1/expert_demos{name}.pkl")
SAVE_DIR = Path("/path/FISH/expert_demos/robotgym/RobotFlipBagel-v1/")

with open(DEMO_PATH, "rb") as f:
    demos, _, _, _ = pkl.load(f)

recorder = TrainVideoRecorder(SAVE_DIR)

recorder.init(demos[0][0])
for demo in demos:
    for frame in demo:
        recorder.record(frame)
recorder.save(f'demo{name}.mp4')
