import pickle as pkl
from pathlib import Path
import numpy as np

folder = Path("/path/fish/expert_demos/robotgym/RobotFlipBagel-v1")
demo_file = "expert_demos_noon1_224_25"

demo_path = folder / (demo_file + ".pkl")

# Read 
f = open(demo_path, 'rb')
obs, state, act, rew = pkl.load(f)

actions = np.concatenate(act, axis=0)
amax = np.max(actions, axis=0)
amin = np.min(actions, axis=0)

for i in range(len(act)):
    act[i] = (act[i] - amin) / (amax - amin)

# Write
f = open(folder / (demo_file+"_scaled.pkl"), 'wb')
pkl.dump([obs, state, act, rew], f)

print(f"Action max: {list(amax)}")
print(f"Action min: {list(amin)}")
