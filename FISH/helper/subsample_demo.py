import pickle as pkl
import numpy as np
from pathlib import Path

expert_dataset_folder = Path("/path/FISH/expert_demos/robotgym/RobotPegInHole-v1")
filename = "expert_demos_combined_train"
num_demos = 10

expert_dataset_path = expert_dataset_folder / (filename+".pkl")
images, states, actions, rewards = pkl.load(open(expert_dataset_path, "rb"))
indices = np.random.choice(len(images), num_demos, replace=False)
images = [images[i] for i in indices]
states = [states[i] for i in indices]
actions = [actions[i] for i in indices]
rewards = [rewards[i] for i in indices]

# save the expert dataset
expert_dataset_path = expert_dataset_folder / (filename+f"_subsampled_{num_demos}.pkl")
pkl.dump([images, states, actions, rewards], open(expert_dataset_path, "wb"))