import pickle as pkl
import numpy as np

demo_name = '_noon1_224'
demo_length = 25

demo_path = f'/path/FISH/expert_demos/robotgym/RobotFlipBagel-v1/expert_demos{demo_name}.pkl'
save_path = f'/path/FISH/expert_demos/robotgym/RobotFlipBagel-v1/expert_demos{demo_name}_{demo_length}.pkl'

# Load data
with open(demo_path, 'rb') as f:
    observations, states, actions, rewards = pkl.load(f)

# Modify length
for i in range(len(observations)):
    if observations[i].shape[0] > demo_length:
        observations[i] = observations[i][:demo_length]
        states[i] = states[i][:demo_length]
        actions[i] = actions[i][:demo_length]
        rewards[i] = rewards[i][:demo_length]
    elif observations[i].shape[0] < demo_length:
        # Pad with last frame
        pad_obs = np.tile(observations[i][-1], (demo_length - observations[i].shape[0], 1, 1, 1))
        pad_states = np.tile(states[i][-1], (demo_length - states[i].shape[0], 1))
        pad_actions = np.tile(actions[i][-1], (demo_length - actions[i].shape[0], 1))
        pad_rewards = np.tile(rewards[i][-1], (demo_length - rewards[i].shape[0]))

        # Modify length
        observations[i] = np.concatenate((observations[i], pad_obs), axis=0)
        states[i] = np.concatenate((states[i], pad_states), axis=0)
        actions[i] = np.concatenate((actions[i], pad_actions), axis=0)
        rewards[i] = np.concatenate((rewards[i], pad_rewards), axis=0)

# Save data
with open(save_path, 'wb') as f:
    pkl.dump([observations, states, actions, rewards], f)