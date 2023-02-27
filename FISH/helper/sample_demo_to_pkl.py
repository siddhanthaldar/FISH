import pickle as pkl

demo_name = 'expert_demos_stripes_train_stacked_3'
demo_path = f'/path/FISH/expert_demos/robotgym/RobotPegInHole-v1/{demo_name}.pkl'
demo_number = 1
save_path = f'/path/FISH/expert_demos/robotgym/RobotPegInHole-v1/{demo_name}_1demo.pkl'

with open(demo_path, 'rb') as f:
    observations, states, actions, rewards = pkl.load(f)

observations = observations[demo_number:demo_number+1]
states = states[demo_number:demo_number+1]
actions = actions[demo_number:demo_number+1]
rewards = rewards[demo_number:demo_number+1]

with open(save_path, 'wb') as f:
    pkl.dump([observations, states, actions, rewards], f)