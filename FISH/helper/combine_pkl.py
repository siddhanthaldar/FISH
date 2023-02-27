import os
import numpy as np
import pickle as pkl

folder = '/path/FISH/expert_demos/robotgym/RobotPegInHole-v1/'
train_files = ['expert_demos.pkl']
test_files = []
save_name = 'stripes'
traj_len = 25

################################### Save as trajectories ###################################
if len(train_files) > 0:
    # train
    observations = []
    states = []
    actions = []
    rewards = []
    for file in train_files:
        with open(folder + file, 'rb') as f:
            data = pkl.load(f)
            for i in range(data[0].shape[0]):
                if data[0][i].shape[0] < traj_len:
                    observation_buffer = np.array([data[0][i][-1] for _ in range(traj_len - data[0][i].shape[0])])
                    state_buffer = np.array([data[1][i][-1] for _ in range(traj_len - data[1][i].shape[0])])
                    action_buffer = np.array([data[2][i][-1] for _ in range(traj_len - data[2][i].shape[0])])
                    reward_buffer = np.array([data[3][i][-1] for _ in range(traj_len - data[3][i].shape[0])])

                    #concatenate
                    observations.append(np.concatenate((data[0][i], observation_buffer), axis=0))
                    states.append(np.concatenate((data[1][i], state_buffer), axis=0))
                    actions.append(np.concatenate((data[2][i], action_buffer), axis=0))
                    rewards.append(np.concatenate((data[3][i], reward_buffer), axis=0))
                elif data[0][i].shape[0] > traj_len:
                    observations.append(data[0][i][:traj_len])
                    states.append(data[1][i][:traj_len])
                    actions.append(data[2][i][:traj_len])
                    rewards.append(data[3][i][:traj_len])
                else:    
                    observations.append(data[0][i])
                    states.append(data[1][i])
                    actions.append(data[2][i])
                    rewards.append(data[3][i])

    # save the data
    with open(folder + f'expert_demos_{save_name}_train.pkl', 'wb') as f:
        pkl.dump([observations, states, actions, rewards], f)

# test
if len(test_files) > 0:
    observations = []
    states = []
    actions = []
    rewards = []

    for file in test_files:
        with open(folder + file, 'rb') as f:
            data = pkl.load(f)
            for i in range(data[0].shape[0]):
                if data[0][i].shape[0] < traj_len:
                    observation_buffer = np.array([data[0][i][-1] for _ in range(traj_len - data[0][i].shape[0])])
                    state_buffer = np.array([data[1][i][-1] for _ in range(traj_len - data[1][i].shape[0])])
                    action_buffer = np.array([data[2][i][-1] for _ in range(traj_len - data[2][i].shape[0])])
                    reward_buffer = np.array([data[3][i][-1] for _ in range(traj_len - data[3][i].shape[0])])

                    #concatenate
                    observations.append(np.concatenate((data[0][i], observation_buffer), axis=0))
                    states.append(np.concatenate((data[1][i], state_buffer), axis=0))
                    actions.append(np.concatenate((data[2][i], action_buffer), axis=0))
                    rewards.append(np.concatenate((data[3][i], reward_buffer), axis=0))
                else:    
                    observations.append(data[0][i])
                    states.append(data[1][i])
                    actions.append(data[2][i])
                    rewards.append(data[3][i])

    # save the data
    with open(folder + f'expert_demos_{save_name}_test.pkl', 'wb') as f:
        pkl.dump([observations, states, actions, rewards], f)