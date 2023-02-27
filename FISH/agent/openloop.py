import torch
import torch.nn as nn
from torchvision import transforms as T
import torch.nn.functional as F

class HigherThresholdNearestNeighborBuffer(object):
	def __init__(self, buffer_size, tolerance = 2):
		self.buffer_size = buffer_size
		self.tolerance = tolerance
		self.exempted_queue = []
		self.seen_queue = {}

	def reset(self):
		self.exempted_queue = []
		self.seen_queue = {}

	def put(self, item):
		self.exempted_queue.append(item)
		if len(self.exempted_queue) > self.buffer_size:
			self.exempted_queue.pop(0)

	def get(self):
		item = self.exempted_queue[0]
		self.exempted_queue.pop(0)
		return item

	def choose(self, nn_idxs):
		for idx in range(len(nn_idxs)):
			item = nn_idxs[idx].item()
			if item not in self.exempted_queue:
				if item in self.seen_queue.keys():
					self.seen_queue[item] += 1

					if self.seen_queue[item] > self.tolerance:
						self.seen_queue[item] = 0
						self.put(nn_idxs[idx].item())
						continue
				else:
					self.seen_queue[item] = 1

				return idx
				
		return len(nn_idxs) - 1

class Identity(nn.Module):
    '''
    Author: Janne Spijkervet
    url: https://github.com/Spijkervet/SimCLR
    '''
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Openloop:
	def __init__(self, obs_shape):

		self.MAX_BUFFER_SIZE = 5
		self.TOLERANCE = 1
		self.buffer = HigherThresholdNearestNeighborBuffer(self.MAX_BUFFER_SIZE, self.TOLERANCE)

		# Track openloop time step
		self.curr_step = 0

		self.train()

	def __repr__(self):
		return "openloop"

	def train(self, training=True):
		self.training = training
		
	def act(self, obs, step, eval_mode):
		action = self.actions[self.curr_step]
		self.curr_step += 1
		return action.squeeze(0).numpy()


	def save_representations(self, observations, actions, batch_size, eval_mode=False):
		self.actions = torch.as_tensor(actions).float()
