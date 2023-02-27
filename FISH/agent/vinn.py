import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
import torch.nn.functional as F

import utils
from byol_pytorch import BYOL
from agent.encoder import Encoder

class Identity(nn.Module):
	'''
	Author: Janne Spijkervet
	url: https://github.com/Spijkervet/SimCLR
	'''
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x

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

class VINN:
	def __init__(self, 
				 obs_shape,
				 device,
				 lr,
				 feature_dim,
				 use_tb,
				 augment,
				 obs_type,
				 encoder_type,
				 pretrained,
				 image_size=256):
		self.obs_shape = obs_shape
		self.device = device
		self.lr = lr
		self.use_tb = use_tb
		self.augment = augment
		self.obs_type = obs_type
		self.encoder_type = encoder_type
		self.image_size = image_size
		self.use_encoder = True if obs_type=='pixels' else False
		self.count = 0

		self.MAX_BUFFER_SIZE = 3
		self.TOLERANCE = 1
		self.k_buffer = self.MAX_BUFFER_SIZE + 1
		self.k = 1
		self.buffer = HigherThresholdNearestNeighborBuffer(self.MAX_BUFFER_SIZE, self.TOLERANCE)

		# augmentations
		if self.use_encoder:
			if self.encoder_type == 'small':
				# small encoder
				self.encoder = Encoder(obs_shape).to(device)
				repr_dim = self.encoder.repr_dim
				pretrained = False
			elif self.encoder_type == 'resnet18':
				# ResNet18 Encoder
				from torchvision import models
				self.encoder = models.resnet18(pretrained=pretrained)
				self.encoder.fc = Identity()
				self.encoder = self.encoder.to(device)
				repr_dim = 512

			if pretrained:
				MEAN = torch.tensor([0.485, 0.456, 0.406])
				STD = torch.tensor([0.229, 0.224, 0.225])
			else:
				MEAN = torch.tensor([0.0, 0.0, 0.0])
				STD = torch.tensor([1.0, 1.0, 1.0])
			self.customAug = T.Compose([
								T.RandomResizedCrop(image_size, scale=(0.6,1.0)),
								T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), p=.3),
								T.RandomGrayscale(p=0.2),
								T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
								T.Normalize(
									mean=MEAN,
									std=STD)])
			self.customAugEval = T.Compose([
								T.Normalize(
									mean=MEAN,
									std=STD)])							


			# BYOL Learner
			self.learner = BYOL(
				self.encoder,
				image_size = self.image_size,
				hidden_layer = "avgpool" if self.encoder_type == 'resnet18' else -1,
				augment_fn = self.customAug
			)
		# optimizers
		if self.use_encoder:
			self.learner_opt = torch.optim.Adam(self.learner.parameters(), lr=self.lr)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=4)

		self.train()

	def __repr__(self):
		return "vinn"

	def train(self, training=True):
		self.training = training
		if self.use_encoder:
			self.encoder.train(training)

	def act(self, obs, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device).float()[-3:]
		
		obs = obs.unsqueeze(0)

		if self.encoder_type == 'small':
			obs = self.encoder(obs) if self.use_encoder else obs
		elif self.encoder_type == 'resnet18':
			obs = self.encoder(self.customAugEval(obs / 255.0)) if self.use_encoder else obs
		
		# distance computation
		obs = obs.unsqueeze(1)
		reps = self.representations.unsqueeze(0)
		dist = torch.norm(obs - reps, dim=-1)
		topk = torch.topk(dist, k=self.k, dim=-1, largest=False)		
		weights = F.softmax(-topk.values, dim=-1)
		
		# compute action
		topk_actions = self.actions[topk.indices]
		top_idx = self.buffer.choose(topk.indices[0])
		action = topk_actions[0, top_idx]
		
		return action.squeeze(0).cpu().numpy(), action.squeeze(0).cpu().numpy()


	def save_representations(self, observations, actions, batch_size, eval_mode=False):
		self.representations = []
		observations = observations[:, -3:]
		self.observations = torch.as_tensor(observations, device=self.device).float()
		self.actions = torch.as_tensor(actions, device=self.device).float()
		for idx in range(0, observations.shape[0], batch_size):
			obs = torch.as_tensor(observations[idx:idx+batch_size], device=self.device).float()
			if self.encoder_type == 'small':
				self.representations.append(self.encoder(obs).detach() if self.use_encoder else obs.detach())
			elif self.encoder_type == 'resnet18':
				self.representations.append(self.encoder(self.customAugEval(obs/255.0)).detach() if self.use_encoder else obs.detach())
		self.representations = torch.cat(self.representations, dim=0)
		
	def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False):
		metrics = dict()

		batch = next(expert_replay_iter)
		obs, _ = utils.to_torch(batch, self.device)
		obs = obs[:, -3:]

		if self.use_encoder:
			obs = obs / 255.0 if self.encoder_type == 'resnet18' else obs
			
			# loss 
			loss = self.learner(obs.float())

			# optimize
			self.learner_opt.zero_grad(set_to_none=True)
			loss.backward()
			self.learner_opt.step()
			self.learner.update_moving_average()

		if self.use_tb:
			metrics['loss'] = loss.item()

		return metrics


	def save_snapshot(self):
		if self.use_encoder:
			keys_to_save = ['encoder']
			payload = {k: self.__dict__[k] for k in keys_to_save}
		else:
			payload = {}
		return payload

	def load_snapshot(self, payload):
		for k, v in payload.items():
			if k == 'encoder':
				self.__dict__[k] = v

		self.encoder = self.encoder.to(self.device)
		
		# BYOL Learner
		self.learner = BYOL(
			self.encoder,
			image_size = self.image_size,
			hidden_layer = 'avgpool',
			augment_fn = self.customAug
		)

		# optimizers
		self.learner_opt = torch.optim.Adam(self.learner.parameters(), lr=self.lr)

	def load_snapshot_eval(self, payload, bc=False):
		for k, v in payload.items():
			if k == 'encoder':
				self.__dict__[k] = v

		self.encoder = self.encoder.to(self.device)