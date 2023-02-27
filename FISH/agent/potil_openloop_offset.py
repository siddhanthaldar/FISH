"""
This takes potil_vinn_offset and compute q-filter on encoder_vinn and vinn_action_qfilter
"""
import os
import hydra
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms as T
import torch.nn.functional as F

import utils
from agent.encoder import Encoder
from rewarder import optimal_transport_plan, cosine_distance, euclidean_distance

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


class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, offset_mask):
		super().__init__()


		self.policy = nn.Sequential(nn.Linear(feature_dim + action_shape[0]*100, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, action_shape[0]))
		self.offset_mask = torch.tensor(offset_mask).float().to(torch.device('cuda'))

		self.apply(utils.weight_init)

	def forward(self, obs, action, std):

		action = action.repeat(1, 100)
		h = torch.cat((obs, action), dim=1)
		mu = self.policy(h) 
		mu = torch.tanh(mu) * self.offset_mask

		std = torch.ones_like(mu) * std * self.offset_mask

		dist = utils.TruncatedNormal(mu, std)
		return dist


class Critic(nn.Module):
	def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
		super().__init__()

		self.Q1 = nn.Sequential(
			nn.Linear(feature_dim + action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.Q2 = nn.Sequential(
			nn.Linear(feature_dim + action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.apply(utils.weight_init)

	def forward(self, obs, action):
		h_action = torch.cat([obs, action], dim=-1)
		q1 = self.Q1(h_action)
		q2 = self.Q2(h_action)
		return q1, q2


class POTILVINNAgent:
	def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
				 hidden_dim, critic_target_tau, num_expl_steps,
				 update_every_steps, stddev_schedule, stddev_clip, use_tb, augment,
				 rewards, sinkhorn_rew_scale, update_target_every,
				 auto_rew_scale, auto_rew_scale_factor,suite_name, obs_type, encoder_type, 
				 bc_weight_type, bc_weight_schedule, pretrained, offset_scale_factor, offset_mask):
		self.device = device
		self.lr = lr
		self.critic_target_tau = critic_target_tau
		self.update_every_steps = update_every_steps
		self.use_tb = use_tb
		self.num_expl_steps = num_expl_steps
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.augment = augment
		self.rewards = rewards
		self.sinkhorn_rew_scale = sinkhorn_rew_scale
		self.update_target_every = update_target_every
		self.auto_rew_scale = auto_rew_scale
		self.auto_rew_scale_factor = auto_rew_scale_factor
		self.use_encoder = True if obs_type=='pixels' else False
		self.encoder_type = encoder_type
		self.bc_weight_type = bc_weight_type
		self.bc_weight_schedule = bc_weight_schedule
		self.offset_scale_factor = offset_scale_factor
		
		self.MAX_BUFFER_SIZE = 5
		self.TOLERANCE = 1
		self.k_buffer = self.MAX_BUFFER_SIZE + 1
		self.k = 1
		self.buffer = HigherThresholdNearestNeighborBuffer(self.MAX_BUFFER_SIZE, self.TOLERANCE)

		# For storing demos
		self.observations = None
		self.actions = None

		# models
		if self.use_encoder:
			if self.encoder_type == 'small':
				#small encoder
				self.encoder = Encoder(obs_shape).to(device)
				self.encoder_target = Encoder(obs_shape).to(device)
				self.encoder_target.load_state_dict(self.encoder.state_dict())
				repr_dim = self.encoder.repr_dim
			elif self.encoder_type == 'resnet18':
				# resnet18
				from torchvision import models
				self.encoder = models.resnet18(pretrained=True).to(device)
				self.encoder.fc = Identity()
				self.encoder_target = models.resnet18(pretrained=True).to(device)
				self.encoder_target.fc = Identity()
				repr_dim = 512
			elif self.encoder_type == 'r3m':
				# r3m
				from r3m import load_r3m
				self.encoder = load_r3m("resnet18").to(device)
				self.encoder_target = load_r3m("resnet18").to(device)
				repr_dim = 512
			elif self.encoder_type == 'mvp':
				# mvp
				import mvp
				self.encoder = mvp.load("vits-mae-hoi").to(device)
				self.encoder_target = mvp.load("vits-mae-hoi").to(device)
				repr_dim = 384

			self.encoder.eval()
			self.encoder_target.eval()

			for param in self.encoder.parameters():
				param.requires_grad = False
			for param in self.encoder_target.parameters():
				param.requires_grad = False
			
		else:
			repr_dim = obs_shape[0]

		# vinn encoder
		if self.use_encoder:
			if self.encoder_type == 'small':
				# small encoder
				self.encoder_vinn = Encoder(obs_shape).to(device)
			elif self.encoder_type == 'resnet18':
				# resnet18
				self.encoder_vinn = models.resnet18(pretrained=pretrained).to(self.device)
				self.encoder_vinn.fc = Identity()
			elif self.encoder_type == 'r3m':
				# R3M Encoder
				self.encoder_vinn = load_r3m("resnet18").to(device)
			elif self.encoder_type == 'mvp':			
				# mvp
				self.encoder_vinn = mvp.load("vits-mae-hoi").to(device)
			
			self.encoder_vinn.eval()
			for param in self.encoder_vinn.parameters():
				param.requires_grad = False

		self.actor = Actor(repr_dim, action_shape, feature_dim,
						   hidden_dim, offset_mask).to(device)

		self.critic = Critic(repr_dim, action_shape, feature_dim,
							 hidden_dim).to(device)
		self.critic_target = Critic(repr_dim, action_shape,
									feature_dim, hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

			
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=4)

		# normalize visual observations
		if self.use_encoder:
			if pretrained:
				MEAN = torch.tensor([0.485, 0.456, 0.406])
				STD = torch.tensor([0.229, 0.224, 0.225])
				self.normalize = T.Compose([T.Normalize(mean=MEAN, std=STD)])
			else:
				self.normalize = None

		self.train()
		self.critic_target.train()

		# current step in episode
		self.count = 0
		# openloop tracker
		self.curr_step = 0

		# make directory called matches
		if not os.path.exists('matches'):
			os.makedirs('matches')


	def __repr__(self):
		return "potil_vinn_offset"

	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.critic.train(training)

	def save_representations(self, observations, actions, batch_size):
		self.representations = []
		self.observations = torch.as_tensor(observations, device=self.device).float()
		self.actions = torch.as_tensor(actions, device=self.device).float()
		for idx in range(0, observations.shape[0]-1, batch_size):
			obs = self.observations[idx:idx+batch_size]
			if self.encoder_type != 'r3m':
				obs = self.normalize(obs/255.0) if self.normalize else obs
			self.representations.append(self.encoder_vinn(obs).detach() if self.use_encoder else obs.detach())
		self.representations = torch.cat(self.representations, dim=0)


	def vinn_act(self, obs):
		image = obs.clone()
		vinn_obs = torch.as_tensor(obs, device=self.device).float()
		if self.encoder_type != 'r3m':
			vinn_obs = self.normalize(vinn_obs/255.0) if self.normalize else vinn_obs
		vinn_obs = self.encoder_vinn(vinn_obs) if self.use_encoder else obs

		# KNN computation
		vinn_obs = vinn_obs.unsqueeze(1)
		reps = self.representations.unsqueeze(0)
		dist = torch.norm(vinn_obs - reps, dim=-1)
		topk = torch.topk(dist, k=self.k_buffer, dim=-1, largest=False)
		weights = F.softmax(-topk.values, dim=-1)
		
		# compute action
		topk_actions = self.actions[topk.indices]

		if obs.shape[0] == 1:
			if self.count == 0:
				self.curr_step = topk.indices[0][0].item()
				self.curr_step = 0  
			action = torch.Tensor(self.actions[min(self.curr_step, self.actions.shape[0]-1)])
			self.count += 1
			self.curr_step += 1
			return action.unsqueeze(0)
		else:
			action = (weights.unsqueeze(-1) * topk_actions).sum(dim=1)
			return action 

	def act(self, obs, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device).float()		
		if obs.ndim == 3:
			obs = obs.unsqueeze(0)

		with torch.no_grad():
			vinn_action = self.vinn_act(obs.clone())

		if self.encoder_type != 'r3m':
			obs = self.normalize(obs/255.0) if self.normalize else obs
		obs = self.encoder(obs) if self.use_encoder else obs

		stddev = utils.schedule(self.stddev_schedule, step)
		dist = self.actor(obs, vinn_action, stddev)
		if eval_mode:
			offset_action = dist.mean
		else:
			offset_action = dist.sample(clip=None)
			if step < self.num_expl_steps:
				offset_action.uniform_(-1.0, 1.0)

		action = vinn_action + offset_action * self.offset_scale_factor

		return action.cpu().numpy()[0], vinn_action.cpu().numpy()[0]

	def update_critic(self, obs, action, vinn_next_action, reward, discount, next_obs, step):
		metrics = dict()

		with torch.no_grad():
			stddev = utils.schedule(self.stddev_schedule, step)
			dist = self.actor(next_obs, vinn_next_action, stddev)

			next_action = vinn_next_action + dist.sample(clip=self.stddev_clip) * self.offset_scale_factor

			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_V = torch.min(target_Q1, target_Q2)
			target_Q = reward + (discount * target_V)

		Q1, Q2 = self.critic(obs, action)

		critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

		# optimize encoder and critic
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()

		if self.use_tb:
			metrics['critic_target_q'] = target_Q.mean().item()
			metrics['critic_q1'] = Q1.mean().item()
			metrics['critic_q2'] = Q2.mean().item()
			metrics['critic_loss'] = critic_loss.item()
			
		return metrics

	def update_actor(self, obs, obs_expert, obs_qfilter, action_expert, vinn_action, vinn_action_expert, bc_regularize, step):
		metrics = dict()

		stddev = utils.schedule(self.stddev_schedule, step)

		# compute action offset
		dist = self.actor(obs, vinn_action, stddev)
		action_offset = dist.sample(clip=self.stddev_clip)
		log_prob = dist.log_prob(action_offset).sum(-1, keepdim=True)

		# compute action
		action = vinn_action + action_offset * self.offset_scale_factor
		Q1, Q2 = self.critic(obs, action)
		Q = torch.min(Q1, Q2)

		# Compute bc weight
		if not bc_regularize:
			bc_weight = 0.0
		elif self.bc_weight_type == "linear":
			bc_weight = utils.schedule(self.bc_weight_schedule, step)
		elif self.bc_weight_type == "qfilter":
			"""
			Soft Q-filtering inspired from 			
			Nair, Ashvin, et al. "Overcoming exploration in reinforcement 
			learning with demonstrations." 2018 IEEE international 
			conference on robotics and automation (ICRA). IEEE, 2018.
			"""
			with torch.no_grad():
				stddev = 0.1
				action_qf = vinn_action.clone()
				Q1_qf, Q2_qf = self.critic(obs_qfilter.clone(), action_qf)
				Q_qf = torch.min(Q1_qf, Q2_qf)
				bc_weight = (Q_qf>Q).float().mean().detach()

		actor_loss = - Q.mean() * (1-bc_weight)

		if bc_regularize:
			stddev = 0.1
			dist_expert = self.actor(obs_expert, vinn_action_expert, stddev)
			action_expert_offset = dist_expert.sample(clip=self.stddev_clip) * self.offset_scale_factor

			true_offset = torch.zeros(action_expert_offset.shape).to(self.device)
			log_prob_expert = dist_expert.log_prob(true_offset).sum(-1, keepdim=True)
			actor_loss += - log_prob_expert.mean()*bc_weight*0.03

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()
		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
			metrics['actor_q'] = Q.mean().item()
			if bc_regularize and self.bc_weight_type == "qfilter":
				metrics['actor_qf'] = Q_qf.mean().item()
			metrics['bc_weight'] = bc_weight
			metrics['regularized_rl_loss'] = -Q.mean().item()* (1-bc_weight)
			metrics['rl_loss'] = -Q.mean().item()
			if bc_regularize:
				metrics['regularized_bc_loss'] = - log_prob_expert.mean().item()*bc_weight*0.03
				metrics['bc_loss'] = - log_prob_expert.mean().item()*0.03
			
		return metrics

	def update(self, replay_iter, expert_replay_iter, ssl_replay_iter, step, bc_regularize=False):
		metrics = dict()

		if step % self.update_every_steps != 0:
			return metrics

		batch = next(replay_iter)
		obs, action, vinn_action, reward, discount, next_obs, vinn_next_action = utils.to_torch(
			batch, self.device)

		# augment
		if self.use_encoder and self.augment:
			obs_qfilter = self.aug(obs.clone().float())
			obs = self.aug(obs.float())
			next_obs = self.aug(next_obs.float())
		else:
			obs_qfilter = obs.clone().float()
			obs = obs.float()
			next_obs = next_obs.float()

		if self.use_encoder:
			# encode
			if self.encoder_type != 'r3m' and self.normalize:
				obs = self.normalize(obs/255.0)
				next_obs = self.normalize(next_obs/255.0)
			obs = self.encoder(obs)
			with torch.no_grad():
				next_obs = self.encoder(next_obs)

		if bc_regularize:
			batch = next(expert_replay_iter)
			obs_expert, action_expert = utils.to_torch(batch, self.device)
			action_expert = action_expert.float()
			if self.k == 1:
				vinn_action_expert = action_expert.clone()
			else:
				vinn_action_expert = self.vinn_act(obs_expert.clone())
			# augment
			if self.use_encoder and self.augment:
				obs_expert = self.aug(obs_expert.float())
			else:
				obs_expert = obs_expert.float()
			# encode
			if bc_regularize and self.bc_weight_type=="qfilter":
				if self.encoder_type != 'r3m':
					obs_qfilter = self.normalize(obs_qfilter/255.0) if self.normalize else obs_qfilter
				obs_qfilter = self.encoder_vinn(obs_qfilter) if self.use_encoder else obs_qfilter
				obs_qfilter = obs_qfilter.detach()
			else:
				obs_qfilter = None
				vinn_action_expert = None
			if self.encoder_type != 'r3m':
				obs_expert = self.normalize(obs_expert/255.0) if self.normalize else obs_expert
			obs_expert = self.encoder(obs_expert) if self.use_encoder else obs_expert 
			# Detach grads
			obs_expert = obs_expert.detach()
		else:
			obs_qfilter = None
			obs_expert = None 
			action_expert = None
			vinn_action_expert = None

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(obs, action, vinn_next_action, reward, discount, next_obs, step))

		# update actor
		metrics.update(self.update_actor(obs.detach(), obs_expert, obs_qfilter, action_expert, vinn_action, vinn_action_expert, bc_regularize, step))

		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics

	def ot_rewarder(self, observations, demos, step):
			
		scores_list = list()
		ot_rewards_list = list()
		
		obs = torch.tensor(observations).to(self.device).float()
		if self.encoder_type != 'r3m':
			obs = self.normalize(obs/255.0) if self.normalize else obs
		obs = self.encoder_target(obs) if self.use_encoder else self.trunk_target(obs)
		for demo in demos:
			exp = torch.tensor(demo).to(self.device).float()
			if self.encoder_type != 'r3m':
				exp = self.normalize(exp/255.0) if self.normalize else exp 
			exp = self.encoder_target(exp) if self.use_encoder else self.trunk_target(exp)
			obs = obs.detach()
			exp = exp.detach()
			
			if self.rewards == 'sinkhorn_cosine':
				cost_matrix = cosine_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn',
					niter=100).float()  # Getting optimal coupling
				ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()
				
			elif self.rewards == 'sinkhorn_euclidean':
				cost_matrix = euclidean_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn',
					niter=100).float()  # Getting optimal coupling
				ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()
				
			elif self.rewards == 'cosine':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(1. - F.cosine_similarity(obs, exp))
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()
				
			elif self.rewards == 'euclidean':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(obs - exp).norm(dim=1)
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()
				
			else:
				raise NotImplementedError()

			scores_list.append(np.sum(ot_rewards))
			ot_rewards_list.append(ot_rewards)

		closest_demo_index = np.argmax(scores_list)
		return ot_rewards_list[closest_demo_index]

	def save_snapshot(self):
		keys_to_save = ['actor', 'critic']
		if self.use_encoder:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload):
		for k, v in payload.items():
			if k == 'encoder':
				self.encoder_vinn = v
	
		self.critic_target.load_state_dict(self.critic.state_dict())
		if self.use_encoder:
			self.encoder.load_state_dict(self.encoder_vinn.state_dict())
			self.encoder_target.load_state_dict(self.encoder.state_dict())

		if self.use_encoder:
			self.encoder_vinn.eval()
			for param in self.encoder_vinn.parameters():
				param.requires_grad = False
	
			self.encoder.eval()
			for param in self.encoder.parameters():
				param.requires_grad = False
			self.encoder_target.eval()
			for param in self.encoder_target.parameters():
				param.requires_grad = False

		self.actor_opt = torch.optim.Adam(self.actor.policy.parameters(), lr=self.lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)


	def load_snapshot_eval(self, payload, bc=False):
		for k, v in payload.items():
			self.__dict__[k] = v
		self.critic_target.load_state_dict(self.critic.state_dict())
		if self.use_encoder:
			self.encoder_target.load_state_dict(self.encoder.state_dict())