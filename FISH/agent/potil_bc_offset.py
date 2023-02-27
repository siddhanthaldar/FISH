import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.encoder import Encoder
from rewarder import optimal_transport_plan, cosine_distance, euclidean_distance
import time
import copy

class DummyClass():
	def __init__(self):
		pass
	def reset(self):
		pass

class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, offset_mask, device):
		super().__init__()

		self.policy = nn.Sequential(nn.Linear(feature_dim + action_shape[0]*100, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, action_shape[0]))
		self.offset_mask = torch.tensor(offset_mask).float().to(device)


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
			nn.Linear(feature_dim + action_shape[0]*100, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.Q2 = nn.Sequential(
			nn.Linear(feature_dim + action_shape[0]*100, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.apply(utils.weight_init)

	def forward(self, obs, bc_action, offset_action):
		bc_action = bc_action.repeat(1, 50)
		offset_action = offset_action.repeat(1, 50)
		h_action = torch.cat([obs, bc_action, offset_action], dim=-1)		
		q1 = self.Q1(h_action)
		q2 = self.Q2(h_action)

		return q1, q2


class POTILAgent:
	def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
				 hidden_dim, critic_target_tau, num_expl_steps,
				 update_every_steps, stddev_schedule, stddev_clip, use_tb, augment,
				 rewards, sinkhorn_rew_scale, update_target_every,
				 auto_rew_scale, auto_rew_scale_factor, suite_name, obs_type,
				 bc_weight_type, bc_weight_schedule, offset_scale_factor, offset_mask):
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
		self.bc_weight_type = bc_weight_type
		self.bc_weight_schedule = bc_weight_schedule
		self.offset_scale_factor = offset_scale_factor

		# models
		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			self.encoder_target = Encoder(obs_shape).to(device)
			# Set to eval mode
			self.encoder.eval()
			self.encoder_target.eval()

			# Turn off gradients for encoder
			for param in self.encoder.parameters():
				param.requires_grad = False
			for param in self.encoder_target.parameters():
				param.requires_grad = False
			repr_dim = self.encoder.repr_dim
		else:
			repr_dim = obs_shape[0]

		self.actor = Actor(repr_dim, action_shape, feature_dim,
						   hidden_dim, offset_mask, self.device).to(device)

		self.critic = Critic(repr_dim, action_shape, feature_dim,
							 hidden_dim).to(device)
		self.critic_target = Critic(repr_dim, action_shape,
									feature_dim, hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		# optimizers
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=4)

		self.train()
		self.critic_target.train()

		# Compatability - not used in this agent
		self.buffer = DummyClass()

	def __repr__(self):
		return "potil"

	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.critic.train(training)

	def act(self, obs, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device)
		stddev = utils.schedule(self.stddev_schedule, step)

		with torch.no_grad():
			obs_bc = self.encoder_bc(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)
			action_bc = self.actor_bc(obs_bc, stddev).mean

		# offset computation
		obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)
		dist = self.actor(obs, action_bc, stddev)

		if eval_mode:
			offset = dist.mean
		else:
			offset = dist.sample(clip=None)
			if step < self.num_expl_steps:
				offset.uniform_(-1.0, 1.0)
		
		action = action_bc + offset * self.offset_scale_factor
		
		return action.cpu().numpy()[0], action_bc.cpu().numpy()[0]


	def update_critic(self, obs, action_base, offset_action, next_action_base, reward, discount, next_obs, step):
		metrics = dict()

		with torch.no_grad():
			stddev = utils.schedule(self.stddev_schedule, step)
			dist = self.actor(next_obs, next_action_base, stddev)
			next_offset_action = dist.sample(clip=self.stddev_clip)
			target_Q1, target_Q2 = self.critic_target(next_obs, next_action_base, next_offset_action)
			target_V = torch.min(target_Q1, target_Q2)
			target_Q = reward + (discount * target_V)

		Q1, Q2 = self.critic(obs, action_base, offset_action)

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

	def update_actor(self, obs, obs_bc, obs_qfilter, action_bc, action_base, action_bc_base, bc_regularize, step):
		metrics = dict()

		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs, action_base, stddev)
		action_offset = dist.sample(clip=self.stddev_clip)
		log_prob = dist.log_prob(action_offset).sum(-1, keepdim=True)

		Q1, Q2 = self.critic(obs, action_base, action_offset)
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
				dist_qf = self.actor_bc(obs_qfilter, stddev)
				action_qf = dist_qf.mean
				Q1_qf, Q2_qf = self.critic(obs_qfilter.clone(), action_qf)
				Q_qf = torch.min(Q1_qf, Q2_qf)
				bc_weight = (Q_qf>Q).float().mean().detach()

		actor_loss = - Q.mean() * (1-bc_weight)

		if bc_regularize:
			stddev = 0.1
			dist_bc = self.actor(obs_bc, stddev)
			log_prob_bc = dist_bc.log_prob(action_bc - action_bc_base).sum(-1, keepdim=True)
			actor_loss += - log_prob_bc.mean()*bc_weight*0.03

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
				metrics['regularized_bc_loss'] = - log_prob_bc.mean().item()*bc_weight*0.03
				metrics['bc_loss'] = - log_prob_bc.mean().item()*0.03
		
		return metrics

	def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False):
		metrics = dict()

		if step % self.update_every_steps != 0:
			return metrics

		batch = next(replay_iter)
		obs, action, action_base, reward, discount, next_obs, next_action_base = utils.to_torch(
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
			obs = self.encoder(obs)
			with torch.no_grad():
				next_obs = self.encoder(next_obs)

		# if bc_regularize:
		batch = next(expert_replay_iter)
		obs_bc, action_bc = utils.to_torch(batch, self.device)
		# base action for regularized bc
		with torch.no_grad():
			stddev = utils.schedule(self.stddev_schedule, step)
			obs_bc_base = self.encoder_bc(obs_bc) if self.use_encoder else obs_bc
			action_bc_base = self.actor_bc(obs_bc_base, stddev).mean
			
		# augment
		if self.use_encoder and self.augment:
			obs_bc = self.aug(obs_bc.float())
		else:
			obs_bc = obs_bc.float()
		# encode
		if bc_regularize and self.bc_weight_type=="qfilter":
			obs_qfilter = self.encoder_bc(obs_qfilter) if self.use_encoder else obs_qfilter
			obs_qfilter = obs_qfilter.detach()
		else:
			obs_qfilter = None
		obs_bc = self.encoder(obs_bc) if self.use_encoder else obs_bc 
		# Detach grads
		obs_bc = obs_bc.detach()

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(obs, action_base.clone(), (action-action_base).clone(), next_action_base, reward, discount, next_obs, step))

		# update actor
		metrics.update(self.update_actor(obs.detach(), obs_bc, obs_qfilter, action_bc, action_base, action_bc_base, bc_regularize, step))

		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics

	def ot_rewarder(self, observations, demos, step):

		scores_list = list()
		ot_rewards_list = list()
		for demo in demos:
			obs = torch.tensor(observations).to(self.device).float()
			obs = self.encoder_target(obs) if self.use_encoder else self.trunk_target(obs)
			exp = torch.tensor(demo).to(self.device).float()
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
			if k=="encoder":
				self.__dict__[k] = v
			elif k=="actor":
				self.actor_bc = v
		self.critic_target.load_state_dict(self.critic.state_dict())
		if self.use_encoder:
			self.encoder_target.load_state_dict(self.encoder.state_dict())

		# Store a copy of the BC policy with frozen weights
		if self.use_encoder:
			self.encoder_bc = copy.deepcopy(self.encoder)
			for param in self.encoder_bc.parameters():
				param.requires_grad = False
		for param in self.actor_bc.parameters():
			param.required_grad = False

		# Update optimizers
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

	def load_snapshot_eval(self, payload, bc=False):
		if bc:
			self.encoder_bc = payload['encoder']
			self.actor_bc = payload['actor']
			for param in self.encoder_bc.parameters():
				param.requires_grad = False
			for param in self.actor_bc.parameters():
				param.required_grad = False
		else:
			for k, v in payload.items():
				self.__dict__[k] = v
			self.critic_target.load_state_dict(self.critic.state_dict())
			if self.use_encoder:
				self.encoder_target.load_state_dict(self.encoder.state_dict())