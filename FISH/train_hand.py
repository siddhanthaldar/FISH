#!/usr/bin/env python3

from shutil import ExecError
# from tkinter import E
import warnings
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import utils
from logger import Logger
from replay_buffer_hand import ReplayBufferStorage, make_replay_loader, make_expert_replay_loader, make_ssl_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
	cfg.agent.obs_shape = obs_spec[cfg.obs_type].shape
	try:
		cfg.agent.action_shape = action_spec.shape
	except:
		pass
	try:
		cfg.agent.env_name = cfg.suite.task_name
	except:
		pass
	return hydra.utils.instantiate(cfg.agent)

class WorkspaceIL:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.setup()

		self.agent = make_agent(self.train_env.observation_spec(),
								self.train_env.action_spec(), cfg)
		# create logger
		self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

		if repr(self.agent) == 'drqv2':
			self.cfg.suite.num_train_frames = self.cfg.num_train_frames_drq
		elif repr(self.agent) == 'bc':
			self.cfg.suite.num_train_frames = self.cfg.num_train_frames_bc
			self.cfg.suite.num_seed_frames = 0
		elif repr(self.agent) in ['vinn', 'tcc']:
			self.cfg.suite.num_train_frames = self.cfg.num_train_frames_ssl
			self.cfg.suite.num_seed_frames = 0

		self.expert_replay_loader = make_expert_replay_loader(
			self.cfg.expert_dataset, self.cfg.batch_size // 2, self.cfg.num_demos, self.cfg.obs_type)
		self.expert_replay_iter = iter(self.expert_replay_loader)

		self.ssl_iterable, self.ssl_replay_loader = make_ssl_replay_loader(
			self.cfg.expert_dataset, self.cfg.batch_size, self.cfg.num_demos, 
			self.cfg.obs_type, return_episode=True if repr(self.agent) == 'tcc' else False)
		self._ssl_replay_iter = iter(self.ssl_replay_loader)
			
		self.timer = utils.Timer()
		self._global_step = 0
		self._global_episode = 0

		with open(self.cfg.expert_dataset, 'rb') as f:
			if self.cfg.obs_type == 'pixels':
				self.expert_demo, _, _, self.expert_reward = pickle.load(f)
			elif self.cfg.obs_type == 'features':
				_, self.expert_demo, _, self.expert_reward = pickle.load(f)
		self.expert_demo = self.expert_demo[:self.cfg.num_demos]#[:,:,-3:]
		self.expert_reward = 0 #np.mean(self.expert_reward[:self.cfg.num_demos])
		
	def setup(self):
		# create envs
		self.train_env = hydra.utils.call(self.cfg.suite.task_make_fn)
		
		# create replay buffer
		data_specs = [
			self.train_env.observation_spec()[self.cfg.obs_type],
			self.train_env.action_spec(),
			specs.Array(self.train_env.action_spec().shape, self.train_env.action_spec().dtype, 'vinn_action'), 
			specs.Array((1, ), np.float32, 'reward'),
			specs.Array((1, ), np.float32, 'discount')
		]

		self.replay_storage = ReplayBufferStorage(data_specs,
												  self.work_dir / 'buffer')

		self.replay_loader = make_replay_loader(
			self.work_dir / 'buffer', self.cfg.replay_buffer_size,
			self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
			self.cfg.suite.save_snapshot, self.cfg.nstep, self.cfg.suite.discount)

		self._replay_iter = None
		self.expert_replay_iter = None

		self.video_recorder = VideoRecorder(
			self.work_dir if self.cfg.save_video else None)
		self.train_video_recorder = TrainVideoRecorder(
			self.work_dir if self.cfg.save_train_video else None)

	@property
	def global_step(self):
		return self._global_step

	@property
	def global_episode(self):
		return self._global_episode

	@property
	def global_frame(self):
		return self.global_step * self.cfg.suite.action_repeat

	@property
	def replay_iter(self):
		if self._replay_iter is None:
			self._replay_iter = iter(self.replay_loader)
		return self._replay_iter

	def eval(self):
		step, episode, total_reward = 0, 0, 0
		eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

		self.video_recorder.init(self.train_env, enabled=True)
		while eval_until_episode(episode):
			print(f"Eval Episode {episode}")
			time_step = self.train_env.reset()
			while not time_step.last():
				with torch.no_grad(), utils.eval_mode(self.agent):
					action, vinn_action = self.agent.act(time_step.observation[self.cfg.obs_type],
											self.global_step,
											eval_mode=True)
				time_step = self.train_env.step(action, vinn_action)
				self.video_recorder.record(self.train_env)
				total_reward += time_step.reward
				step += 1

			episode += 1
			x = input("Press Enter to continue... after reseting env")
			# reset buffer and openloop count
			self.agent.buffer.reset()
			self.agent.openloop_step_count = 0
			self.agent.count = 0

		self.video_recorder.save(f'{self.global_frame}.mp4')
		
		with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
			log('episode_reward', total_reward / episode)
			log('episode_length', step * self.cfg.suite.action_repeat / episode)
			log('episode', self.global_episode)
			log('step', self.global_step)
			if repr(self.agent) != 'drqv2':
				log('expert_reward', self.expert_reward)
		
		# Reset env
		self.train_env.reset()

	def train_il(self):
		# Set num_seed_frames to 0 for ['bc', 'tcc']
		if repr(self.agent) in ['bc', 'vinn', 'tcc']:
			self.cfg.suite.num_seed_frames = 0

		# predicates
		train_until_step = utils.Until(self.cfg.suite.num_train_frames,
									   self.cfg.suite.action_repeat)
		seed_until_step = utils.Until(self.cfg.suite.num_seed_frames,
									  self.cfg.suite.action_repeat)
		eval_every_step = utils.Every(self.cfg.suite.eval_every_frames,
									  self.cfg.suite.action_repeat)

		# Get obs reps from expert data
		if 'vinn' in repr(self.agent) or 'openloop' in repr(self.agent):
			expert_observations, expert_actions = self.ssl_iterable.get_data()
			self.agent.save_representations(expert_observations, expert_actions, batch_size=self.cfg.batch_size)

		episode_step, episode_reward = 0, 0

		if repr(self.agent) not in ['bc', 'vinn', 'tcc']:
			time_steps = list()
			observations = list()
			actions = list()

			time_step = self.train_env.reset()
			time_steps.append(time_step)
			observations.append(time_step.observation[self.cfg.obs_type])
			actions.append(time_step.action)
		
			if 'potil' in repr(self.agent):
				if self.agent.auto_rew_scale:
					self.agent.sinkhorn_rew_scale = 1.  # Set after first episode

			self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])

		metrics = None
		while train_until_step(self.global_step):
			# if self.global_step % 36 == 0:
			# 	self.save_snapshot(save_step=True, eval=False)

			if repr(self.agent) not in ['bc', 'vinn', 'tcc'] and (time_step.last() or self.global_step%36 == 0) and self.global_step!=0:
				self._global_episode += 1
				if self._global_episode % 10 == 0:
					self.train_video_recorder.save(f'{self.global_frame}.mp4')
				# wait until all the metrics schema is populated
				observations = np.stack(observations, 0)
				actions = np.stack(actions, 0)
				if 'potil' in repr(self.agent):
					new_rewards = self.agent.ot_rewarder(
						observations, self.expert_demo, self.global_step)
					new_rewards_sum = np.sum(new_rewards)
					print("REWARD = ", new_rewards_sum)

				elif repr(self.agent) == 'dac':
					new_rewards = self.agent.dac_rewarder(observations, actions)
					new_rewards_sum = np.sum(new_rewards)
				
				if 'potil' in repr(self.agent):
					if self.agent.auto_rew_scale: 
						if self._global_episode == 1:
							self.agent.sinkhorn_rew_scale = self.agent.sinkhorn_rew_scale * self.agent.auto_rew_scale_factor / float(
								np.abs(new_rewards_sum))
							new_rewards = self.agent.ot_rewarder(
								observations, self.expert_demo, self.global_step)
							new_rewards_sum = np.sum(new_rewards)

				for i, elt in enumerate(time_steps):
					elt = elt._replace(
						observation=time_steps[i].observation[self.cfg.obs_type])
					if 'potil' in repr(self.agent) or repr(self.agent) == 'dac':
							elt = elt._replace(reward=new_rewards[i])
					
					self.replay_storage.add(elt, last = (i == len(time_steps) - 1))

				if metrics is not None:
					# log stats
					elapsed_time, total_time = self.timer.reset()
					episode_frame = episode_step * self.cfg.suite.action_repeat
					with self.logger.log_and_dump_ctx(self.global_frame,
													  ty='train') as log:
						log('fps', episode_frame / elapsed_time)
						log('total_time', total_time)
						log('episode_reward', episode_reward)
						log('episode_length', episode_frame)
						log('episode', self.global_episode)
						log('buffer_size', len(self.replay_storage))
						log('step', self.global_step)
						if 'potil' in repr(self.agent) or repr(self.agent) == 'dac':
								log('expert_reward', self.expert_reward)
								log('imitation_reward', new_rewards_sum)

				# reset env
				time_steps = list()
				observations = list()
				actions = list()

				x = input("Press Enter to continue... after reseting env")

				time_step = self.train_env.reset()

				# reset buffer and openloop count
				self.agent.buffer.reset()
				self.agent.openloop_step_count = 0
				self.agent.count = 0
				
				time_steps.append(time_step)
				observations.append(time_step.observation[self.cfg.obs_type])
				actions.append(time_step.action)
				self.train_video_recorder.init(time_step.observation['pixels'])#self.cfg.obs_type])
				# try to save snapshot
				if self.cfg.suite.save_snapshot:
					self.save_snapshot()
				episode_step = 0
				episode_reward = 0
			
			# try to evaluate
			if repr(self.agent) not in ['bc', 'vinn', 'tcc'] and eval_every_step(self.global_step):
				self.logger.log('eval_total_time', self.timer.total_time(),
								self.global_frame)
				self.timer.eval()
				#if self.global_step > 100: #skip first one 
				#self.eval()
				self.timer.eval()
				self.save_snapshot(save_step=True, eval=True)
				
			if repr(self.agent) not in ['bc', 'vinn', 'tcc']:
				# sample action
				with torch.no_grad(), utils.eval_mode(self.agent):
					action, vinn_action = self.agent.act(time_step.observation[self.cfg.obs_type],
											self.global_step,
											eval_mode=False)
					
			# try to update the agent
			if not seed_until_step(self.global_step):
				# Update
				metrics = self.agent.update(self.replay_iter, self.expert_replay_iter, self._ssl_replay_iter, 
											self.global_step, self.cfg.bc_regularize)
				if repr(self.agent) in ['bc', 'vinn', 'tcc'] and self.global_step%100==0:#1000==0:
					# print(f"Step: {self.global_step}, Loss: {metrics['loss']}")
					self.save_snapshot(save_step=True)
					# if metrics is not None:
					# 	# log stats
					# 	with self.logger.log_and_dump_ctx(self.global_frame,
					# 									ty='train') as log:
					# 		log('step', self.global_step)
					# 		print(metrics['actor_loss'])
					# 		try:
					# 			log('loss', metrics['actor_loss'])
					# 		except:
					# 			log('loss', metrics['loss'])
					# 	self.logger.log_metrics(metrics, self.global_frame, ty='train')
				if repr(self.agent) == 'bc': 
					print(f"Step:{self.global_step} actor_loss:{metrics['actor_loss']} actor_logprob:{metrics['actor_logprob']} actor_ent:{metrics['actor_ent']}")
				
				self.logger.log_metrics(metrics, self.global_frame, ty='train')
				# if repr(self.agent) == 'bc' and self.cfg.suite.save_snapshot:
				# 	self.save_snapshot()
				

			if repr(self.agent) not in ['bc', 'vinn', 'tcc']:
				# take env step
				try:
					time_step = self.train_env.step(action, vinn_action)
				except:
					pass
					# time_step = self.train_env.step(np.random.randn((12)) * 0.1)
				episode_reward += time_step.reward

				time_steps.append(time_step)
				observations.append(time_step.observation[self.cfg.obs_type])
				actions.append(time_step.action)

				self.train_video_recorder.record(time_step.observation['pixels'])#self.cfg.obs_type])
				episode_step += 1
			self._global_step += 1

	def save_snapshot(self, save_step=False, eval=False):
		snapshot = self.work_dir / 'weights'
		snapshot.mkdir(parents=True, exist_ok=True)
		if eval:
			snapshot = snapshot / ('snapshot.pt' if not save_step else f'snapshot_{self.global_step}_eval.pt')
		else:
			snapshot = snapshot / ('snapshot.pt' if not save_step else f'snapshot_{self.global_step}.pt')
		keys_to_save = ['timer', '_global_step', '_global_episode']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		payload.update(self.agent.save_snapshot())
		with snapshot.open('wb') as f:
			torch.save(payload, f)
			
	def load_snapshot(self, snapshot):
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		agent_payload = {}
		for k, v in payload.items():
			if k not in self.__dict__:
				agent_payload[k] = v
		self.agent.load_snapshot(agent_payload)
		#self.agent.load_snapshot_eval(agent_payload)

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
	from train_robot_ssl_hand import WorkspaceIL as W
	workspace = W(cfg)
	
	# Load weights
	if cfg.load_bc:
		# import ipdb; ipdb.set_trace()
		snapshot = Path(cfg.bc_weight)
		if snapshot.exists():
			print(f'resuming bc: {snapshot}')
			workspace.load_snapshot(snapshot)
	workspace.train_il()


if __name__ == '__main__':
	main()
