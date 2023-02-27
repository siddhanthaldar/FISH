import warnings

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import hydra
import numpy as np
import torch

import utils
from video import VideoRecorder
import pickle
import time

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_agent(obs_spec, action_spec, cfg):
	cfg.obs_shape = obs_spec['pixels'].shape
	try:
		cfg.action_shape = action_spec.shape
	except:
		pass
	return hydra.utils.instantiate(cfg)


class Workspace:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.setup()

		self.agent = make_agent(self.eval_env.observation_spec(),
								self.eval_env.action_spec(), cfg.agent)
		self.timer = utils.Timer()
		self._global_step = 0
		self._global_episode = 0
			
	def setup(self):
		# create envs
		self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)
		# Turn off random start
		self.eval_env.random_start = False

		self.video_recorder = VideoRecorder(
			self.work_dir if self.cfg.save_video else None)

	@property
	def global_step(self):
		return self._global_step

	@property
	def global_episode(self):
		return self._global_episode

	@property
	def global_frame(self):
		return self.global_step * self.cfg.action_repeat

	def reset(self, eval_idx):
		if not self.eval_env.enable_arm:
			return np.array([0,0,0], dtype=np.float32)
		self.eval_env.arm_refresh(reset=False)
		# Set start position
		try:
			self.eval_env.set_position(self.start_pos[eval_idx])
		except:
			self.eval_env.arm.set_position(self.start_pos[eval_idx])
			if self.eval_env.arm.keep_gripper_closed:
				self.eval_env.arm.close_gripper_fully()
			else:
				self.eval_env.arm.open_gripper_fully()
		time.sleep(0.4)		
		time_step = self.eval_env.step(np.zeros(self.eval_env.action_spec().shape[0], dtype=np.float32),
									   np.zeros(self.eval_env.action_spec().shape[0], dtype=np.float32))	
		return time_step

	def eval(self):
		# Set model to eval mode
		self.agent.train(False)
		
		episode = 0
		eval_until_episode = utils.Until(self.cfg.num_eval)

		if 'vinn' in repr(self.agent) or 'openloop' in repr(self.agent):
			with open(self.cfg.expert_dataset, 'rb') as f:
				if self.cfg.obs_type == 'pixels':
					self.expert_demo, _, self.expert_action, self.expert_reward = pickle.load(f)
				elif self.cfg.obs_type == 'features':
					_, self.expert_demo, self.expert_action, self.expert_reward = pickle.load(f)

				self.expert_demo = self.expert_demo[:self.cfg.num_demos]
				self.expert_action = self.expert_action[:self.cfg.num_demos]
				self.expert_reward = self.expert_reward[:self.cfg.num_demos]
				self.expert_demo = np.concatenate(self.expert_demo, axis=0)
				self.expert_action = np.concatenate(self.expert_action, axis=0)
				self.agent.save_representations(self.expert_demo, self.expert_action, 128)
		
		# Get start points
		if self.cfg.random_start:
			eval_starts = Path(self.cfg.eval_starts) / 'starts.pkl'
			if eval_starts.exists():
				with eval_starts.open('rb') as f:
					self.start_pos = pickle.load(f)
			else:
				eval_starts = Path(self.cfg.eval_starts)
				eval_starts.mkdir(parents=True, exist_ok=True)
				
				# Generate start points
				self.start_pos = []
				try:
					for _ in range(self.cfg.num_eval):
						self.start_pos.append(self.eval_env.get_random_pos())
				except:
					for _ in range(self.cfg.num_eval):
						self.start_pos.append(self.eval_env.arm.get_random_pos())

				# Save start points for the task
				eval_starts = eval_starts / 'starts.pkl'
				with eval_starts.open('wb') as f:
					pickle.dump(self.start_pos, f)
		
		self.video_recorder.init(self.eval_env)
		while eval_until_episode(episode):
			print(f"Episode {episode}")
			step = 0
			time_step = self.eval_env.reset()
			x = input("Press Enter to continue... after reseting env")
			time_step = self.eval_env.reset()
			if self.cfg.random_start:
				time_step = self.reset(episode)
				time.sleep(2)#5)
			if 'vinn' in repr(self.agent):
				self.agent.buffer.reset()
			if 'openloop' in repr(self.agent):
				self.agent.curr_step = 0
			while not time_step.last():
				
				with torch.no_grad(), utils.eval_mode(self.agent):
					action, vinn_action = self.agent.act(
						time_step.observation['pixels'],
						self.global_step,
						eval_mode=True)
				# action = np.clip(action, -1, 1)
				time_step = self.eval_env.step(action, vinn_action)
				self.video_recorder.record(self.eval_env)
				step += 1
			episode += 1
		
		self.video_recorder.save(f'{episode}_eval.mp4')
			
	def load_snapshot(self, snapshot, bc=False):
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		agent_payload = {}
		for k, v in payload.items():
			if k not in self.__dict__:
				agent_payload[k] = v
		self.agent.load_snapshot_eval(agent_payload, bc)


@hydra.main(config_path='cfgs', config_name='config_eval')
def main(cfg):
	from eval_robot import Workspace as W
	root_dir = Path.cwd()
	workspace = W(cfg)

	# Load BC weights
	if cfg.load_bc_weight:
		snapshot = Path(cfg.bc_weight)
		if snapshot.exists():
			print(f'resuming: {snapshot}')
			workspace.load_snapshot(snapshot, bc=True)
		else:
			print(f"Could not load bc weight: {snapshot}")
			exit()

	# Load residual weights	
	if cfg.load_residual_weight:
		snapshot = Path(cfg.residual_weight)
		if snapshot.exists():
			print(f'resuming: {snapshot}')
			workspace.load_snapshot(snapshot)
		else:
			print(f"Could not load residual weight: {snapshot}")
			exit()
	
	workspace.eval()


if __name__ == '__main__':
	main()
