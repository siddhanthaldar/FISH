import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import pickle

# from holobot.robot.allegro.allegro_kdl import AllegroKDL
# from holodex.robot.allegro_kdl import AllegroKDL
from .allegro_kdl import AllegroKDL


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step, last=False):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if last or time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        vinn_action = episode['vinn_action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        vinn_next_action = episode['vinn_action'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount

        return (obs, action, vinn_action, reward, discount, next_obs, vinn_next_action)

    def __iter__(self):
        while True:
            yield self._sample()


class ExpertReplayBuffer(IterableDataset):
    def __init__(self, dataset_path, num_demos, obs_type):
        with open(dataset_path, 'rb') as f:
            # ipdb.set_trace()
            if obs_type == 'pixels':
                obses, _, actions, _ = pickle.load(f)
                # _, obses, _, actions, _ = pickle.load(f)
                obses = np.array(obses)
                # obses = obses[:,:,-3:]
            elif obs_type == 'features':
                _, obses, actions, _ = pickle.load(f)

        if num_demos is None:
            num_demos = len(obses)

        _robot = AllegroKDL()

        self._episodes = []
        for i in range(num_demos):
            converted_action = np.zeros((actions[i].shape[0], 12))
            for j in range(actions[i].shape[0]):
                converted_action[j] = _robot.get_fingertip_coords(actions[i][j])
            episode = dict(observation=obses[i], action=converted_action * 5)
            self._episodes.append(episode)

    def _sample_episode(self):
        episode = random.choice(self._episodes)
        return episode

    def _sample(self):
        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode)) + 1
        obs = episode['observation'][idx]
        action = episode['action'][idx]

        return (obs, action)

    def __iter__(self):
        while True:
            yield self._sample()

class SSLReplayBuffer(IterableDataset):
    def __init__(self, dataset_path, num_demos, obs_type, return_episode):
        with open(dataset_path, 'rb') as f:
            if obs_type == 'pixels':
                obses, _, actions, _ = pickle.load(f)
                obses = np.array(obses)
            elif obs_type == 'features':
                _, obses, actions, _ = pickle.load(f)

        if num_demos is None:
            num_demos = len(obses)

        _robot = AllegroKDL()

        self._episodes = []
        for i in range(num_demos):
            converted_action = np.zeros((actions[i].shape[0], 12))
            for j in range(actions[i].shape[0]):
                converted_action[j] = _robot.get_fingertip_coords(actions[i][j])
            episode = dict(observation=obses[i], action=converted_action * 5)
            self._episodes.append(episode)

        self._return_episode = return_episode

    def get_data(self):
        observations = []
        actions = []
        for episode in self._episodes:
            observations.append(episode['observation'])
            actions.append(episode['action'])
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        return observations, actions

    def _sample_episode(self, internal=False):
        episode = random.choice(self._episodes)
        if internal:
            return episode
        else:
            return (episode['observation'], episode['action'])

    def _sample(self):
        episode = self._sample_episode(internal=True)
        idx = np.random.randint(0, episode_len(episode)) + 1
        obs = episode['observation'][idx]
        action = episode['action'][idx]

        return (obs, action)

    def __iter__(self):
        while True:
            if self._return_episode:
                yield self._sample_episode()
            else:
                yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader


def make_expert_replay_loader(replay_dir, batch_size, num_demos, obs_type):
    iterable = ExpertReplayBuffer(replay_dir, num_demos, obs_type)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=2,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader

def make_ssl_replay_loader(replay_dir, batch_size, num_demos, obs_type, return_episode):
    iterable = SSLReplayBuffer(replay_dir, num_demos, obs_type, return_episode)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=2,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return iterable, loader