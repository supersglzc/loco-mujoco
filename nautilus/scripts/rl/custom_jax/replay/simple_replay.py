"""
Numpy cyclic replay buffer for off-policy algorithms (SAC / TD3).

Same API surface as custom_torch's ReplayBuffer (`add_to_buffer(traj)` /
`sample_batch(bs)`) but stored as numpy on the host. JAX agents convert to
jnp at sample time so the GPU only sees the minibatch.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim, action_dim: int):
        self.obs_dim = (obs_dim,) if isinstance(obs_dim, int) else tuple(obs_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.next_p = 0
        self.if_full = False
        self.cur_capacity = 0
        self.buf_obs = np.empty((self.capacity, *self.obs_dim), dtype=np.float32)
        self.buf_action = np.empty((self.capacity, self.action_dim), dtype=np.float32)
        self.buf_reward = np.empty((self.capacity, 1), dtype=np.float32)
        self.buf_next_obs = np.empty((self.capacity, *self.obs_dim), dtype=np.float32)
        self.buf_done = np.empty((self.capacity, 1), dtype=bool)

    def add_to_buffer(self, trajectory):
        obs, actions, rewards, next_obs, dones = trajectory
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, *self.obs_dim)
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, self.action_dim)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1, *self.obs_dim)
        dones = np.asarray(dones, dtype=bool).reshape(-1, 1)
        n = rewards.shape[0]
        p = self.next_p + n

        if p > self.capacity:
            self.if_full = True
            head = self.capacity - self.next_p
            self.buf_obs[self.next_p:self.capacity] = obs[:head]
            self.buf_action[self.next_p:self.capacity] = actions[:head]
            self.buf_reward[self.next_p:self.capacity] = rewards[:head]
            self.buf_next_obs[self.next_p:self.capacity] = next_obs[:head]
            self.buf_done[self.next_p:self.capacity] = dones[:head]
            tail = p - self.capacity
            self.buf_obs[:tail] = obs[head:head + tail]
            self.buf_action[:tail] = actions[head:head + tail]
            self.buf_reward[:tail] = rewards[head:head + tail]
            self.buf_next_obs[:tail] = next_obs[head:head + tail]
            self.buf_done[:tail] = dones[head:head + tail]
            p = tail
        else:
            self.buf_obs[self.next_p:p] = obs
            self.buf_action[self.next_p:p] = actions
            self.buf_reward[self.next_p:p] = rewards
            self.buf_next_obs[self.next_p:p] = next_obs
            self.buf_done[self.next_p:p] = dones

        self.next_p = p
        self.cur_capacity = self.capacity if self.if_full else self.next_p

    def sample_batch(self, batch_size: int):
        idx = np.random.randint(0, self.cur_capacity, size=batch_size)
        return (
            self.buf_obs[idx],
            self.buf_action[idx],
            self.buf_reward[idx],
            self.buf_next_obs[idx],
            self.buf_done[idx].astype(np.float32),
        )
