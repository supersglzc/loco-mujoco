"""
N-step return helper for vec envs (per-env nstep window). Numpy host-side.

Same shape contract as custom_torch's NStepReplay.add_to_buffer:
    inputs: obs (n_envs, T, obs), action (n_envs, T, A), reward (n_envs, T, 1),
            next_obs (n_envs, T, obs), done (n_envs, T, 1)
    output: (obs, action, reward, next_obs, done) flat across envs and time
"""
from __future__ import annotations

import numpy as np


class NStepReplay:
    def __init__(self, obs_dim, action_dim: int, num_envs: int = 1,
                 nstep: int = 3, gamma: float = 0.99):
        self.obs_dim = (obs_dim,) if isinstance(obs_dim, int) else tuple(obs_dim)
        self.action_dim = int(action_dim)
        self.num_envs = int(num_envs)
        self.nstep = int(nstep)
        self.gamma = float(gamma)
        self.nstep_buf_obs = np.empty((self.num_envs, self.nstep, *self.obs_dim), dtype=np.float32)
        self.nstep_buf_action = np.empty((self.num_envs, self.nstep, self.action_dim), dtype=np.float32)
        self.nstep_buf_next_obs = np.empty((self.num_envs, self.nstep, *self.obs_dim), dtype=np.float32)
        self.nstep_buf_reward = np.empty((self.num_envs, self.nstep, 1), dtype=np.float32)
        self.nstep_buf_done = np.empty((self.num_envs, self.nstep, 1), dtype=bool)
        self.nstep_count = 0
        self.gamma_array = np.array(
            [self.gamma ** i for i in range(self.nstep)], dtype=np.float32
        ).reshape(-1, 1)

    def add_to_buffer(self, obs, actions, rewards, next_obs, dones):
        if self.nstep <= 1:
            return obs, actions, rewards, next_obs, dones

        out_obs, out_action, out_reward, out_next_obs, out_done = [], [], [], [], []
        for i in range(obs.shape[1]):
            self.nstep_buf_obs = self._fifo(self.nstep_buf_obs, obs[:, i])
            self.nstep_buf_next_obs = self._fifo(self.nstep_buf_next_obs, next_obs[:, i])
            self.nstep_buf_done = self._fifo(self.nstep_buf_done, dones[:, i])
            self.nstep_buf_action = self._fifo(self.nstep_buf_action, actions[:, i])
            self.nstep_buf_reward = self._fifo(self.nstep_buf_reward, rewards[:, i])
            self.nstep_count += 1
            if self.nstep_count < self.nstep:
                continue
            out_obs.append(self.nstep_buf_obs[:, 0])
            out_action.append(self.nstep_buf_action[:, 0])
            r, no, d = _compute_nstep_return(
                self.nstep_buf_next_obs, self.nstep_buf_done,
                self.nstep_buf_reward, self.gamma_array,
            )
            out_reward.append(r)
            out_next_obs.append(no)
            out_done.append(d)
        if not out_obs:
            return None
        return (
            np.concatenate(out_obs),
            np.concatenate(out_action),
            np.concatenate(out_reward),
            np.concatenate(out_next_obs),
            np.concatenate(out_done),
        )

    @staticmethod
    def _fifo(queue, new_arr):
        return np.concatenate([queue[:, 1:], new_arr[:, None]], axis=1)


def _compute_nstep_return(buf_next_obs, buf_done, buf_reward, gamma_array):
    buf_done_flat = buf_done.squeeze(-1)
    n_envs, n_step = buf_done_flat.shape
    done_envs = np.where(buf_done_flat.any(axis=1))[0]
    done_steps = np.argmax(buf_done_flat, axis=1)

    done = buf_done[:, -1].copy()
    next_obs = buf_next_obs[:, -1].copy()
    if len(done_envs) > 0:
        done[done_envs] = True
        for e in done_envs:
            s = int(done_steps[e])
            next_obs[e] = buf_next_obs[e, s]

    mask = np.ones(buf_done_flat.shape, dtype=bool)
    if len(done_envs) > 0:
        for e in done_envs:
            s = int(done_steps[e])
            mask[e, s + 1:] = False
    discounted = buf_reward * gamma_array
    discounted = (discounted * mask[..., None]).sum(axis=1)
    return discounted, next_obs, done
