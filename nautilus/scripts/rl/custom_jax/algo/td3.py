"""TD3 (JAX/Flax) — twin-delayed DDPG with target smoothing.

Off-policy. Same env contract as PPO/SAC: vmap rollout REQUIRED (env exposes
MJX-style step/reset). Rollout is host-side Python loop because of replay
buffer interaction; gradient kernels are jitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import flax
import numpy as np
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from algo.ac_base import AgentBase
from models.mlp import TanhDeterministicActor, DoubleQ
from replay.nstep_replay import NStepReplay


@dataclass
class AgentTD3(AgentBase):
    def __post_init__(self):
        super().__post_init__()
        hidden = tuple(self.cfg.algo.get("hidden_dims", (256, 256)))
        activation = str(self.cfg.algo.get("activation", "relu"))
        use_obs_rms = bool(self.cfg.algo.get("use_obs_rms", True))

        self.actor_net = TanhDeterministicActor(
            action_dim=self.action_dim, hidden_dims=hidden, activation=activation,
            use_obs_rms=use_obs_rms,
        )
        self.critic_net = DoubleQ(hidden_dims=hidden, activation=activation,
                                  use_obs_rms=use_obs_rms)

        keys = self.split(3)
        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_act = jnp.zeros((1, self.action_dim))
        actor_init = self.actor_net.init(keys[0], dummy_obs)
        critic_init = self.critic_net.init(keys[1], dummy_obs, dummy_act)

        actor_lr = float(self.cfg.algo.get("actor_lr", 3e-4))
        critic_lr = float(self.cfg.algo.get("critic_lr", 3e-4))
        mgn = self.cfg.algo.get("max_grad_norm", None)
        mgn = None if mgn is None else float(mgn)

        # train_state.params holds {'params': ...} only; run_stats lives separately.
        self.actor_state = TrainState.create(
            apply_fn=self.actor_net.apply,
            params={'params': actor_init['params']},
            tx=self.make_optimizer(actor_lr, max_grad_norm=mgn),
        )
        self.critic_state = TrainState.create(
            apply_fn=self.critic_net.apply,
            params={'params': critic_init['params']},
            tx=self.make_optimizer(critic_lr, max_grad_norm=mgn),
        )
        self.actor_run_stats = actor_init.get('run_stats', {})
        self.critic_run_stats = critic_init.get('run_stats', {})
        # Target nets: same params + same RMS as their online counterparts at init.
        self.target_actor_params = {'params': jax.tree_util.tree_map(lambda x: x, actor_init['params'])}
        self.target_critic_params = {'params': jax.tree_util.tree_map(lambda x: x, critic_init['params'])}

        self.n_step = NStepReplay(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            num_envs=int(self.cfg.num_envs),
            nstep=int(self.cfg.algo.get("nstep", 1)),
            gamma=float(self.cfg.algo.get("gamma", 0.99)),
        )
        self.env_state = None
        self._update_count = 0

    @property
    def actor(self):
        return _ActorView(self)

    @property
    def critic(self):
        return _CriticView(self)

    def reset_agent(self):
        n = int(self.cfg.num_envs)
        # loco-mujoco's VecEnv.reset is jax.vmap'd over rng_key axis 0; it expects
        # an (N, 2) array of split PRNGKeys. NO env_id kwarg is supported.
        reset_rngs = jax.random.split(self.split(2), n)
        obs, env_state = self.env.reset(reset_rngs)
        self.obs = obs
        self.dones = jnp.zeros(n, dtype=jnp.float32)
        self.env_state = env_state
        return self.obs

    def explore_env(self, env, timesteps: int, random: bool):
        if self.env_state is None:
            self.reset_agent()
        n = int(self.cfg.num_envs)
        T = int(timesteps)
        explore_noise = float(self.cfg.algo.get("explore_noise", 0.1))

        traj_obs = np.empty((n, T, self.obs_dim), dtype=np.float32)
        traj_act = np.empty((n, T, self.action_dim), dtype=np.float32)
        traj_rew = np.empty((n, T, 1), dtype=np.float32)
        traj_next = np.empty((n, T, self.obs_dim), dtype=np.float32)
        traj_done = np.empty((n, T, 1), dtype=bool)

        # Raw episode return / length from LogWrapper (raw even with NormalizeVecReward).
        from loco_mujoco.core.wrappers.mjx import LogEnvState as _LogEnvState

        obs = self.obs
        env_state = self.env_state
        for i in range(T):
            if random:
                key = self.split(1)
                action = jax.random.uniform(key, (n, self.action_dim), minval=-1.0, maxval=1.0)
            else:
                key = self.split(1)
                # Threaded run_stats — actor RMS updates each rollout step.
                action, self.actor_run_stats = _td3_sample_action(
                    self.actor_state.params['params'], self.actor_net.apply,
                    self.actor_run_stats, key, obs, explore_noise,
                )
            next_obs, reward, absorbing, done, info, env_state = env.step(env_state, action)
            d_np = np.asarray(done, dtype=bool)
            log_state = env_state.find(_LogEnvState)
            rer_np = np.asarray(log_state.metrics.returned_episode_returns)
            rel_np = np.asarray(log_state.metrics.returned_episode_lengths)
            if d_np.any():
                idx = np.where(d_np)[0]
                self.return_tracker.update(rer_np[idx])
                self.step_tracker.update(rel_np[idx])

            traj_obs[:, i] = np.asarray(obs)
            traj_act[:, i] = np.asarray(action)
            traj_rew[:, i, 0] = np.asarray(reward, np.float32) * float(self.cfg.algo.get("reward_scale", 1.0))
            traj_next[:, i] = np.asarray(next_obs)
            traj_done[:, i, 0] = np.asarray(absorbing, dtype=bool)
            obs = next_obs
        self.obs = obs
        self.env_state = env_state

        # Sync critic + target_actor + target_critic RMS to actor RMS — single source of obs stats.
        self.critic_run_stats = self.actor_run_stats

        nstep = int(self.cfg.algo.get("nstep", 1))
        if nstep > 1:
            data = self.n_step.add_to_buffer(traj_obs, traj_act, traj_rew, traj_next, traj_done)
        else:
            data = (traj_obs, traj_act, traj_rew, traj_next, traj_done)
        return data, T * n

    def update_net(self, memory):
        update_times = int(self.cfg.algo.get("update_times", 1))
        bs = int(self.cfg.algo.get("batch_size", 256))
        tau = float(self.cfg.algo.get("tau", 0.005))
        gamma = float(self.cfg.algo.get("gamma", 0.99))
        nstep = int(self.cfg.algo.get("nstep", 1))
        policy_delay = int(self.cfg.algo.get("policy_delay", 2))
        policy_noise = float(self.cfg.algo.get("policy_noise", 0.2))
        noise_clip = float(self.cfg.algo.get("noise_clip", 0.5))

        critic_losses, actor_losses, q_values = [], [], []
        for _ in range(update_times):
            obs, action, reward, next_obs, done = memory.sample_batch(bs)
            obs_j = jnp.asarray(obs)
            action_j = jnp.asarray(action)
            reward_j = jnp.asarray(reward.squeeze(-1))
            next_obs_j = jnp.asarray(next_obs)
            done_j = jnp.asarray(done.squeeze(-1).astype(np.float32))
            key_c = self.split(1)

            (self.critic_state, self.target_critic_params,
             c_loss, q_mean) = _td3_critic_step(
                self.critic_state, self.target_actor_params,
                self.target_critic_params,
                self.actor_run_stats, self.critic_run_stats,
                self.critic_net.apply, self.actor_net.apply,
                obs_j, action_j, reward_j, next_obs_j, done_j,
                gamma, nstep, tau, policy_noise, noise_clip, key_c,
            )
            critic_losses.append(float(c_loss))
            q_values.append(float(q_mean))

            self._update_count += 1
            if self._update_count % policy_delay == 0:
                (self.actor_state, self.target_actor_params,
                 a_loss) = _td3_actor_step(
                    self.actor_state, self.target_actor_params,
                    self.critic_state.params,
                    self.actor_run_stats, self.critic_run_stats,
                    self.actor_net.apply, self.critic_net.apply,
                    obs_j, tau,
                )
                actor_losses.append(float(a_loss))

        return {
            "train/critic_loss": float(np.mean(critic_losses)),
            "train/actor_loss":  float(np.mean(actor_losses)) if actor_losses else 0.0,
            "train/q_value":     float(np.mean(q_values)),
            "train/episode_length": float(self.step_tracker.mean()),
            **self.reward_log_info(),
        }

    def serialize(self) -> dict:
        return {
            "agent_conf": {
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "obs_dim": int(self.obs_dim),
                "action_dim": int(self.action_dim),
            },
            "agent_state": {
                "actor": flax.serialization.to_state_dict(self.actor_state),
                "critic": flax.serialization.to_state_dict(self.critic_state),
                "target_actor": flax.serialization.to_state_dict(self.target_actor_params),
                "target_critic": flax.serialization.to_state_dict(self.target_critic_params),
                "actor_run_stats": flax.serialization.to_state_dict(self.actor_run_stats),
                "critic_run_stats": flax.serialization.to_state_dict(self.critic_run_stats),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict, env, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create(payload["agent_conf"]["config"])
        agent = cls(env=env, cfg=cfg,
                    obs_dim=payload["agent_conf"]["obs_dim"],
                    action_dim=payload["agent_conf"]["action_dim"])
        agent.actor_state = flax.serialization.from_state_dict(
            agent.actor_state, payload["agent_state"]["actor"])
        agent.critic_state = flax.serialization.from_state_dict(
            agent.critic_state, payload["agent_state"]["critic"])
        agent.target_actor_params = flax.serialization.from_state_dict(
            agent.target_actor_params, payload["agent_state"]["target_actor"])
        agent.target_critic_params = flax.serialization.from_state_dict(
            agent.target_critic_params, payload["agent_state"]["target_critic"])
        if "actor_run_stats" in payload["agent_state"]:
            agent.actor_run_stats = flax.serialization.from_state_dict(
                agent.actor_run_stats, payload["agent_state"]["actor_run_stats"])
            agent.critic_run_stats = flax.serialization.from_state_dict(
                agent.critic_run_stats, payload["agent_state"]["critic_run_stats"])
        return agent


# -----------------------------------------------------------------------------
# Pure-JAX kernels
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("actor_apply",))
def _td3_sample_action(actor_params_inner, actor_apply, actor_run_stats, key, obs, explore_noise):
    """Sample one TD3 action AND return updated run_stats."""
    a_mean, upd = actor_apply(
        {'params': actor_params_inner, 'run_stats': actor_run_stats},
        obs, mutable=['run_stats'],
    )
    noise = jax.random.normal(key, a_mean.shape) * explore_noise
    return jnp.clip(a_mean + noise, -1.0, 1.0), upd['run_stats']


@partial(jax.jit, static_argnames=("critic_apply", "actor_apply"))
def _td3_critic_step(critic_state, target_actor_params, target_critic_params,
                     actor_run_stats, critic_run_stats,
                     critic_apply, actor_apply,
                     obs, action, reward, next_obs, done,
                     gamma, nstep, tau, policy_noise, noise_clip, rng):
    # Apply with mutable=['run_stats'] but DISCARD updates (RMS only updates from rollout).
    a_next, _ = actor_apply(
        {'params': target_actor_params['params'], 'run_stats': actor_run_stats},
        next_obs, mutable=['run_stats'],
    )
    noise = jnp.clip(jax.random.normal(rng, a_next.shape) * policy_noise,
                     -noise_clip, noise_clip)
    a_next = jnp.clip(a_next + noise, -1.0, 1.0)
    (q1_t, q2_t), _ = critic_apply(
        {'params': target_critic_params['params'], 'run_stats': critic_run_stats},
        next_obs, a_next, mutable=['run_stats'],
    )
    target = reward + (1.0 - done) * (gamma ** nstep) * jnp.minimum(q1_t, q2_t)

    def _loss(params):
        (q1, q2), _ = critic_apply(
            {'params': params['params'], 'run_stats': critic_run_stats},
            obs, action, mutable=['run_stats'],
        )
        return ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean(), \
               jnp.minimum(q1, q2).mean()

    (loss, q_mean), grads = jax.value_and_grad(_loss, has_aux=True)(critic_state.params)
    critic_state = critic_state.apply_gradients(grads=grads)
    target_critic_params = jax.tree_util.tree_map(
        lambda t, c: tau * c + (1.0 - tau) * t, target_critic_params, critic_state.params,
    )
    return critic_state, target_critic_params, loss, q_mean


@partial(jax.jit, static_argnames=("actor_apply", "critic_apply"))
def _td3_actor_step(actor_state, target_actor_params, critic_params,
                    actor_run_stats, critic_run_stats,
                    actor_apply, critic_apply, obs, tau):
    def _loss(params):
        a, _ = actor_apply(
            {'params': params['params'], 'run_stats': actor_run_stats},
            obs, mutable=['run_stats'],
        )
        (q1, _q2), _ = critic_apply(
            {'params': critic_params['params'], 'run_stats': critic_run_stats},
            obs, a, mutable=['run_stats'],
        )
        return -q1.mean()

    loss, grads = jax.value_and_grad(_loss)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    target_actor_params = jax.tree_util.tree_map(
        lambda t, c: tau * c + (1.0 - tau) * t, target_actor_params, actor_state.params,
    )
    return actor_state, target_actor_params, loss


# -- Views for eval/render -------------------------------------------------
class _ActorView:
    def __init__(self, agent):
        self.agent = agent

    def get_actions(self, obs, sample: bool = True):
        return self.agent.actor_net.apply(
            {'params': self.agent.actor_state.params['params'],
             'run_stats': self.agent.actor_run_stats}, obs)

    def __call__(self, obs):
        return self.get_actions(obs, sample=False)


class _CriticView:
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, obs, action):
        q1, q2 = self.agent.critic_net.apply(
            {'params': self.agent.critic_state.params['params'],
             'run_stats': self.agent.critic_run_stats},
            obs, action,
        )
        return jnp.minimum(q1, q2)
