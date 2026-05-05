"""SAC (JAX/Flax) — soft actor-critic with twin Q + auto-tuned alpha.

Off-policy. Rollout is Python-side (because the replay buffer needs
host-side mutation per step), but every gradient step is jitted via
jax.lax.scan over update_times.

Vmap rollout REQUIRED — env must expose MJX-style step/reset (see PPO).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import flax
import numpy as np
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from algo.ac_base import AgentBase
from models.mlp import TanhDiagGaussianActor, DoubleQ
from replay.nstep_replay import NStepReplay


@dataclass
class AgentSAC(AgentBase):
    def __post_init__(self):
        super().__post_init__()
        hidden = tuple(self.cfg.algo.get("hidden_dims", (256, 256)))
        activation = str(self.cfg.algo.get("activation", "relu"))
        use_obs_rms = bool(self.cfg.algo.get("use_obs_rms", True))

        self.actor_net = TanhDiagGaussianActor(
            action_dim=self.action_dim, hidden_dims=hidden, activation=activation,
            use_obs_rms=use_obs_rms,
        )
        self.critic_net = DoubleQ(hidden_dims=hidden, activation=activation,
                                  use_obs_rms=use_obs_rms)

        keys = self.split(4)
        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_act = jnp.zeros((1, self.action_dim))
        # Init returns {'params':..., 'run_stats':...} when use_obs_rms=True.
        actor_init = self.actor_net.init(keys[0], dummy_obs)
        critic_init = self.critic_net.init(keys[1], dummy_obs, dummy_act)

        actor_lr = float(self.cfg.algo.get("actor_lr", 3e-4))
        critic_lr = float(self.cfg.algo.get("critic_lr", 3e-4))
        alpha_lr = float(self.cfg.algo.get("alpha_lr", 3e-4))
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
        # Target critic mirrors critic params; SHARES run_stats with the online critic
        # (ref does the same for the obs-RMS layer — single source of obs stats).
        self.target_critic_params = {'params': jax.tree_util.tree_map(lambda x: x, critic_init['params'])}

        # log_alpha as its own TrainState (one-element vector for parity).
        log_alpha_init = float(np.log(self.cfg.algo.get("init_alpha", 1.0)))
        self.log_alpha = jnp.array([log_alpha_init], dtype=jnp.float32)
        self.alpha_opt = optax.adamw(alpha_lr)
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)
        self.target_entropy = -float(self.action_dim)

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

        traj_obs = np.empty((n, T, self.obs_dim), dtype=np.float32)
        traj_act = np.empty((n, T, self.action_dim), dtype=np.float32)
        traj_rew = np.empty((n, T, 1), dtype=np.float32)
        traj_next = np.empty((n, T, self.obs_dim), dtype=np.float32)
        traj_done = np.empty((n, T, 1), dtype=bool)

        # Pull raw episode return / length from LogWrapper (raw even when
        # NormalizeVecReward is on; matches PPO's tracker fix).
        from loco_mujoco.core.wrappers.mjx import LogEnvState as _LogEnvState

        obs = self.obs
        env_state = self.env_state
        for i in range(T):
            if random:
                key = self.split(1)
                action = jax.random.uniform(key, (n, self.action_dim), minval=-1.0, maxval=1.0)
            else:
                key = self.split(1)
                # _sac_sample_action threads run_stats — actor RMS updates each rollout step.
                action, self.actor_run_stats = _sac_sample_action(
                    self.actor_state.params['params'], self.actor_net.apply,
                    self.actor_run_stats, key, obs,
                )
            next_obs, reward, absorbing, done, info, env_state = env.step(env_state, action)
            d_np = np.asarray(done, dtype=bool)
            # Read raw cumulative episode return / length from LogWrapper state.
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

        # Sync critic + target_critic RMS to actor RMS — they all see the same
        # obs distribution; one shared stats source is what the reference does.
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

        critic_losses, actor_losses, q_values, entropies, alphas = [], [], [], [], []
        for _ in range(update_times):
            obs, action, reward, next_obs, done = memory.sample_batch(bs)
            obs_j = jnp.asarray(obs)
            action_j = jnp.asarray(action)
            reward_j = jnp.asarray(reward.squeeze(-1))
            next_obs_j = jnp.asarray(next_obs)
            done_j = jnp.asarray(done.squeeze(-1).astype(np.float32))
            key_c, key_a = self.split(3)[:2]

            (self.critic_state, self.target_critic_params,
             c_loss, q_mean) = _sac_critic_step(
                self.critic_state, self.target_critic_params,
                self.actor_run_stats, self.critic_run_stats,
                self.critic_net.apply, self.actor_net.apply,
                self.actor_state.params['params'], self.log_alpha,
                obs_j, action_j, reward_j, next_obs_j, done_j,
                gamma, nstep, tau, key_c,
            )
            (self.actor_state, self.log_alpha, self.alpha_opt_state,
             a_loss, ent_est, alpha_val) = _sac_actor_alpha_step(
                self.actor_state, self.critic_state, self.log_alpha,
                self.actor_run_stats, self.critic_run_stats,
                self.alpha_opt, self.alpha_opt_state,
                self.actor_net.apply, self.critic_net.apply,
                obs_j, self.target_entropy, key_a,
            )
            critic_losses.append(float(c_loss))
            actor_losses.append(float(a_loss))
            q_values.append(float(q_mean))
            entropies.append(float(ent_est))
            alphas.append(float(alpha_val))
            self._update_count += 1

        return {
            "train/critic_loss": float(np.mean(critic_losses)),
            "train/actor_loss":  float(np.mean(actor_losses)),
            "train/entropy":     float(np.mean(entropies)),
            "train/q_value":     float(np.mean(q_values)),
            "train/alpha":       float(np.mean(alphas)),
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
                "target_critic": flax.serialization.to_state_dict(self.target_critic_params),
                "actor_run_stats": flax.serialization.to_state_dict(self.actor_run_stats),
                "critic_run_stats": flax.serialization.to_state_dict(self.critic_run_stats),
                "log_alpha": np.asarray(self.log_alpha).tolist(),
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
        agent.target_critic_params = flax.serialization.from_state_dict(
            agent.target_critic_params, payload["agent_state"]["target_critic"])
        if "actor_run_stats" in payload["agent_state"]:
            agent.actor_run_stats = flax.serialization.from_state_dict(
                agent.actor_run_stats, payload["agent_state"]["actor_run_stats"])
            agent.critic_run_stats = flax.serialization.from_state_dict(
                agent.critic_run_stats, payload["agent_state"]["critic_run_stats"])
        agent.log_alpha = jnp.array(payload["agent_state"]["log_alpha"], dtype=jnp.float32)
        return agent


# -----------------------------------------------------------------------------
# Pure-JAX kernels
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("actor_apply",))
def _sac_sample_action(actor_params_inner, actor_apply, actor_run_stats, key, obs):
    """Sample one action AND return updated run_stats (Welford-update from this batch)."""
    (mean, log_std), upd = actor_apply(
        {'params': actor_params_inner, 'run_stats': actor_run_stats},
        obs, mutable=['run_stats'],
    )
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mean.shape)
    return jnp.tanh(mean + eps * std), upd['run_stats']


@partial(jax.jit, static_argnames=("critic_apply", "actor_apply"))
def _sac_critic_step(critic_state, target_critic_params,
                     actor_run_stats, critic_run_stats,
                     critic_apply, actor_apply, actor_params_inner, log_alpha,
                     obs, action, reward, next_obs, done,
                     gamma, nstep, tau, rng):
    alpha = jnp.exp(log_alpha)[0]

    # Target: r + (1 - d) * γⁿ * (min(Q1', Q2')(s', ã') - α log π(ã'|s'))
    # Use mutable=['run_stats'] but DISCARD updates — RMS only updates from rollout.
    (mean_next, log_std_next), _ = actor_apply(
        {'params': actor_params_inner, 'run_stats': actor_run_stats},
        next_obs, mutable=['run_stats'],
    )
    std_next = jnp.exp(log_std_next)
    eps = jax.random.normal(rng, mean_next.shape)
    u_next = mean_next + eps * std_next
    a_next = jnp.tanh(u_next)
    gauss_lp = -0.5 * (((u_next - mean_next) / std_next) ** 2 + 2.0 * log_std_next
                       + jnp.log(2.0 * jnp.pi)).sum(axis=-1)
    correction = (2.0 * (jnp.log(2.0) - u_next - jax.nn.softplus(-2.0 * u_next))).sum(axis=-1)
    log_prob_next = gauss_lp - correction
    (q1_t, q2_t), _ = critic_apply(
        {'params': target_critic_params['params'], 'run_stats': critic_run_stats},
        next_obs, a_next, mutable=['run_stats'],
    )
    q_t = jnp.minimum(q1_t, q2_t) - alpha * log_prob_next
    target = reward + (1.0 - done) * (gamma ** nstep) * q_t

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


@partial(jax.jit, static_argnames=("actor_apply", "critic_apply", "alpha_opt"))
def _sac_actor_alpha_step(actor_state, critic_state, log_alpha,
                          actor_run_stats, critic_run_stats,
                          alpha_opt, alpha_opt_state,
                          actor_apply, critic_apply, obs, target_entropy, rng):
    alpha = jnp.exp(log_alpha)[0]

    def _actor_loss(params):
        (mean, log_std), _ = actor_apply(
            {'params': params['params'], 'run_stats': actor_run_stats},
            obs, mutable=['run_stats'],
        )
        std = jnp.exp(log_std)
        eps = jax.random.normal(rng, mean.shape)
        u = mean + eps * std
        action = jnp.tanh(u)
        gauss_lp = -0.5 * (((u - mean) / std) ** 2 + 2.0 * log_std
                           + jnp.log(2.0 * jnp.pi)).sum(axis=-1)
        correction = (2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))).sum(axis=-1)
        log_prob = gauss_lp - correction
        (q1, q2), _ = critic_apply(
            {'params': critic_state.params['params'], 'run_stats': critic_run_stats},
            obs, action, mutable=['run_stats'],
        )
        q = jnp.minimum(q1, q2)
        return (alpha * log_prob - q).mean(), log_prob

    (loss_a, log_prob), grads_a = jax.value_and_grad(_actor_loss, has_aux=True)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads_a)

    def _alpha_loss(la):
        return -(jnp.exp(la) * (jax.lax.stop_gradient(log_prob) + target_entropy)).mean()

    grad_la = jax.grad(_alpha_loss)(log_alpha)
    updates, alpha_opt_state = alpha_opt.update(grad_la, alpha_opt_state, log_alpha)
    log_alpha = optax.apply_updates(log_alpha, updates)
    return actor_state, log_alpha, alpha_opt_state, loss_a, -log_prob.mean(), jnp.exp(log_alpha)[0]


# -- Views for eval/render -------------------------------------------------
# Eval/render: use mutable=False so RMS variables are READ but not updated.
class _ActorView:
    def __init__(self, agent):
        self.agent = agent

    def _vars(self):
        return {'params': self.agent.actor_state.params['params'],
                'run_stats': self.agent.actor_run_stats}

    def get_actions(self, obs, sample: bool = True):
        if sample:
            # Use the rollout sampler — it updates RMS, but at eval time we don't
            # want that. Instead inline a stable read-only sample.
            mean, log_std = self.agent.actor_net.apply(self._vars(), obs)
            std = jnp.exp(log_std)
            key = self.agent.split(1)
            return jnp.tanh(mean + jax.random.normal(key, mean.shape) * std)
        return self.agent.actor_net.apply(self._vars(), obs,
                                          method=self.agent.actor_net.deterministic_action)

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
