"""PPO (JAX/Flax) — proximal policy optimization with clipped surrogate + GAE.

Ports the algorithmic recipe from loco_mujoco/algorithms/ppo_jax.py but uses
the simpler MLP networks under custom_jax/models/mlp.py (no RunningMeanStd
mutable state, no actor/critic obs grouping). Single-GPU, no pmap.

Vmap rollout is REQUIRED — the env must expose the MJX-style contract:
    env.reset(reset_keys)              -> (obs, env_state)
        # reset_keys: (N, 2) jnp array of split PRNGKeys (one per env)
    env.step(env_state, action)        -> (obs, reward, absorbing, done, info, env_state)

env_wrapper.create_env(cfg) returns such an env (errors loudly if the
underlying benchmark doesn't expose it).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import flax
import numpy as np
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from algo.ac_base import AgentBase
from models.mlp import DiagGaussianActor, MLPCritic


class _Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    absorbing: jnp.ndarray


@dataclass
class AgentPPO(AgentBase):
    def __post_init__(self):
        super().__post_init__()
        hidden = tuple(self.cfg.algo.get("hidden_dims", (256, 256)))
        activation = str(self.cfg.algo.get("activation", "tanh"))
        init_log_std = float(self.cfg.algo.get("init_log_std", 0.0))
        use_obs_rms = bool(self.cfg.algo.get("use_obs_rms", True))

        self.actor_net = DiagGaussianActor(
            action_dim=self.action_dim, hidden_dims=hidden,
            activation=activation, init_log_std=init_log_std,
            use_obs_rms=use_obs_rms,
        )
        self.critic_net = MLPCritic(hidden_dims=hidden, activation=activation,
                                    use_obs_rms=use_obs_rms)

        keys = self.split(3)
        dummy_obs = jnp.zeros((1, self.obs_dim))
        # Init with mutable=['run_stats'] so the network has both `params` and
        # `run_stats` collections. RMS lives outside train_state (no gradient).
        actor_init = self.actor_net.init(keys[0], dummy_obs)
        critic_init = self.critic_net.init(keys[1], dummy_obs)

        actor_lr = float(self.cfg.algo.get("actor_lr", 3e-4))
        critic_lr = float(self.cfg.algo.get("critic_lr", actor_lr))
        wd = float(self.cfg.algo.get("weight_decay", 0.0))
        mgn = self.cfg.algo.get("max_grad_norm", 0.5)
        mgn = None if mgn is None else float(mgn)

        # Split params from run_stats. params goes into TrainState (gradient updates);
        # run_stats stored separately on the agent (Welford-updates from rollout obs).
        self.actor_state = TrainState.create(
            apply_fn=self.actor_net.apply,
            params={'params': actor_init['params']},
            tx=self.make_optimizer(actor_lr, max_grad_norm=mgn, weight_decay=wd),
        )
        self.critic_state = TrainState.create(
            apply_fn=self.critic_net.apply,
            params={'params': critic_init['params']},
            tx=self.make_optimizer(critic_lr, max_grad_norm=mgn, weight_decay=wd),
        )
        self.actor_run_stats = actor_init.get('run_stats', {})
        self.critic_run_stats = critic_init.get('run_stats', {})

        # MJX-style env_state — populated on first reset_agent()
        self.env_state = None

    # -- Public surface (matches custom_torch contract) --------------------
    @property
    def actor(self):  # parity with custom_torch eval/render that use agent.actor
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

    def explore_env(self, env, timesteps: int, random: bool = False):
        if self.env_state is None:
            self.reset_agent()
        keys = self.split(timesteps + 1)
        scan_keys = jnp.stack(list(keys))

        # Episode-return / length tracking now reads RAW values from the
        # underlying LogWrapper's state via env_state.find(LogEnvState).metrics.
        # LogWrapper sits INSIDE NormalizeVecReward in our env stack, so its
        # `episode_returns` accumulates the raw reward signal regardless of
        # whether reward norm is on — making the training-time tracker
        # comparable to /nautilus:rl-eval (which always uses raw reward).
        # See loco_mujoco/algorithms/ppo_jax.py for the same pattern.
        from loco_mujoco.core.wrappers import LogWrapper as _LogWrapper
        from loco_mujoco.core.wrappers.mjx import LogEnvState as _LogEnvState

        def _step(carry, key):
            env_state, last_obs, actor_rs, critic_rs = carry
            # apply with mutable=['run_stats'] threads obs through the in-network
            # RunningMeanStd which Welford-updates from the current batch.
            (mean, log_std), actor_upd = self.actor_net.apply(
                {'params': self.actor_state.params['params'], 'run_stats': actor_rs},
                last_obs, mutable=['run_stats'],
            )
            actor_rs = actor_upd['run_stats']
            std = jnp.exp(log_std)
            eps = jax.random.normal(key, mean.shape)
            action = mean + eps * std
            log_prob = -0.5 * (((action - mean) / std) ** 2 + 2.0 * log_std
                               + jnp.log(2.0 * jnp.pi)).sum(axis=-1)
            value, critic_upd = self.critic_net.apply(
                {'params': self.critic_state.params['params'], 'run_stats': critic_rs},
                last_obs, mutable=['run_stats'],
            )
            critic_rs = critic_upd['run_stats']
            obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)
            # Pull raw cumulative episode return / length from LogEnvState.
            # `returned_episode_returns` carries the raw episodic return at the
            # done step (cleared between completions). Mask by done so we only
            # log COMPLETED episodes (matches loco_mujoco PPOJax callback).
            log_state = env_state.find(_LogEnvState)
            comp_ret = jnp.where(done, log_state.metrics.returned_episode_returns, 0.0)
            comp_len = jnp.where(done, log_state.metrics.returned_episode_lengths, 0.0)
            transition = _Transition(
                obs=last_obs, action=action, log_prob=log_prob, value=value,
                reward=reward, done=done.astype(jnp.float32),
                absorbing=absorbing.astype(jnp.float32),
            )
            return (env_state, obsv, actor_rs, critic_rs), (transition, comp_ret, comp_len, done)

        init_carry = (self.env_state, self.obs,
                      self.actor_run_stats, self.critic_run_stats)
        (env_state, last_obs, actor_rs, critic_rs), \
            (traj, comp_ret, comp_len, done_mask) = jax.lax.scan(_step, init_carry, scan_keys)
        self.env_state = env_state
        self.obs = last_obs
        self.dones = traj.done[-1]
        # Persist updated RMS for the next explore_env / update_net call.
        self.actor_run_stats = actor_rs
        self.critic_run_stats = critic_rs

        comp_ret_np = np.asarray(comp_ret)
        comp_len_np = np.asarray(comp_len)
        done_np = np.asarray(done_mask).astype(bool)
        if done_np.any():
            self.return_tracker.update(comp_ret_np[done_np])
            self.step_tracker.update(comp_len_np[done_np])

        # Bootstrap value uses the FROZEN run_stats from after rollout (no further update).
        last_value, _ = self.critic_net.apply(
            {'params': self.critic_state.params['params'], 'run_stats': self.critic_run_stats},
            last_obs, mutable=['run_stats'],
        )
        return (traj, last_value), int(timesteps) * int(self.cfg.num_envs)

    def update_net(self, data):
        traj, last_value = data
        gamma = float(self.cfg.algo.get("gamma", 0.99))
        gae_lambda = float(self.cfg.algo.get("gae_lambda", 0.95))
        clip = float(self.cfg.algo.get("clip_range", 0.2))
        ent_coef = float(self.cfg.algo.get("ent_coef", 0.0))
        vf_coef = float(self.cfg.algo.get("vf_coef", 0.5))
        n_epochs = int(self.cfg.algo.get("n_epochs", 10))
        n_minibatches = int(self.cfg.algo.get("n_minibatches", 4))

        advantages, returns = _gae(traj, last_value, gamma, gae_lambda)

        T, N = traj.reward.shape
        flat = jax.tree_util.tree_map(lambda x: x.reshape(T * N, *x.shape[2:]),
                                      (traj.obs, traj.action, traj.log_prob,
                                       advantages, returns, traj.value))
        rng = self.split(1)
        (self.actor_state, self.critic_state, log_info) = _ppo_update(
            self.actor_state, self.critic_state,
            self.actor_net.apply, self.critic_net.apply,
            self.actor_run_stats, self.critic_run_stats,
            flat, rng, n_epochs=n_epochs, n_minibatches=n_minibatches,
            clip=clip, ent_coef=ent_coef, vf_coef=vf_coef,
        )
        out = {k: float(v) for k, v in log_info.items()}
        out["train/episode_length"] = float(self.step_tracker.mean())
        out.update(self.reward_log_info())
        return out

    # -- Serialization (mirrors loco_mujoco JaxRLAlgorithmBase) ------------
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
        if "actor_run_stats" in payload["agent_state"]:
            agent.actor_run_stats = flax.serialization.from_state_dict(
                agent.actor_run_stats, payload["agent_state"]["actor_run_stats"])
            agent.critic_run_stats = flax.serialization.from_state_dict(
                agent.critic_run_stats, payload["agent_state"]["critic_run_stats"])
        return agent


# -----------------------------------------------------------------------------
# Pure-JAX update kernel — jitted on first call.
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("actor_apply", "critic_apply",
                                    "n_epochs", "n_minibatches"))
def _ppo_update(actor_state, critic_state, actor_apply, critic_apply,
                actor_run_stats, critic_run_stats,
                flat_data, rng, *, n_epochs, n_minibatches,
                clip, ent_coef, vf_coef):
    obs, action, old_log_prob, advantages, returns, old_values = flat_data
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _loss(params_actor, params_critic, obs_b, act_b, old_lp, adv_b, ret_b, old_v):
        # apply with mutable=['run_stats'] so the network reads RMS, but the
        # returned `_updates` dict is DISCARDED — RMS only updates from rollout.
        # Mirrors loco_mujoco/algorithms/ppo_jax.py:_loss_fn.
        (mean, log_std), _ = actor_apply(
            {'params': params_actor['params'], 'run_stats': actor_run_stats},
            obs_b, mutable=['run_stats'],
        )
        std = jnp.exp(log_std)
        log_prob = -0.5 * (((act_b - mean) / std) ** 2 + 2.0 * log_std
                           + jnp.log(2.0 * jnp.pi)).sum(axis=-1)
        ratio = jnp.exp(log_prob - old_lp)
        loss_pg = -jnp.minimum(ratio * adv_b,
                                jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * adv_b).mean()
        # Value loss with PPO-style clipping.
        v, _ = critic_apply(
            {'params': params_critic['params'], 'run_stats': critic_run_stats},
            obs_b, mutable=['run_stats'],
        )
        v_clipped = old_v + jnp.clip(v - old_v, -clip, clip)
        loss_v_unclipped = jnp.square(v - ret_b)
        loss_v_clipped   = jnp.square(v_clipped - ret_b)
        loss_v = 0.5 * jnp.maximum(loss_v_unclipped, loss_v_clipped).mean()
        entropy = (log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(axis=-1).mean()
        loss = loss_pg + vf_coef * loss_v - ent_coef * entropy
        approx_kl = ((ratio - 1.0) - (log_prob - old_lp)).mean()
        clip_frac = (jnp.abs(ratio - 1.0) > clip).astype(jnp.float32).mean()
        return loss, (loss_pg, loss_v, entropy, approx_kl, clip_frac)

    grad_fn = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)

    def _epoch(carry, key):
        actor_state, critic_state = carry
        n = obs.shape[0]
        idx = jax.random.permutation(key, n)
        mb_size = n // n_minibatches

        def _mb(carry_inner, mb_idx):
            actor_state, critic_state = carry_inner
            sel = jax.lax.dynamic_slice(idx, (mb_idx * mb_size,), (mb_size,))
            (loss, (lp, lv, ent, kl, cf)), (g_a, g_c) = grad_fn(
                actor_state.params, critic_state.params,
                obs[sel], action[sel], old_log_prob[sel],
                advantages[sel], returns[sel], old_values[sel],
            )
            actor_state = actor_state.apply_gradients(grads=g_a)
            critic_state = critic_state.apply_gradients(grads=g_c)
            return (actor_state, critic_state), (lp, lv, ent, kl, cf)

        (actor_state, critic_state), stats = jax.lax.scan(
            _mb, (actor_state, critic_state), jnp.arange(n_minibatches),
        )
        return (actor_state, critic_state), stats

    epoch_keys = jax.random.split(rng, n_epochs)
    (actor_state, critic_state), all_stats = jax.lax.scan(
        _epoch, (actor_state, critic_state), epoch_keys,
    )
    pg, vl, ent, kl, cf = all_stats
    log_info = {
        "train/loss/policy": pg.mean(),
        "train/loss/value": vl.mean(),
        "train/entropy": ent.mean(),
        "train/approx_kl": kl.mean(),
        "train/clip_fraction": cf.mean(),
    }
    return actor_state, critic_state, log_info


def _gae(traj: _Transition, last_value, gamma: float, gae_lambda: float):
    """Compute GAE advantages + returns from a (T, N) transition batch."""
    def _step(carry, t):
        gae, next_val = carry
        delta = traj.reward[t] + gamma * next_val * (1.0 - traj.absorbing[t]) - traj.value[t]
        gae = delta + gamma * gae_lambda * (1.0 - traj.done[t]) * gae
        return (gae, traj.value[t]), gae

    T = traj.reward.shape[0]
    init = (jnp.zeros_like(last_value), last_value)
    _, advantages_rev = jax.lax.scan(_step, init, jnp.arange(T - 1, -1, -1))
    advantages = advantages_rev[::-1]
    returns = advantages + traj.value
    return advantages, returns


# -- Lightweight views so eval.py / render.py can call agent.actor.* -------
# Eval / render: use mutable=False (default) — RMS variables are read but never
# updated; apply returns the bare output (mean, log_std), not (output, updates).
# IMPORTANT: mutable=[] returns a (output, {}) tuple — DO NOT use it here, it
# would alias `mean` to the (mean, log_std) tuple and break downstream env.step.
class _ActorView:
    def __init__(self, agent):
        self.agent = agent

    def _vars(self):
        return {'params': self.agent.actor_state.params['params'],
                'run_stats': self.agent.actor_run_stats}

    def get_actions(self, obs, sample: bool = True):
        mean, log_std = self.agent.actor_net.apply(self._vars(), obs)
        if not sample:
            return mean
        key = self.agent.split(1)
        std = jnp.exp(log_std)
        return mean + jax.random.normal(key, mean.shape) * std

    def __call__(self, obs):
        return self.get_actions(obs, sample=False)


class _CriticView:
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, obs):
        return self.agent.critic_net.apply(
            {'params': self.agent.critic_state.params['params'],
             'run_stats': self.agent.critic_run_stats},
            obs)
