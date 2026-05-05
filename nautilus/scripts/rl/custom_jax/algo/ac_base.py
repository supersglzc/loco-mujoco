"""AgentBase — shared bookkeeping for SAC / TD3 / PPO (JAX/Flax port).

Mirrors custom_torch/algo/ac_base.py but stripped of torch deps. Holds the
host-side trackers (return / length / per-reward-term), and centralizes
optax optimizer construction. Each algorithm subclass (PPO / SAC / TD3) owns
its own Flax modules + TrainState(s) and update kernels.

Contract honored by scripts/{train,eval,render}.py.template:
    agent = AgentXXX(env=env, cfg=cfg)
    agent.reset_agent() -> obs (jnp)
    agent.explore_env(env, timesteps, random=False) -> (data, total_steps)
    agent.update_net(data) -> log_dict
    agent.actor_params           — for serialize / eval
    agent.serialize() / from_dict(payload, env, cfg) — for checkpointing
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf.dictconfig import DictConfig

from utils.common import Tracker
from utils.jax_util import RewardTermTracker


@dataclass
class AgentBase:
    env: Any
    cfg: DictConfig
    obs_dim: int = None
    action_dim: int = None
    rng: Any = None  # jax PRNG key

    def __post_init__(self):
        # Resolve observation/action spaces. Two conventions are supported:
        #   (a) gymnasium-style: env.observation_space / env.action_space
        #   (b) loco-mujoco-style: env.info.observation_space / .action_space
        #       (LogWrapper + VecEnv proxy `info` → MDPInfo with .observation_space)
        if self.obs_dim is None:
            obs_space = (getattr(self.env, "observation_space", None)
                         or self.env.info.observation_space)
            shape = obs_space.shape
            self.obs_dim = shape if len(shape) > 1 else int(shape[0])
        if self.action_dim is None:
            act_space = (getattr(self.env, "action_space", None)
                         or self.env.info.action_space)
            self.action_dim = int(act_space.shape[-1])

        if self.rng is None:
            seed = int(self.cfg.get("seed", 42))
            self.rng = jax.random.PRNGKey(seed)

        n = int(self.cfg.num_envs)
        tlen = int(self.cfg.algo.get("tracker_len", 100))
        self.return_tracker = Tracker(tlen)
        self.success_tracker = Tracker(tlen)
        self.step_tracker = Tracker(tlen)
        self.term_tracker = RewardTermTracker(num_envs=n, tracker_len=tlen)
        self.current_returns = np.zeros(n, dtype=np.float32)
        self.current_lengths = np.zeros(n, dtype=np.float32)
        self.obs = None
        self.dones = None

    # -- PRNG helpers -------------------------------------------------------
    def split(self, n: int = 2):
        keys = jax.random.split(self.rng, n)
        self.rng = keys[0]
        return keys[1:] if n > 2 else keys[1]

    # -- Bookkeeping (host-side numpy) -------------------------------------
    def reset_agent(self):
        out = self.env.reset()
        if isinstance(out, tuple):
            obs = out[0]
        else:
            obs = out
        self.obs = jnp.asarray(np.asarray(obs), dtype=jnp.float32)
        n = int(self.cfg.num_envs)
        self.dones = jnp.zeros(n, dtype=jnp.float32)
        self.current_returns = np.zeros(n, dtype=np.float32)
        self.current_lengths = np.zeros(n, dtype=np.float32)
        return self.obs

    def update_tracker(self, reward, done, info):
        r = np.asarray(reward, dtype=np.float32).reshape(-1)
        d = np.asarray(done, dtype=bool).reshape(-1)
        if r.shape[0] != self.current_returns.shape[0]:
            return  # shape mismatch — bail out silently
        self.current_returns += r
        self.current_lengths += 1.0
        done_idx = np.where(d)[0]
        if len(done_idx) > 0:
            self.return_tracker.update(self.current_returns[done_idx])
            self.step_tracker.update(self.current_lengths[done_idx])
            if isinstance(info, dict) and "success" in info:
                succ = info["success"]
                if hasattr(succ, "__len__"):
                    self.success_tracker.update(np.asarray(succ).reshape(-1)[done_idx])
            self.current_returns[done_idx] = 0.0
            self.current_lengths[done_idx] = 0.0

        if isinstance(info, dict) and "detailed_reward" in info:
            detailed = info["detailed_reward"]
            if isinstance(detailed, dict):
                self.term_tracker.update(detailed, d)

    def reward_log_info(self) -> dict:
        out = {"reward/total/episodic_return_mean": float(self.return_tracker.mean())}
        out.update(self.term_tracker.log_info())
        return out

    # -- Optimizer construction (mirrors loco_mujoco PPOJax._get_optimizer) -
    @staticmethod
    def make_optimizer(lr: float, max_grad_norm: float | None = None,
                       weight_decay: float = 0.0,
                       anneal_lr: bool = False) -> optax.GradientTransformation:
        chain = []
        if max_grad_norm is not None:
            chain.append(optax.clip_by_global_norm(float(max_grad_norm)))
        if anneal_lr:
            # inject_hyperparams allows runtime LR mutation (adaptive_lr path).
            chain.append(optax.inject_hyperparams(optax.adamw)(
                learning_rate=float(lr),
                weight_decay=float(weight_decay),
                eps=1e-5,
            ))
        else:
            chain.append(optax.adamw(
                learning_rate=float(lr),
                weight_decay=float(weight_decay),
                eps=1e-5,
            ))
        return optax.chain(*chain)
