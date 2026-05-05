"""
Flax MLP actors / critics for PPO, SAC, TD3.

Mirrors the custom_torch counterparts so the upstream contract is identical:
    DiagGaussianActor      — PPO policy (state-independent log-std)
    TanhDiagGaussianActor  — SAC policy (tanh-squashed Gaussian, state-dep log-std)
    TanhDeterministicActor — TD3 policy (tanh-bounded deterministic)
    MLPCritic              — PPO value head (scalar)
    DoubleQ                — twin Q networks (SAC / TD3)

All modules use orthogonal init √2 for hidden layers, 0.01 for the policy
output head, 1.0 for the critic — the convention in the loco-mujoco PPO
reference (algorithms/common/networks.py FullyConnectedNet).
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal


def _activation(name: str):
    return getattr(nn, name)


class MLP(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    output_dim: int = 1
    activation: str = "tanh"
    output_init_scale: float = 0.01  # 0.01 for policy heads, 1.0 for value heads

    @nn.compact
    def __call__(self, x):
        act = _activation(self.activation)
        for h in self.hidden_dims:
            x = nn.Dense(h, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = act(x)
        x = nn.Dense(self.output_dim,
                     kernel_init=orthogonal(self.output_init_scale),
                     bias_init=constant(0.0))(x)
        return x


class RunningMeanStd(nn.Module):
    """Welford-style running mean / variance, normalizes input by current stats.

    Mirrors loco_mujoco/algorithms/common/networks.py:RunningMeanStd.

    Variables live in the 'run_stats' collection. When apply is called with
    `mutable=['run_stats']`, the stats update from the current batch (Welford
    parallel-streams algorithm) and the FORWARD uses the post-update stats.
    When called with `mutable=[]` (or any list not containing 'run_stats'),
    the stats are read-only and the forward uses the existing values.

    Use:
        net.init(rng, dummy_x)                                     # creates {'params':..., 'run_stats':...}
        out = net.apply({'params': p, 'run_stats': rs}, x,
                        mutable=['run_stats'])                     # tuple: (out, updates)
        out, _ = net.apply({'params': p, 'run_stats': rs}, x,
                           mutable=[])                             # forward only, no update
    """

    @nn.compact
    def __call__(self, x):
        # Preserve the input rank so we don't accidentally strip batch=1 dims
        # at the output. The reference loco-mujoco RMS does an unconditional
        # squeeze which breaks downstream concat with action vectors when the
        # batch happens to be 1 (e.g. during init or n=1 eval).
        was_1d = (x.ndim == 1)
        x_2d = jnp.atleast_2d(x)
        mean = self.variable('run_stats', 'mean', lambda: jnp.zeros(x_2d.shape[-1]))
        var = self.variable('run_stats', 'var', lambda: jnp.ones(x_2d.shape[-1]))
        count = self.variable('run_stats', 'count', lambda: jnp.array(1e-6))
        batch_mean = jnp.mean(x_2d, axis=0)
        batch_var = jnp.var(x_2d, axis=0) + 1e-6
        batch_count = x_2d.shape[0]
        if self.is_mutable_collection('run_stats'):
            updated_count = count.value + batch_count
            delta = batch_mean - mean.value
            new_mean = mean.value + delta * batch_count / updated_count
            m_a = var.value * count.value
            m_b = batch_var * batch_count
            M2 = m_a + m_b + jnp.square(delta) * count.value * batch_count / updated_count
            new_var = M2 / updated_count
            mean.value = new_mean
            var.value = new_var
            count.value = updated_count
        else:
            new_mean = mean.value
            new_var = var.value
        normalized = (x_2d - new_mean) / jnp.sqrt(new_var + 1e-8)
        # Restore original rank: 1D input → 1D output, 2D input → 2D output.
        return normalized[0] if was_1d else normalized


# -----------------------------------------------------------------------------
# Actors
# -----------------------------------------------------------------------------
class DiagGaussianActor(nn.Module):
    """PPO actor — diagonal Gaussian, state-independent log-std (a Flax param).

    When `use_obs_rms=True` (default), an in-network `RunningMeanStd` layer
    normalizes inputs. The 'run_stats' variable collection updates only when
    apply is called with `mutable=['run_stats']`.
    """
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "tanh"
    init_log_std: float = 0.0
    use_obs_rms: bool = False

    @nn.compact
    def __call__(self, x):
        if self.use_obs_rms:
            x = RunningMeanStd()(x)
        mean = MLP(self.hidden_dims, self.action_dim, self.activation,
                   output_init_scale=0.01)(x)
        log_std = self.param(
            "log_std",
            nn.initializers.constant(self.init_log_std),
            (self.action_dim,),
        )
        log_std = jnp.broadcast_to(log_std, mean.shape)
        return mean, log_std

    def sample_and_logprob(self, params, rng, x):
        mean, log_std = self.apply(params, x)
        std = jnp.exp(log_std)
        eps = jax.random.normal(rng, mean.shape)
        action = mean + eps * std
        # Diagonal Gaussian log-prob (sum over action dims).
        logp = -0.5 * (((action - mean) / std) ** 2 + 2.0 * log_std
                       + jnp.log(2.0 * jnp.pi)).sum(axis=-1)
        return action, logp

    def log_prob(self, params, x, action):
        mean, log_std = self.apply(params, x)
        std = jnp.exp(log_std)
        return -0.5 * (((action - mean) / std) ** 2 + 2.0 * log_std
                       + jnp.log(2.0 * jnp.pi)).sum(axis=-1)

    def entropy(self, params, x):
        _, log_std = self.apply(params, x)
        return (log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(axis=-1)


class TanhDiagGaussianActor(nn.Module):
    """SAC actor — tanh-squashed Gaussian with state-dependent log-std.

    `use_obs_rms=True` adds an in-network RunningMeanStd input layer.
    """
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    use_obs_rms: bool = False

    @nn.compact
    def __call__(self, x):
        if self.use_obs_rms:
            x = RunningMeanStd()(x)
        # output = (mean, raw_log_std) of shape (..., 2 * action_dim)
        out = MLP(self.hidden_dims, 2 * self.action_dim, self.activation,
                  output_init_scale=0.01)(x)
        mean, log_std = jnp.split(out, 2, axis=-1)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample_and_logprob(self, params, rng, x):
        mean, log_std = self.apply(params, x)
        std = jnp.exp(log_std)
        eps = jax.random.normal(rng, mean.shape)
        u = mean + eps * std
        action = jnp.tanh(u)
        # log_prob with tanh correction (jacobian of tanh).
        gaussian_logp = -0.5 * (((u - mean) / std) ** 2 + 2.0 * log_std
                                + jnp.log(2.0 * jnp.pi)).sum(axis=-1)
        # log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))   [numerical-stable]
        correction = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
        logp = gaussian_logp - correction.sum(axis=-1)
        return action, logp

    def deterministic_action(self, x):
        # Designed to be called via `module.apply(params, x, method='deterministic_action')`.
        # The bound-method dance is what makes Flax thread the params through the
        # underlying parameterized layers — DO NOT call self.apply(params, x) again
        # here (we're already inside an apply context).
        mean, _ = self(x)
        return jnp.tanh(mean)


class TanhDeterministicActor(nn.Module):
    """TD3 actor — deterministic, tanh-bounded. Optional in-network RMS."""
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"
    use_obs_rms: bool = False

    @nn.compact
    def __call__(self, x):
        if self.use_obs_rms:
            x = RunningMeanStd()(x)
        x = MLP(self.hidden_dims, self.action_dim, self.activation,
                output_init_scale=0.01)(x)
        return jnp.tanh(x)


# -----------------------------------------------------------------------------
# Critics
# -----------------------------------------------------------------------------
class MLPCritic(nn.Module):
    """PPO critic — single scalar value head.

    When `use_obs_rms=True` (default), an in-network `RunningMeanStd` layer
    normalizes inputs. Maintained INDEPENDENTLY from the actor's RMS so the
    critic's stats track the obs distribution it has actually seen (in PPO
    that's the same as the actor's, but the design generalizes).
    """
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "tanh"
    use_obs_rms: bool = False

    @nn.compact
    def __call__(self, x):
        if self.use_obs_rms:
            x = RunningMeanStd()(x)
        v = MLP(self.hidden_dims, 1, self.activation, output_init_scale=1.0)(x)
        return jnp.squeeze(v, axis=-1)


class DoubleQ(nn.Module):
    """Twin Q networks (SAC / TD3). Concatenates obs + action then runs two MLPs.

    `use_obs_rms=True` normalizes ONLY the obs (not the action — actions are
    already in [-1, 1] from tanh-squashed policies).
    """
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"
    use_obs_rms: bool = False

    @nn.compact
    def __call__(self, obs, action):
        if self.use_obs_rms:
            obs = RunningMeanStd()(obs)
        x = jnp.concatenate([obs, action], axis=-1)
        q1 = MLP(self.hidden_dims, 1, self.activation, output_init_scale=1.0,
                 name="q1")(x)
        q2 = MLP(self.hidden_dims, 1, self.activation, output_init_scale=1.0,
                 name="q2")(x)
        return jnp.squeeze(q1, axis=-1), jnp.squeeze(q2, axis=-1)
