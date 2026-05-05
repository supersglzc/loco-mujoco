"""
loco-mujoco — env factory for custom_jax train/eval/render.

REQUIRES vmap rollout: the env must expose the MJX-style contract
    env.reset(reset_keys)              -> (obs, env_state)
    env.step(env_state, action)        -> (obs, reward, absorbing, done, info, env_state)

This is the contract loco-mujoco's RLFactory + LogWrapper + VecEnv produces.
For other JAX/MJX benchmarks that follow the same convention this works as-is;
benchmarks WITHOUT this contract (gymnasium-style step/reset) cannot use
custom_jax — fall back to custom_torch instead.

Mirrors the upstream loco-mujoco PPOJax._wrap_env pattern (see
loco_mujoco/algorithms/ppo_jax.py), including:

  - env_params from the cfg → kwargs passed to RLFactory.make()
    (lets you override `reward_type`, `goal_type`, `reward_params`,
    `terminal_state_type`, `horizon`, `headless`, etc. without touching
    Python code; matches loco-mujoco/examples/training_examples/jax_rl/conf.yaml)
  - NormalizeVecReward(env, gamma) when cfg.normalize_env is True
    (rescales returns to ~unit variance — keeps the critic loss stable)

Exposes:
    create_env(cfg)         — vmap'd env for train + eval (num_envs from cfg)
    create_render_env(cfg)  — single env for render.py (n=1)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO     = Path(__file__).resolve().parents[4]   # actual repo root
NAUTILUS = Path(__file__).resolve().parents[3]   # <repo>/nautilus
sys.path.insert(0, str(NAUTILUS))
sys.path.insert(0, str(REPO))


def _resolve_env_params(cfg) -> dict:
    """Extract env_params from cfg (top-level OR cfg.experiment.env_params).

    Returns a plain dict suitable for `**kwargs` expansion into
    `RLFactory.make(...)`. The dict's `env_name` key is removed since it's
    passed positionally as `task`.
    """
    from omegaconf import DictConfig, OmegaConf
    raw = None
    if "env_params" in cfg:
        raw = cfg.env_params
    elif "experiment" in cfg and isinstance(cfg.experiment, DictConfig) and "env_params" in cfg.experiment:
        raw = cfg.experiment.env_params
    if raw is None:
        return {}
    d = OmegaConf.to_container(raw, resolve=True) if isinstance(raw, DictConfig) else dict(raw)
    d.pop("env_name", None)  # passed positionally, not as kwarg
    return d


def _build_loco_mujoco_env(task: str, env_params: dict, normalize_env: bool, gamma: float):
    """Build a loco-mujoco MJX env wrapped with LogWrapper + VecEnv (+ optional reward norm).

    Mirrors the canonical wrapper stack from loco_mujoco.algorithms.ppo_jax
    (see PPOJax._wrap_env): LogWrapper for episode-return / length tracking,
    VecEnv for the vmap'd reset/step contract, and NormalizeVecReward when
    `normalize_env` is True.

    loco-mujoco maintains TWO parallel registries:
      - `<Name>`     → CPU-only Mujoco backend (no `_first_data`, no mjx_reset)
      - `Mjx<Name>`  → JAX/MJX backend (what custom_jax requires)
    Auto-prepend the `Mjx` prefix if the caller passed the CPU name and the
    Mjx variant exists. Pass-through if it's already prefixed.
    """
    from loco_mujoco import RLFactory
    from loco_mujoco.core.wrappers import LogWrapper, VecEnv
    from loco_mujoco.environments.base import LocoEnv
    if not task.startswith("Mjx"):
        candidate = f"Mjx{task}"
        if candidate in LocoEnv.registered_envs:
            task = candidate
    env = RLFactory.make(task, **env_params)
    env = LogWrapper(env)
    env = VecEnv(env)
    if normalize_env:
        from loco_mujoco.core.wrappers import NormalizeVecReward
        env = NormalizeVecReward(env, gamma)
    return env


def _build_generic_mjx_env(task: str, env_params: dict, normalize_env: bool, gamma: float):
    """Hook for other JAX/MJX benchmarks. Override per repo as needed."""
    raise NotImplementedError(
        f"custom_jax requires a vmap'd MJX-style env for task '{task}'. "
        "If this is loco-mujoco the auto-detect path should have triggered. "
        "Otherwise wire your benchmark here in env_wrapper._build_generic_mjx_env."
    )


def create_env(cfg):
    """Build the training env. Branches by import availability + env_params from cfg."""
    task = str(cfg.get("task"))
    env_params = _resolve_env_params(cfg)
    normalize_env = bool(cfg.get("normalize_env", False))
    gamma = float(cfg.get("gamma", cfg.get("algo", {}).get("gamma", 0.99)) if hasattr(cfg, "get") else 0.99)
    try:
        import loco_mujoco  # noqa: F401
        env = _build_loco_mujoco_env(task, env_params, normalize_env, gamma)
    except ImportError:
        env = _build_generic_mjx_env(task, env_params, normalize_env, gamma)
    _assert_mjx_contract(env, task)
    return env


def create_render_env(cfg):
    """Build a render env (n=1, same MJX contract). Used by render.py.

    For loco-mujoco the same VecEnv works with num_envs=1; for other benchmarks
    you may need a render-mode-enabled variant. Default = same as create_env
    with num_envs forced to 1 inside cfg. NormalizeVecReward is intentionally
    DISABLED in render mode — we want raw rewards in the video for diagnostics.
    """
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.num_envs = 1
    cfg.normalize_env = False
    return create_env(cfg)


def _assert_mjx_contract(env, task: str) -> None:
    """Fail fast if the env doesn't honor the MJX vmap contract."""
    if not (hasattr(env, "reset") and hasattr(env, "step")):
        raise TypeError(f"env for task '{task}' is missing reset/step methods")
    # Other duck-typed checks (mdp_info / info) happen at agent init time.
