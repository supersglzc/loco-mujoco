"""
loco-mujoco — open a HEADED viewer for a trained RL policy (custom_jax).

Single env (n=1), CPU MuJoCo backend, GLFW window via env.render(). Loads a
pickle checkpoint, runs n_steps with the trained policy, displays each frame
in a live window. NO MP4 written — for video, use render.py instead.

Mirrors loco-mujoco's PPOJax.play_policy(use_mujoco=True) reference:
  https://github.com/robfiras/loco-mujoco/blob/main/loco_mujoco/algorithms/ppo_jax.py

Requires:
  - host has a display (DISPLAY set in the shell)
  - the env supports the CPU MuJoCo backend (drop the `Mjx` prefix from the task
    id when building); the JAX-trained checkpoint is portable across both
    backends because the network only sees the obs vector.

Example:
    python nautilus/scripts/rl/custom_jax/visualize.py algo=ppo task=UnitreeH1 \\
        checkpoint=nautilus/outputs/<trial_id>/AgentPPO_saved.pkl n_steps=2000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

import jax
import jax.numpy as jnp

REPO     = Path(__file__).resolve().parents[4]
NAUTILUS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(NAUTILUS))
sys.path.insert(0, str(REPO))

CUSTOM_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(CUSTOM_ROOT))

from algo import alg_name_to_path  # noqa: E402
from utils.common import load_class_from_path  # noqa: E402
from utils.checkpoint import load_agent_dict  # noqa: E402


def _resolve_algo_class(name: str):
    target = "Agent" + name.upper() if not name.startswith("Agent") else name
    if target in alg_name_to_path:
        return load_class_from_path(target, alg_name_to_path[target])
    for cls_name, path in alg_name_to_path.items():
        if cls_name.lower() == target.lower():
            return load_class_from_path(cls_name, path)
    raise KeyError(f"Algorithm '{name}' not found.")


def _build_cpu_env(task: str, env_params: dict):
    """Build the CPU MuJoCo backend (no Mjx prefix).

    For loco-mujoco: `RLFactory.make('UnitreeH1')` → CPU env with GLFW viewer.
    Strip the `Mjx` prefix from the task id if present so the CPU registry
    entry is hit instead of the JAX one.
    """
    from loco_mujoco import RLFactory
    cpu_task = task[3:] if task.startswith("Mjx") else task
    return RLFactory.make(cpu_task, **env_params)


def _resolve_env_params(cfg) -> dict:
    raw = None
    if "env_params" in cfg:
        raw = cfg.env_params
    elif "experiment" in cfg and "env_params" in cfg.experiment:
        raw = cfg.experiment.env_params
    if raw is None:
        return {}
    d = OmegaConf.to_container(raw, resolve=True) if isinstance(raw, DictConfig) else dict(raw)
    d.pop("env_name", None)
    return d


@hydra.main(version_base=None, config_path="../../../configs/rl", config_name="ppo")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    if "algo" not in cfg or not isinstance(cfg.get("algo"), DictConfig):
        algo_name = str(cfg.get("algo", "ppo"))
        cfg.algo = OmegaConf.create({
            "name": algo_name,
            "actor_lr": 3.0e-4, "critic_lr": 3.0e-4, "alpha_lr": 3.0e-4,
            "init_alpha": 1.0, "gamma": 0.99,
            "gae_lambda": 0.95, "clip_range": 0.2,
            "ent_coef": 0.0, "vf_coef": 0.5, "tau": 0.005,
            "tracker_len": 100, "explore_noise": 0.0,
            "policy_noise": 0.2, "noise_clip": 0.5, "policy_delay": 2,
            "nstep": 1, "reward_scale": 1.0,
            "hidden_dims": [256, 256], "activation": "tanh",
        })
    cfg.num_envs = 1
    cfg.gpu_sim = True   # the agent class itself still runs on GPU even though env is CPU

    task = cfg.get("task")
    ckpt_path = cfg.get("checkpoint")
    n_steps = int(cfg.get("n_steps", 2000))
    if task is None:
        print("[error] task=<id> is required.", file=sys.stderr); sys.exit(2)
    if not ckpt_path:
        print("[error] checkpoint=<path> is required.", file=sys.stderr); sys.exit(2)
    ckpt = Path(ckpt_path)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not ckpt.exists():
        print(f"[error] checkpoint not found: {ckpt}", file=sys.stderr); sys.exit(2)

    if not os.environ.get("DISPLAY"):
        print("[warn] DISPLAY is unset — the GLFW window will fail to open. "
              "If you're on SSH, use X11 forwarding (`ssh -X`) or run on a host with a display.",
              file=sys.stderr)

    env_params = _resolve_env_params(cfg)
    env = _build_cpu_env(task, env_params)

    # Build the agent against the same env so dims line up, then load checkpoint.
    payload = load_agent_dict(ckpt)
    agent_cls = _resolve_algo_class(str(cfg.algo.name))
    agent = agent_cls.from_dict(payload, env=env)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = np.atleast_2d(np.asarray(obs))

    print(f"[visualize] {n_steps} steps on CPU env '{task}'. Press the window's close button to stop early.",
          flush=True)
    ep_return = 0.0
    ep_len = 0
    completed = []
    for step in range(n_steps):
        action_jax = agent.actor.get_actions(jnp.asarray(obs), sample=False)
        action = np.asarray(action_jax).reshape(-1)
        out = env.step(action)
        if len(out) == 5:
            obs_next, reward, absorbing, done, info = out
        else:
            obs_next, reward, done, info = out
            absorbing = done
        ep_return += float(np.asarray(reward).sum())
        ep_len += 1
        try:
            env.render(record=False)
        except Exception as e:
            print(f"[visualize] env.render() failed: {e}", file=sys.stderr)
            break
        if bool(np.asarray(done).any()):
            completed.append((ep_return, ep_len))
            print(f"  episode {len(completed)}: return={ep_return:.3f} length={ep_len}", flush=True)
            ep_return = 0.0
            ep_len = 0
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
        else:
            obs = obs_next
        obs = np.atleast_2d(np.asarray(obs))

    try:
        env.stop()
    except Exception:
        pass
    if completed:
        rs = [r for r, _ in completed]
        ls = [l for _, l in completed]
        print(f"[visualize] done. {len(completed)} episodes: mean_return={np.mean(rs):.3f} ± {np.std(rs):.3f}, "
              f"mean_length={np.mean(ls):.1f}", flush=True)
    else:
        print(f"[visualize] done. no episode completed in {n_steps} steps "
              f"(in-progress return={ep_return:.3f}, length={ep_len})", flush=True)


if __name__ == "__main__":
    main()
