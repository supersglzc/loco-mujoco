"""
loco-mujoco — RL evaluation entry point (custom_jax).

Loads a pickle checkpoint (saved by train.py via utils.checkpoint.save_agent),
runs `n_episodes` deterministic rollouts on the SAME MJX vmap env shape used
by train.py, prints aggregate metrics + writes metrics.json next to the
checkpoint.

Example:
    python nautilus/scripts/rl/custom_jax/eval.py algo=ppo task=UnitreeH1 \\
        checkpoint=nautilus/outputs/<trial_id>/AgentPPO_saved.pkl \\
        n_episodes=10
"""
from __future__ import annotations

import json
import os
import sys
import time
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
from utils.common import load_class_from_path, set_random_seed  # noqa: E402
from utils.checkpoint import load_agent_dict  # noqa: E402
from env_wrapper import create_env  # noqa: E402


def _resolve_algo_class(name: str):
    target = "Agent" + name.upper() if not name.startswith("Agent") else name
    if target in alg_name_to_path:
        return load_class_from_path(target, alg_name_to_path[target])
    for cls_name, path in alg_name_to_path.items():
        if cls_name.lower() == target.lower():
            return load_class_from_path(cls_name, path)
    raise KeyError(f"Algorithm '{name}' not found.")


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
    cfg.num_envs = int(cfg.get("n_envs", cfg.get("num_envs", 1)))
    cfg.gpu_sim = True
    # Force RAW reward at eval: NormalizeVecReward replaces env.step's reward
    # with reward/sqrt(var); during training that scaling is consistent across
    # iters but during eval the wrapper starts with fresh stats and would
    # divide by an arbitrarily-small std as it warms up — the eval mean_return
    # would be incomparable across runs and meaningless. Always disable here.
    cfg.normalize_env = False

    task = cfg.get("task")
    ckpt_path = cfg.get("checkpoint")
    n_episodes = int(cfg.get("n_episodes", 10))
    if task is None:
        print("[error] task=<id> is required.", file=sys.stderr); sys.exit(2)
    if not ckpt_path:
        print("[error] checkpoint=<path> is required.", file=sys.stderr); sys.exit(2)
    ckpt = Path(ckpt_path)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not ckpt.exists():
        print(f"[error] checkpoint not found: {ckpt}", file=sys.stderr); sys.exit(2)
    set_random_seed(int(cfg.get("seed", 42)))

    env = create_env(cfg)
    payload = load_agent_dict(ckpt)
    agent_cls = _resolve_algo_class(str(cfg.algo.name))
    # Use the saved training cfg (max_grad_norm, hidden_dims, etc.) so the agent's
    # optimizer chain + network shapes match the checkpoint exactly. The CLI cfg
    # is only used for the eval-loop knobs (n_episodes, n_envs, task) — those are
    # carried by the env we already built above.
    agent = agent_cls.from_dict(payload, env=env)

    n = int(cfg.num_envs)
    # VecEnv.reset is vmap'd over rng_key axis 0; expects (N, 2) split keys.
    reset_rngs = jax.random.split(jax.random.PRNGKey(int(cfg.get("seed", 0))), n)
    obs, env_state = env.reset(reset_rngs)

    # ------------------------------------------------------------------
    # UNBIASED EVAL: run for `eval_total_steps`, aggregate ALL completions.
    #
    # The previous protocol stopped at the first `n_episodes` completions. With
    # vmap'd envs that have a wide goal-difficulty distribution
    # (e.g. GoalRandomRootVelocity in loco-mujoco), the FIRST completions are
    # systematically the FASTEST-FAILING — biasing mean_length downward.
    # Counting all completions over a fixed step budget matches what the
    # training-time step_tracker measures and is comparable to the train log.
    # ------------------------------------------------------------------
    eval_total_steps = int(cfg.get("eval_total_steps", max(2000, n_episodes * 200 // n)))
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    cur_returns = np.zeros(n, dtype=np.float32)
    cur_lengths = np.zeros(n, dtype=np.int64)
    start = time.time()
    for step in range(eval_total_steps):
        action = agent.actor.get_actions(obs, sample=False)
        obs, reward, absorbing, done, info, env_state = env.step(env_state, action)
        r_np = np.asarray(reward, dtype=np.float32).reshape(-1)
        d_np = np.asarray(done, dtype=bool).reshape(-1)
        cur_returns += r_np
        cur_lengths += 1
        for i in np.where(d_np)[0]:
            completed_returns.append(float(cur_returns[i]))
            completed_lengths.append(int(cur_lengths[i]))
            cur_returns[i] = 0.0
            cur_lengths[i] = 0

    # Drop the first quartile of completions to remove the fresh-reset
    # transient — those episodes are biased toward fast-failing goals
    # because they finish first while easy-goal episodes are still mid-run.
    if len(completed_returns) >= 16:
        burn_in = max(1, len(completed_returns) // 4)
        completed_returns = completed_returns[burn_in:]
        completed_lengths = completed_lengths[burn_in:]

    metrics = {
        "algo": str(cfg.algo.name),
        "task": str(task),
        "checkpoint": str(ckpt),
        "eval_total_steps": eval_total_steps,
        "n_envs": n,
        "n_episodes_completed": len(completed_returns),
        "mean_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "std_return":  float(np.std(completed_returns)) if completed_returns else 0.0,
        "mean_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
        "wall_time_sec": time.time() - start,
    }
    print(f"[eval] mean_return={metrics['mean_return']:.4f} ± {metrics['std_return']:.4f} "
          f"(n={metrics['n_episodes_completed']})", flush=True)
    (ckpt.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
