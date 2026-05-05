"""
loco-mujoco — RL training entry point (custom_jax).

JAX/Flax flavor of the trainer. Hydra-driven, single-process, single-GPU.
Picks an algorithm from algo/ via cfg.algo.name, builds an MJX-style vmap
env via env_wrapper.create_env(), and loops:

    explore_env(env, horizon_len) → trajectory_or_replay_data
    update_net(memory_or_traj)    → log_info_dict

Writes:
    nautilus/outputs/<algo>_<task>_<YYYYMMDD-HHMMSS>/
        <AgentClass>_saved.pkl       — pickle({agent_conf, agent_state})
        resolved_config.yaml         — Hydra config snapshot
        tb/                          — TensorBoard event files
        metrics.jsonl                — per-record (step, key, value) log
        curves/                      — PNG plots of training curves
        train.log                    — stdout/stderr mirror

W&B is wired automatically when cfg.wandb is set (use `wandb=null` to disable).

Example:
    python nautilus/scripts/rl/custom_jax/train.py algo=ppo task=UnitreeH1 max_step=1_000_000
    python nautilus/scripts/rl/custom_jax/train.py algo=sac task=UnitreeH1 max_step=5000 wandb=null
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from itertools import count
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# JAX memory hygiene — avoid eager 95% pre-allocation that breaks shared GPUs.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

import jax  # noqa: E402

REPO     = Path(__file__).resolve().parents[4]   # actual repo root
NAUTILUS = Path(__file__).resolve().parents[3]   # <repo>/nautilus
sys.path.insert(0, str(NAUTILUS))
sys.path.insert(0, str(REPO))

CUSTOM_ROOT = Path(__file__).resolve().parents[0]  # nautilus/scripts/rl/custom_jax
sys.path.insert(0, str(CUSTOM_ROOT))

from algo import alg_name_to_path  # noqa: E402
from utils.common import set_random_seed, load_class_from_path  # noqa: E402
from utils.checkpoint import save_agent  # noqa: E402

# DataLogger lives at <repo>/nautilus/utils/data_logger.py — same shadowing trick
# as custom_torch (custom_jax has its own utils/ package that takes precedence
# in sys.path, so we load DataLogger by absolute file path).
import importlib.util as _ilu  # noqa: E402
_dl_spec = _ilu.spec_from_file_location("_repo_data_logger", NAUTILUS / "utils" / "data_logger.py")
_dl_mod = _ilu.module_from_spec(_dl_spec)
_dl_spec.loader.exec_module(_dl_mod)
DataLogger = _dl_mod.DataLogger

from env_wrapper import create_env  # noqa: E402  (sibling import)


def _resolve_algo_class(name: str):
    target = "Agent" + name.upper() if not name.startswith("Agent") else name
    if target in alg_name_to_path:
        return load_class_from_path(target, alg_name_to_path[target])
    for cls_name, path in alg_name_to_path.items():
        if cls_name.lower() == target.lower():
            return load_class_from_path(cls_name, path)
    raise KeyError(f"Algorithm '{name}' not found. Available: {sorted(alg_name_to_path)}")


@hydra.main(version_base=None, config_path="../../../configs/rl", config_name="ppo")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # Synthesize cfg.algo namespace if the unified yaml put hyperparams at top level.
    if "algo" not in cfg or not isinstance(cfg.get("algo"), DictConfig):
        algo_name = str(cfg.get("algo", "ppo"))
        cfg.algo = OmegaConf.create({
            "name": algo_name,
            "actor_lr": cfg.get("learning_rate", 3.0e-4),
            "critic_lr": cfg.get("critic_learning_rate", cfg.get("learning_rate", 3.0e-4)),
            "alpha_lr": cfg.get("alpha_lr", 3.0e-4),
            "init_alpha": cfg.get("init_alpha", 1.0),
            "gamma": cfg.get("gamma", 0.99),
            "gae_lambda": cfg.get("gae_lambda", 0.95),
            "clip_range": cfg.get("clip_range", 0.2),
            "ent_coef": cfg.get("ent_coef", 0.0),
            "vf_coef": cfg.get("vf_coef", 0.5),
            "tau": cfg.get("tau", 0.005),
            "batch_size": cfg.get("batch_size", 256),
            "update_times": cfg.get("update_times", 1),
            "warm_up": cfg.get("warm_up", 1000),
            "horizon_len": cfg.get("n_steps", 16),
            "memory_size": cfg.get("buffer_size", 1_000_000),
            "n_epochs": cfg.get("n_epochs", 10),
            "n_minibatches": cfg.get("n_minibatches", 4),
            "max_grad_norm": cfg.get("max_grad_norm", 0.5),
            "weight_decay": cfg.get("weight_decay", 0.0),
            "explore_noise": cfg.get("explore_noise", 0.1),
            "policy_noise": cfg.get("policy_noise", 0.2),
            "noise_clip": cfg.get("noise_clip", 0.5),
            "policy_delay": cfg.get("policy_delay", 2),
            "nstep": cfg.get("nstep", 1),
            "reward_scale": cfg.get("reward_scale", 1.0),
            "tracker_len": cfg.get("tracker_len", 100),
            "hidden_dims": cfg.get("hidden_dims", [256, 256]),
            "activation": cfg.get("activation", "tanh"),
            # Network knobs the agent classes read via cfg.algo.get(...).
            # If the user's yaml puts them at top level (which our default
            # parallel.yaml does), forward them into cfg.algo so the agent
            # picks them up instead of falling back to the hard-coded default.
            "init_log_std":   cfg.get("init_log_std", 0.0),
            "use_obs_rms":    cfg.get("use_obs_rms", True),
            "learnable_std":  cfg.get("learnable_std", True),
            "obs_norm":       cfg.get("obs_norm", True),
            "value_norm":     cfg.get("value_norm", True),
            "handle_timeout": cfg.get("handle_timeout", True),
            "no_tgt_actor":   cfg.get("no_tgt_actor", False),
            "use_gae":        cfg.get("use_gae", True),
        })

    cfg.num_envs = int(cfg.get("n_envs", cfg.get("num_envs", 8)))
    cfg.gpu_sim = True  # custom_jax always uses an MJX vmap'd env

    task = cfg.get("task")
    if task is None:
        print("[error] task=<id> is required (Hydra override).", file=sys.stderr)
        sys.exit(2)
    set_random_seed(int(cfg.get("seed", 42)))

    env = create_env(cfg)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_slug = str(task).replace('/', '_')
    seed = int(cfg.get("seed", 42))
    algo_slug = str(cfg.algo.name).lower()
    # Filesystem dir keeps the timestamp so parallel reruns don't collide.
    trial_id = f"{algo_slug}_{task_slug}_seed{seed}_{ts}"
    log_dir = NAUTILUS / "outputs" / trial_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "curves").mkdir(parents=True, exist_ok=True)

    # W&B run name drops the timestamp so the algo+seed pair is the readable label.
    wandb_run_name = f"{algo_slug}_{task_slug}_seed{seed}"

    use_wandb = bool(cfg.get("wandb")) and cfg.get("wandb") not in ("null", "None", "")
    dl = DataLogger(
        log_dir=str(log_dir),
        log_tb=True,
        log_wandb=use_wandb,
        project=str(cfg.wandb) if use_wandb else None,
        run_name=wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=True) if use_wandb else None,
    )
    dl.log_text("config", OmegaConf.to_yaml(cfg, resolve=True), step=0)

    metrics_log: list[dict] = []
    metrics_jsonl = log_dir / "metrics.jsonl"
    _jsonl_fp = metrics_jsonl.open("w", buffering=1)

    def _record(key: str, value: float, step: int):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(f):
            return
        dl.record(key, f, step=step)
        rec = {"step": int(step), "key": key, "value": f}
        metrics_log.append(rec)
        _jsonl_fp.write(json.dumps(rec) + "\n")

    agent_cls = _resolve_algo_class(str(cfg.algo.name))
    agent = agent_cls(env=env, cfg=cfg)

    is_off_policy = str(cfg.algo.name).lower() != "ppo"
    memory = None
    if is_off_policy:
        from replay.simple_replay import ReplayBuffer
        memory = ReplayBuffer(
            capacity=int(cfg.algo.memory_size),
            obs_dim=agent.obs_dim, action_dim=agent.action_dim,
        )

    agent.reset_agent()
    global_steps = 0
    if is_off_policy:
        traj, used = agent.explore_env(env, int(cfg.algo.warm_up), random=True)
        if traj is not None:
            memory.add_to_buffer(traj)
        global_steps += int(used)

    max_step = int(cfg.get("total_timesteps", 1_000_000))
    log_interval = int(cfg.get("log_interval", int(cfg.algo.horizon_len) * int(cfg.num_envs)))
    print(f"[train] algo={cfg.algo.name} task={task} num_envs={cfg.num_envs} max_step={max_step} "
          f"log_interval={log_interval} wandb={'on' if use_wandb else 'off'}", flush=True)

    next_log_at = log_interval
    try:
        for it in count():
            traj, used = agent.explore_env(env, int(cfg.algo.horizon_len), random=False)
            global_steps += int(used)
            if is_off_policy:
                if traj is not None:
                    memory.add_to_buffer(traj)
                log_info = agent.update_net(memory)
            else:
                log_info = agent.update_net(traj)

            if global_steps >= next_log_at:
                for k, v in (log_info or {}).items():
                    _record(k, v, global_steps)
                print(f"[train] iter={it} steps={global_steps}/{max_step} "
                      f"return={log_info.get('reward/total/episodic_return_mean', 0.0):.3f}", flush=True)
                next_log_at = ((global_steps // log_interval) + 1) * log_interval

            if global_steps >= max_step:
                break

        ckpt = save_agent(log_dir, agent)
        dl.log_text("final/checkpoint_path", str(ckpt), step=global_steps)
        print(f"[train] saved checkpoint to {ckpt}", flush=True)
        if bool(cfg.get("record_video", True)):
            _render_and_log_video(agent, cfg, log_dir, global_steps, use_wandb,
                                  fps=int(cfg.get("render_fps", 30)),
                                  max_steps=int(cfg.get("render_max_steps", 1000)))
    finally:
        _jsonl_fp.close()
        dl.close()
        (log_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
        _plot_curves(metrics_log, log_dir / "curves", title_prefix=trial_id)


def _render_and_log_video(agent, cfg: DictConfig, log_dir: Path, step: int,
                          use_wandb: bool, fps: int = 30, max_steps: int = 1000):
    """Run a deterministic rollout with the trained custom_jax agent and write
    `<log_dir>/render.mp4`. If `use_wandb`, also log the video to W&B."""
    import numpy as np
    import jax.numpy as jnp
    from omegaconf import OmegaConf as _OC
    render_cfg = _OC.create(_OC.to_container(cfg, resolve=True))
    render_cfg.num_envs = 1
    try:
        from env_wrapper import create_render_env
        env = create_render_env(render_cfg)
    except Exception as e:
        print(f"[render] could not build render env: {e}", file=sys.stderr)
        return None

    # Reset on the render env (single env). VecEnv.reset is vmap'd over rng_key
    # axis 0 → expects (N, 2) split PRNGKeys; no env_id kwarg.
    n = 1
    reset_rngs = jax.random.split(jax.random.PRNGKey(int(cfg.get("seed", 0))), n)
    obs, env_state = env.reset(reset_rngs)

    frames = []
    ep_return = 0.0
    for _ in range(max_steps):
        # Try to grab a frame via env.mjx_render (loco-mujoco) or env.render().
        f = _grab_frame(env, env_state)
        if f is not None:
            frames.append(f)
        action = agent.actor.get_actions(obs, sample=False)
        obs, reward, absorbing, done, info, env_state = env.step(env_state, action)
        ep_return += float(np.asarray(reward).sum())
        if bool(np.asarray(done).any()):
            break

    if not frames:
        print("[render] no frames captured — skipping mp4 write", file=sys.stderr)
        return None
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore[no-redef]
    mp4 = log_dir / "render.mp4"
    imageio.mimsave(str(mp4), frames, fps=fps, codec="libx264", quality=8)
    print(f"[render] wrote {mp4} ({len(frames)} frames, return={ep_return:.4f})", flush=True)
    if use_wandb:
        try:
            import wandb
            wandb.log({"render/video": wandb.Video(str(mp4), fps=fps, format="mp4")}, step=step)
        except Exception as e:
            print(f"[render] wandb video log failed: {e}", file=sys.stderr)
    return mp4


def _grab_frame(env, env_state):
    """Try MJX render first, then a generic env.render()."""
    import numpy as np
    for fn_name in ("mjx_render", "render"):
        fn = getattr(env, fn_name, None)
        if fn is None:
            continue
        try:
            out = fn(env_state) if fn_name == "mjx_render" else fn()
        except Exception:
            continue
        if isinstance(out, np.ndarray):
            return out.astype(np.uint8)
        if isinstance(out, (list, tuple)) and out:
            arr = out[0]
            if hasattr(arr, "shape"):
                return np.asarray(arr).astype(np.uint8)
    return None


def _plot_curves(metrics: list[dict], out_dir: Path, title_prefix: str = "") -> None:
    if not metrics:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key: dict[str, list[tuple[int, float]]] = {}
    for r in metrics:
        by_key.setdefault(r["key"], []).append((r["step"], r["value"]))
    written = 0
    for key, pts in by_key.items():
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, linewidth=1.2, marker="o" if len(pts) <= 3 else None, markersize=4)
        ax.set_xlabel("step")
        ax.set_ylabel(key)
        ax.set_title(f"{title_prefix}\n{key}" if title_prefix else key)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = key.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"{safe}.png", dpi=110)
        plt.close(fig)
        written += 1
    print(f"[plot] wrote {written}/{len(by_key)} curves to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
