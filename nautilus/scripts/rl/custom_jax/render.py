"""
loco-mujoco — render a trained policy to video.mp4 (custom_jax).

Single env (n=1) on the MJX vmap path. Loads a pickle checkpoint, runs ONE
episode with the policy, and writes:

    <checkpoint-dir>/render.mp4
    <checkpoint-dir>/render_contact_sheet.jpg

Example:
    python nautilus/scripts/rl/custom_jax/render.py algo=ppo task=UnitreeH1 \\
        checkpoint=nautilus/outputs/<trial_id>/AgentPPO_saved.pkl
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
from env_wrapper import create_render_env  # noqa: E402


def _resolve_algo_class(name: str):
    target = "Agent" + name.upper() if not name.startswith("Agent") else name
    if target in alg_name_to_path:
        return load_class_from_path(target, alg_name_to_path[target])
    for cls_name, path in alg_name_to_path.items():
        if cls_name.lower() == target.lower():
            return load_class_from_path(cls_name, path)
    raise KeyError(f"Algorithm '{name}' not found.")


def _grab_frame(env, env_state):
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
            return np.asarray(out[0]).astype(np.uint8)
    return None


def _write_mp4(frames, path: Path, fps: int):
    if not frames:
        print("[render] no frames captured — skipping mp4", file=sys.stderr); return
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore[no-redef]
    imageio.mimsave(str(path), frames, fps=fps, codec="libx264", quality=8)


def _write_contact_sheet(frames, path: Path):
    if not frames:
        return
    try:
        from PIL import Image
    except ImportError:
        return
    n = len(frames)
    indices = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    sample = [Image.fromarray(frames[i]) for i in indices]
    h = max(im.height for im in sample)
    w = sum(im.width for im in sample)
    sheet = Image.new("RGB", (w, h))
    x = 0
    for im in sample:
        sheet.paste(im, (x, 0))
        x += im.width
    sheet.save(str(path), quality=85)


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
    cfg.gpu_sim = True

    task = cfg.get("task")
    ckpt_path = cfg.get("checkpoint")
    fps = int(cfg.get("fps", 30))
    max_steps = int(cfg.get("max_steps", 1000))
    if task is None:
        print("[error] task=<id> is required.", file=sys.stderr); sys.exit(2)
    if not ckpt_path:
        print("[error] checkpoint=<path> is required.", file=sys.stderr); sys.exit(2)
    ckpt = Path(ckpt_path)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not ckpt.exists():
        print(f"[error] checkpoint not found: {ckpt}", file=sys.stderr); sys.exit(2)

    env = create_render_env(cfg)
    payload = load_agent_dict(ckpt)
    agent_cls = _resolve_algo_class(str(cfg.algo.name))
    # Use the saved training cfg so the agent's optimizer + network shapes match
    # the checkpoint exactly. The CLI cfg only carries render-loop knobs.
    agent = agent_cls.from_dict(payload, env=env)

    n = 1
    # VecEnv.reset is vmap'd over rng_key axis 0; expects (N, 2) split keys.
    reset_rngs = jax.random.split(jax.random.PRNGKey(int(cfg.get("seed", 0))), n)
    obs, env_state = env.reset(reset_rngs)

    frames = []
    ep_return = 0.0
    for _ in range(max_steps):
        f = _grab_frame(env, env_state)
        if f is not None:
            frames.append(f)
        action = agent.actor.get_actions(obs, sample=False)
        obs, reward, absorbing, done, info, env_state = env.step(env_state, action)
        ep_return += float(np.asarray(reward).sum())
        if bool(np.asarray(done).any()):
            f = _grab_frame(env, env_state)
            if f is not None:
                frames.append(f)
            break

    out_dir = ckpt.parent
    _write_mp4(frames, out_dir / "render.mp4", fps=fps)
    _write_contact_sheet(frames, out_dir / "render_contact_sheet.jpg")
    print(f"[render] wrote {out_dir / 'render.mp4'} ({len(frames)} frames, return={ep_return:.4f})", flush=True)


if __name__ == "__main__":
    main()
