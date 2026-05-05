"""
loco-mujoco — random-action rollout (env sanity check).

Purpose
-------
Builds the benchmark env, runs N random-action steps, and verifies that
`env.step(...)` returns a finite numeric reward on every step. This is the
fastest end-to-end check that env-generator + benchmark-generator wired the
simulator correctly. NO trained policy is needed; NO video is produced.

Used as the L1 smoke tier by benchmark-generator and as the day-1 sanity
check a user can run after `bash nautilus/setup_uv.sh`.

Example
-------
    python scripts/run_random.py --task UnitreeH1 --n-steps 10
    python scripts/run_random.py --task UnitreeH1 --n-steps 100 --seed 0
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--task", default="UnitreeH1",
                   help="Task identifier (default: %(default)s)")
    p.add_argument("--n-steps", type=int, default=10,
                   help="Number of env.step() calls (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0,
                   help="env.reset(seed=...) when supported (default: %(default)s)")
    return p.parse_args()


def build_env(task: str):
    """Build the benchmark env with rendering disabled.

    loco-mujoco RL backend: the MuJoCo (CPU) env is built via `RLFactory.make`.
    No render flag is needed for L1 — rendering is a separate `env.render()`
    call that L1 never makes.
    """
    from loco_mujoco import RLFactory
    env = RLFactory.make(task)
    return env


def sample_action(env):
    """Random action expression — Gaussian sample of the action dim.

    loco-mujoco's action_space is a `Box` exposed via `env.info.action_space`;
    `np.random.randn(action_dim)` is the canonical random rollout used in
    `tests/test_task_factories.py`.
    """
    return np.random.randn(env.info.action_space.shape[0])


def main():
    args = parse_args()
    env = build_env(args.task)
    try:
        env.reset(seed=args.seed)
    except TypeError:
        env.reset()

    bad = 0
    for i in range(args.n_steps):
        out = env.step(sample_action(env))
        # Support 4-tuple (gym) and 5-tuple (gymnasium) and dm_env-style TimeStep.
        reward = out[1] if isinstance(out, tuple) else getattr(out, "reward", None)
        if reward is None or not np.isfinite(reward):
            print(f"step {i}: reward={reward!r} (NOT finite)", file=sys.stderr)
            bad += 1
        else:
            print(f"step {i}: reward={float(reward):.4f}")

    if bad:
        print(f"FAIL: {bad}/{args.n_steps} steps returned non-finite reward", file=sys.stderr)
        sys.exit(1)
    print(f"L1 OK: {args.n_steps} steps, all rewards finite")


if __name__ == "__main__":
    main()
