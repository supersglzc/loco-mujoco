"""
JAX-side utilities — PRNG splitter, RewardTermTracker, optax LR-anneal helper.

RewardTermTracker mirrors the per-term-reward behaviour in
custom_torch/algo/ac_base.py (update_tracker block). It runs on the host
(numpy / Python) — train.py calls it after each env.step batch, OUTSIDE the
jitted update kernel, so jnp ↔ np conversion is free.
"""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from utils.common import Tracker


def split_key(rng, n: int = 2):
    """Tiny convenience wrapper. `keys = split_key(rng, 4)` returns 4 PRNG keys."""
    return jax.random.split(rng, n)


class RewardTermTracker:
    """Per-term episodic-return tracker.

    Maintains, for each reward term, a running per-env accumulator that resets
    when an env's `done` is True, plus a Tracker over completed-episode totals.

    Use:
        tracker = RewardTermTracker(num_envs=N, tracker_len=100)
        # called after each env.step(...):
        tracker.update(reward_terms_dict, dones_array)
        # at log time:
        log_info = tracker.log_info()  # {"reward/<term>/episodic_return_mean": ...}
    """

    def __init__(self, num_envs: int, tracker_len: int = 100):
        self.num_envs = int(num_envs)
        self.tracker_len = int(tracker_len)
        self._cum: Dict[str, np.ndarray] = {}
        self._trackers: Dict[str, Tracker] = {}

    def _ensure(self, term: str) -> None:
        if term not in self._cum:
            self._cum[term] = np.zeros(self.num_envs, dtype=np.float32)
            self._trackers[term] = Tracker(self.tracker_len)

    def update(self, reward_terms: Dict[str, jnp.ndarray] | None, dones: jnp.ndarray | np.ndarray) -> None:
        if not reward_terms:
            return
        d = np.asarray(dones, dtype=bool).reshape(-1)
        if d.shape[0] != self.num_envs:
            return
        done_idx = np.where(d)[0]
        for term, val in reward_terms.items():
            if term.startswith("_"):
                continue  # gymnasium-1.x VectorEnv mask sentinels
            self._ensure(term)
            v = np.asarray(val, dtype=np.float32).reshape(-1)
            if v.shape[0] != self.num_envs:
                continue
            self._cum[term] += v
            if len(done_idx) > 0:
                self._trackers[term].update(self._cum[term][done_idx])
                self._cum[term][done_idx] = 0.0

    def log_info(self) -> dict:
        return {
            f"reward/{k}/episodic_return_mean": float(t.mean())
            for k, t in self._trackers.items()
        }
