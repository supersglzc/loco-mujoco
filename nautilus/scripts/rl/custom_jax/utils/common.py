"""
Common utilities — Tracker, set_random_seed, load_class_from_path,
list_class_names. Shared with custom_torch (no torch imports).
"""
from __future__ import annotations

import ast
import importlib.util
import random
import sys
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def set_random_seed(seed: int | None = None) -> int:
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def load_class_from_path(cls_name: str, path):
    mod_name = f"MOD{cls_name}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def list_class_names(dir_path) -> dict:
    dir_path = Path(dir_path)
    out: dict[str, Path] = {}
    for py_file in dir_path.rglob("*.py"):
        if not py_file.is_file() or py_file.name == "__init__.py":
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                out[node.name] = py_file
    return out


class Tracker:
    """Fixed-window moving statistic over recent episode metrics."""

    def __init__(self, max_len: int):
        self.moving_average: deque = deque(maxlen=max_len)
        self.max_len = max_len

    def update(self, value):
        if isinstance(value, np.ndarray):
            self.moving_average.extend(value.tolist())
        elif isinstance(value, Sequence):
            self.moving_average.extend(value)
        else:
            self.moving_average.append(value)

    def mean(self) -> float:
        if not self.moving_average:
            return 0.0
        return float(np.mean(self.moving_average))

    def std(self) -> float:
        if not self.moving_average:
            return 0.0
        return float(np.std(self.moving_average))


def handle_timeout(dones: np.ndarray, info: dict) -> np.ndarray:
    """Mask `dones` so envs killed by TimeLimit don't bootstrap as terminal."""
    timeout = info.get("TimeLimit.truncated") if isinstance(info, dict) else None
    if timeout is None and isinstance(info, dict):
        timeout = info.get("time_outs")
    if timeout is None:
        return dones
    timeout_arr = np.asarray(timeout, dtype=bool)
    return np.asarray(dones) * (~timeout_arr)
