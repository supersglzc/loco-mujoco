"""
Checkpoint save / load — mirrors loco_mujoco/algorithms/common/base_algorithm.py.

Each agent has:
    serialize() → {"agent_conf": <dict>, "agent_state": <dict>}
    from_dict(d) → (agent_conf, agent_state)

We pickle that dict to <dir>/<class_name>_saved.pkl.

Same on-disk shape as the upstream loco-mujoco JaxRLAlgorithmBase, so a
checkpoint produced here is interchangeable with one produced by the
reference IPPOJax / PPOJax classes.
"""
from __future__ import annotations

import pickle
from pathlib import Path

_SUFFIX = ".pkl"


def save_agent(out_dir, agent) -> Path:
    """Pickle the agent's serialized conf+state to <out_dir>/<ClassName>_saved.pkl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{type(agent).__name__}_saved{_SUFFIX}"
    payload = agent.serialize()
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_agent_dict(ckpt_path) -> dict:
    """Load and return the raw serialized payload {agent_conf, agent_state}.

    Resurrecting it into a live agent is class-specific (each Agent class owns
    its own from_dict). eval.py / render.py call this then dispatch via
    AgentXXX.from_dict(payload, env, cfg).
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such checkpoint: {path}")
    if path.suffix != _SUFFIX:
        raise ValueError(f"Expected {_SUFFIX}, got {path.suffix}")
    with open(path, "rb") as f:
        return pickle.load(f)
