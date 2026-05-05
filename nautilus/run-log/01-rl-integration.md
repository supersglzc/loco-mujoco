# 01 — rl-integration-generator (custom_jax)

- **Subagent:** rl-integration-generator
- **Timestamp:** 2026-05-04T20:13Z
- **Algorithm source:** `custom_jax`
- **Algorithm slug:** `custom_jax` (rendered tree at `nautilus/scripts/rl/custom_jax/`)
- **Algorithms:** ppo, sac, td3
- **Logging mode:** local (W&B disabled)
- **Env-generator backend:** uv (`/home/steven/code/agentic/loco-mujoco/.venv`)
- **Default smoke task:** UnitreeH1 (auto-promoted to MjxUnitreeH1 by env_wrapper)

## Per-algorithm T1-T5 result

| Algorithm | T1 train | T2 eval | T3 render | T4 plot | T5 log | Trial dir |
|---|---|---|---|---|---|---|
| ppo | OK (it=7, return=6.88) | OK (mean_return=5.16) | OK (62 frames, 549 KB MP4) | OK (7 PNG curves) | OK (TB=1, jsonl=7 lines) | `nautilus/outputs/ppo_UnitreeH1_20260504-200049/` |
| sac | OK (it=18, return=6.40) | OK (mean_return=14.12) | OK (54 frames, 452 KB MP4) | OK (7 PNG curves) | OK (TB=1, jsonl=133 lines) | `nautilus/outputs/sac_UnitreeH1_20260504-200623/` |
| td3 | OK (it=18, return=7.03) | OK (mean_return=5.08) | OK (31 frames, 146 KB MP4) | OK (5 PNG curves) | OK (TB=1, jsonl=95 lines) | `nautilus/outputs/td3_UnitreeH1_20260504-201108/` |

All five tiers PASS for all three algorithms.

## Plugin patches applied during this run

This was the first end-to-end exercise of `custom_jax`; six bugs surfaced and were patched in-place at `/home/steven/code/agentic/claude-nautilus/`. All patches now live in the templates so future renders are clean.

1. **`scripts/rl-integration-generator/render_rl_suite.py`** — pre-flight relaxed to accept the uv backend (presence of `<repo>/.venv/bin/python` + `<repo>/nautilus/setup_uv.sh`), no longer demands `docker/Dockerfile`. Output paths moved under `<repo>/nautilus/`: `nautilus/scripts/rl/<slug>/`, `nautilus/configs/rl/`, `nautilus/utils/data_logger.py`, `nautilus/rl-integration.md`. JSON receipt + tune.py path updated to match.
2. **`scripts/rl-integration-generator/validate_rl_suite.py`** — T3 yaml parse now scans `<repo>/nautilus/configs/rl/` (was `<repo>/configs/rl/`).
3. **`templates/rl-integration-generator/custom_jax/scripts/env_wrapper.py.template`** — `_build_loco_mujoco_env` (a) imports `LogWrapper` (was the non-existent `RichLogWrapper`); (b) auto-prepends the `Mjx` prefix when the user passes a CPU env name (e.g. `UnitreeH1` → `MjxUnitreeH1`) so the JAX/MJX backend gets selected via `RLFactory.make`.
4. **`templates/rl-integration-generator/custom_jax/algo/{ppo,sac,td3}.py.template`** — `reset_agent` now uses `jax.random.split(self.split(2), n)` and calls `self.env.reset(reset_rngs)` without the unsupported `env_id=` kwarg. Matches `loco_mujoco.algorithms.ppo_jax.PPOJax` canonical usage.
5. **`templates/rl-integration-generator/custom_jax/algo/ac_base.py.template`** — `__post_init__` resolves obs/action shapes via `getattr(env, 'observation_space', None) or env.info.observation_space` so it works against both gymnasium-style envs AND loco-mujoco's `MDPInfo`-via-`info` proxy.
6. **`templates/rl-integration-generator/custom_jax/models/mlp.py.template`** — `TanhDiagGaussianActor.deterministic_action` signature changed from `(self, params, x)` to `(self, x)`; the previous form double-applied params and broke `module.apply(params, x, method='deterministic_action')` in `_ActorView.get_actions`.
7. **`templates/rl-integration-generator/custom_jax/scripts/{eval,render}.py.template`** — `agent_cls.from_dict(payload, env=env)` no longer passes the CLI cfg, so the saved training cfg's `max_grad_norm` etc. are honored and the optimizer chain has the same #elements as the saved opt_state.
8. **`templates/rl-integration-generator/configs/{ppo,sac,td3}.yaml.template`** — added `checkpoint: null`, `n_episodes: 10`, `render_max_steps: 1000` so eval/render CLIs can use `checkpoint=<path>` (Hydra struct mode no longer rejects the override).
9. **`templates/rl-integration-generator/custom_jax/algo/ppo.py.template`** docstring + `env_wrapper.py.template` docstring — corrected the env contract to drop the `env_id=` kwarg.

## Outcome

custom_jax now produces a working ppo/sac/td3 training pipeline against loco-mujoco's MJX humanoid envs. All artifacts (`AgentXXX_saved.pkl`, `tb/`, `metrics.jsonl`, `curves/*.png`, `render.mp4`, `resolved_config.yaml`) land in `nautilus/outputs/<algo>_<task>_<ts>/`.
