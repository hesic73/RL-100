# RL-100 fork — progress & notes

Working notes for our fork. `README.md` / `INSTALL.md` are upstream;
`INSTALL_blackwell.md` is ours (RTX 5090 / Blackwell). This doc tracks what we
changed, how to run it, and what we learned.

## What we changed vs upstream

- **Refactor**: `train.py` is now a thin entry; training logic lives in
  `rl_100/training/` (`workspace`, `bc_trainer`, `critic_trainer`, `offline_rl`,
  `online_rl`, `distill`). Removed `train_ddp.py` / `train_real.py` — single-GPU only.
- **ManiSkill PickCube task** (ours): env `rl_100/env/maniskill/`, runner
  `env_runner/maniskill_runner.py` (eval + `idql_run` + fixed-initial-state eval +
  sparse-reward rollout collector), `dataset/maniskill_dataset.py` (decoupled from
  `PushTDataset`), task config `config/task/maniskill_pickcube.yaml`, demo→zarr
  tools in `tools/maniskill/`.
- **Bug fixes**: IQL Bellman discount (`gamma**1` for the non-chunk single-step
  transition), missing `import hydra`, `output_dir` method/attr shadow, online
  `iql_ft` undefined names, online buffer made image-optional (point-cloud-only).

## How to run (ManiSkill PickCube)

```bash
# 1) data: motion-planning demos -> FPS zarr (difficulty via --env-kwargs)
bash scripts/maniskill/prepare_pickcube_data.sh            # default difficulty
# 2) offline RL, official CHUNK config (recommended): stage1 makes BC+critic+dynamics,
#    stage2 sweeps BPPO reusing them.
bash scripts/maniskill/train_policy_chunk_two_stage.sh 42
# non-chunk variant (single-step): scripts/maniskill/train_policy_two_stage.sh
```

## Key findings

- **Use the official CHUNK config** (`policy.model=dp3`, `n_action_steps=16`,
  `chunk_as_single_action=True`). With it, BC already **solves PickCube (100%)**,
  the IQL critic converges, and the OPE (dynamics rollout-Q ≈ 4.1) is stable. The
  **non-chunk single-step** config (skipnet, `n_action_steps=1`) was the source of
  every earlier pathology (policy collapse, flat/useless OPE) — not a code bug.
- **OPE needs a long rollout horizon to work on sparse reward**: 16-step chunks ×
  ~5 rollout steps ≈ 80 env steps > one ~70-step episode, so the model rollout
  reaches the sparse terminal reward and rollout-Q becomes an informative,
  non-cheating selection signal. With `n_action_steps=1`, 5 rollout steps = 5 env
  steps → never reaches reward → flat OPE (~0.03, useless).
- **`unio4.is_update_old_policy`**: upstream default `True` is fine under chunk
  (informative OPE); under the non-chunk noisy OPE it promoted on noise and caused
  collapse.
- **No headroom on PickCube**: chunk BC is already 100%, so offline RL / the
  flywheel can't show a gain here. Raising goal height + spawn width alone did NOT
  help (clean motion-planner demos stay fully imitable). To demonstrate offline-RL
  / flywheel value we need a **harder task** or **fewer / sub-optimal demos**.

## Conventions / gotchas

- **Sparse reward only** (real-world fidelity): success=1 on the terminal step,
  else 0 — for demos, rollouts, and the critic, consistently.
- **Difficulty is config-driven, decoupled from code**: set
  `task.env_runner.env_kwargs` (e.g. `{cube_spawn_half_size:0.10,goal_z_max:0.25}`);
  the env source keeps the original defaults. Demo generation and zarr conversion
  must use the **same** `--env-kwargs` so demos and eval match.
- `only_bc=True` **stops after BC** in our `train.py` (upstream `train_ddp.py` did
  not stop). Use `only_bc=False` for the offline stages.
- Non-chunk `unio4.rollout_length` must be a multiple of `n_action_steps`.
- Hydra list overrides must be real lists (`policy.down_dims=[256,512,1024]`), not
  quoted strings (a quoted string crashes the dp3 UNet).
- `data/` and `logs/` are git-ignored — datasets, demos, and checkpoints are not
  committed; regenerate the zarr with `prepare_pickcube_data.sh`.

## Status / next

- **Done**: BC + offline RL + critic/dynamics/OPE reproduced and healthy on
  PickCube (chunk config).
- **Next**: a harder setting (harder task or fewer/sub-optimal demos) to give
  offline RL / the flywheel real headroom; then build the sim off2off flywheel
  loop (collect → merge → retrain). Online RL and CM distillation are out of
  scope for now.
