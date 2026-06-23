# RL-100 x ManiSkill PickCube - Handoff

Status as of 2026-06-23. This document covers the uncommitted work that integrates ManiSkill
PickCube into RL-100 on an RTX 5090 (Blackwell), the current results, and the open problems.

## TL;DR

- A behavior-cloning (BC) policy on our own `PickCubeRL100-v1` task trains end-to-end and works:
  **~22% task success** (11/50 episodes, point-cloud DP3 policy, action chunking).
- Two separate bugs both presented as "0% success", both fixed:
  - The offline-RL stages crashed under action chunking, which prevented a clean BC run (see "The bug").
  - The eval success metric was silently always 0: `info['success']` from MultiStepWrapper is a numpy
    array, but the runner / visualize_checkpoint guarded it with `isinstance(s, (list, tuple))`, which is
    False for ndarrays, so `success` was never set. The policy was actually succeeding the whole time.
- Offline RL (the stages after BC) is NOT YET supported under action chunking. It now fails with
  an explicit error instead of a cryptic crash. Implementing it is the main open task.

## Environment

- conda env `rl100` (Python 3.10, torch 2.11.0+cu128 with sm_120, mani_skill 3.0.1, numpy pinned 1.26.4).
- Setup: see `INSTALL_blackwell.md` and `requirements-blackwell.txt`.
- `rl_100` is a pip-installed editable package (`pip install -e RL-100 --no-deps --no-build-isolation`).
  Do NOT use `PYTHONPATH`. Run everything from the repo root `/home/hsc/26summer/RL-100`.

## How to run (all from repo root, `conda activate rl100`)

```bash
# 1. Generate demos locally (motion planning, no download) + build the FPS zarr
bash scripts/maniskill/prepare_pickcube_data.sh        # -> data/maniskill_pickcube.zarr

# 2. Train BC (no args; edit the vars at the top of the script)
bash scripts/maniskill/train_pickcube.sh               # -> logs/maniskill_pickcube_run0_seed42/

# 3. Evaluate a checkpoint (success rate + per-episode mp4s)
python tools/maniskill/visualize_checkpoint.py \
  --ckpt logs/maniskill_pickcube_run0_seed42/checkpoints/latest.ckpt \
  --num-episodes 50 --seed 5000
```

Step 3 prints `success_rate` (~0.22 over 50 episodes) and saves per-episode mp4s. It reads
`control_mode` from the checkpoint cfg, so it cannot hit the control-mode-mismatch pitfall.

## Uncommitted changes

### Modified (tracked)

- `RL-100/train.py`
  - `ROOT_DIR`: `parent.parent.parent` -> `parent.parent`. The script `os.chdir`s to the repo root
    `/home/hsc/26summer/RL-100`; all data/log paths are relative to there. (Previously it chdir'd to
    `/home/hsc/26summer`, ABOVE the repo, which scattered data/logs.)
  - Prints the wandb run URL (the wandb logger is set to ERROR level, which otherwise hides it).
  - Sets `env_runner.current_epoch` before each eval so eval videos are named by epoch.
  - After the BC loop (~line 652): `if cfg.only_bc: return` (explicit "BC only" stop), and
    `if n_obs_steps>1 and not chunk_as_single_action: raise RuntimeError(...)` (explicit, clear error
    for offline RL under chunking, replacing a cryptic downstream einsum crash). See "The bug" below.
- `RL-100/rl_100/env/__init__.py`: reduced to only `from .dmc import ...`. The Adroit / DexArt /
  MetaWorld / UR5 / Franka / Flipping imports were removed so that `import rl_100.env` (done by the
  maniskill task) does not pull in legacy/broken real-robot deps. SIDE EFFECT: those legacy tasks no
  longer import. If they must coexist, restore them as lazy imports instead of deleting.
- `.gitignore`: un-ignore `CLAUDE.md`; ignore `*.so`.
- `RL-100/rl_100/unidpg/transition_model/utils/logger.py` + `RL-100/rl_100/unidpg/dynamics_eval_batch.py`:
  the dynamics-stage logger hardcoded a top-level `log/` dir (`ROOT_DIR = "log"`), separate from the
  run's `logs/...` dir, so an offline-RL run produced BOTH `log/` and `logs/` at the repo root.
  `make_log_dirs` now takes an optional `root_dir`, and the dynamics logger nests under the run's
  output dir (`<output_dir>/saved_models_*/dynamics_logs/...`). Only reachable once offline RL runs.

### Eval success-metric fix (in the new maniskill files below)

`MultiStepWrapper` returns `info['success']` as a numpy array (`take_last_n` -> `np.array(...)` for
non-tensor values), but `maniskill_runner.py` and `visualize_checkpoint.py` read it as
`if isinstance(s, (list, tuple))` - False for ndarrays - so `success` was never set and every
info-based eval reported 0 while the policy was actually succeeding ~22%. Both now read it as
`np.asarray(info.get('success')).reshape(-1)`. The env adapter (`maniskill_pickcube.py:step`) now
raises if `'success'` is missing from ManiSkill's info instead of silently returning False.

### New (untracked)

- `RL-100/rl_100/env/maniskill/maniskill_pickcube.py` - one file, 3 sections:
  1. `PickCubeRL100Env(BaseEnv)` registered as `PickCubeRL100-v1` (max_episode_steps=100). Our own
     task adapted from mani_skill's `pick_cube.py` (NOT a kwargs-hacked subclass). Difficulty knobs are
     clean `__init__` params (cube_spawn_half_size=0.06, goal_xy_half_size=0.06, goal_z_max=0.1, ...).
     Orbbec Gemini-like camera (640x400, fov 65 deg, front-above) so the cube is visible in the cloud.
  2. `process_point_cloud` (workspace crop + `fpsample` FPS) and `build_agent_pos`
     (qpos9 + tcp_pose7 + goal_pos3 = 19). Robot is KEPT in the cloud (no segmentation - real cameras
     cannot segment). Mirrors RL-100's real-robot `franka/realsense_color.py`.
  3. `ManiSkillPointcloudEnv(gym.Env)` - single-env gym-0.21 adapter (batched torch -> single numpy).
- `RL-100/rl_100/env/maniskill/__init__.py` - exports the above (also registers the env on import).
- `RL-100/rl_100/env_runner/maniskill_runner.py` - `ManiSkillRunner`: rollouts, success_rate,
  mean_returns, local mp4s under `<run>/eval_videos/epoch_<NNNN>/`.
- `RL-100/rl_100/config/task/maniskill_pickcube.yaml` - task cfg. shape_meta: point_cloud [512,3],
  agent_pos [19], action [8]. Reuses `PushTDataset`. `zarr_path: data/maniskill_pickcube.zarr` (relative).
- `RL-100/pyproject.toml` - makes `rl_100` pip-installable editable.
- `scripts/maniskill/{prepare_pickcube_data,train_pickcube}.sh` - no positional args; edit the vars
  at the top. Run from repo root.
- `tools/maniskill/`:
  - `generate_demos.py` - motion-planning demos on `PickCubeRL100-v1` (actions + env_states).
  - `convert_states_to_zarr.py` - state-replays demos through the env, writes the FPS zarr.
  - `visualize_checkpoint.py` - rollout a checkpoint, print success_rate, save mp4s (loads ema_model).
  - `debug_bc.py` - SELF-CONTAINED BC loop that bypasses train.py: trains via `policy.compute_loss`,
    then runs in-domain action-reproduction + closed-loop grasp/success eval. This is the cleanest way
    to validate the policy/data/obs pipeline independent of the offline-RL machinery.
  - `dump_pointcloud.py` + `view_pcd.py` - dump raw .pcd, view with open3d.
  - `smoke_test.py` - hydra-composed end-to-end import/train-step/eval check.
- `INSTALL_blackwell.md`, `requirements-blackwell.txt`, `CLAUDE.md`.

## The bug behind "0% / doesn't grasp" (fixed)

`train.py run()` runs the main BC loop (loss converges, checkpoints save) and then ALWAYS falls into
critic -> `train_dynamics` -> `finetune_dp3` (BPPO offline RL), even with `only_bc=True` (the flag is
never checked inside `update_distribution`). Under action chunking (`n_obs_steps=2`) the dynamics
crashes, so the run dies right after BC.

Exact root cause (verified by instrumenting `ensemble_dynamics_for_batch.py:step`): `prediction_mode`
defaults to `"last"`, so the dynamics is built with `obs_dim = feature_dim = 128` (single step). During
dynamics training it is fed single-step features (128, ok). But BPPO's `NStepValueEstimation`
(uni_ppo.py:411) feeds it the full 2-step concatenated policy feature (256), so `obs_act = 256 + action`
mismatches the model's first `EnsembleLinear` -> `einsum 256 vs 384`. Setting `dynamics.prediction_mode=full`
fixes that einsum, but the crash then MOVES DOWNSTREAM to the critic Q/value net
(`net.py:312 torch.cat([s, a_embed])`, s 3-dim vs a_embed 2-dim).

Conclusion: offline RL under chunking is a cascade of single-step assumptions (dynamics + Q-net +
value-net + N-step advantage rollout), plus a separate `chunk_as_single_action=True` code path that
needs `task.finetune_dataset` / `task.critic_dataset` configs that do not exist for this task.

Fix applied: `only_bc=True` now stops cleanly after BC; `only_bc=False` under chunking raises an
explicit, descriptive error instead of the cryptic einsum. (An earlier attempt used a bare
`if only_bc: return` framed as "skipping the broken offline RL" - that was a silent dodge of a real
bug and was replaced by the explicit guard above.)

## Current results

- BC, 100 local MP demos, chunking (horizon=5 / n_obs=2 / n_action=4), 400 epochs: trains clean,
  no crash. Final checkpoint, `visualize_checkpoint.py --num-episodes 50 --seed 5000`:
  success_rate 0.22 (11/50). (Note: success is stochastic across rollouts because the diffusion policy
  samples; expect a few % variance run-to-run.)
- `debug_bc.py` clean BC (4000 steps): in-domain action-reproduction MAE 0.016 (action std 1.21).
  Confirms the data / policy / obs pipeline is correct independent of train.py.
- IMPORTANT: the per-eval `success_rate: 0.000` lines you may see in OLD training logs were the
  info-success-type bug, NOT the true rate. After the fix the runner reports real success during training.

## Open problems / next steps

1. (Largest) Implement offline RL under action chunking - the lever to push success above the BC
   ceiling. Requires making the dynamics, Q-net, value-net, and N-step advantage rollout all handle
   `n_obs_steps>1`, and wiring up the `chunk_as_single_action=True` path with the right
   finetune/critic datasets. First step is known (`dynamics.prediction_mode=full`); the next crash
   is `net.py:312`. This is a multi-component, iterative effort.
2. BC success is modest (~22%): grasps but places imprecisely. Cheap levers first: more demos
   (currently 100) and/or more epochs.
3. `env/__init__.py` was gutted - legacy tasks (Adroit/Franka/...) now fail to import. Convert to
   lazy imports if coexistence is needed.
4. Nothing is committed yet. Suggested split: one commit for the ManiSkill integration, one for the
   train.py `only_bc`/guard fix. Decide whether `CLAUDE.md` should be tracked.
5. Artifacts (gitignored): demos zarr at `data/maniskill_pickcube.zarr`, logs at
   `logs/maniskill_pickcube_run0_seed42/`.
