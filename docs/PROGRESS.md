# RL-100 fork — progress & notes

Working notes for our fork. `README.md` / `INSTALL.md` are upstream;
`INSTALL_blackwell.md` is ours (RTX 5090 / Blackwell). This doc tracks what we
changed, how to run it, and what we learned.

## What we changed vs upstream

- **Refactor**: `train.py` is now a thin entry; training logic lives in
  `rl_100/training/` (`workspace`, `bc_trainer`, `critic_trainer`, `offline_rl`,
  `online_rl`, `distill`). Removed `train_ddp.py` / `train_real.py` — single-GPU only.
- **ManiSkill tasks** (ours): env `rl_100/env/maniskill/` (PickCube + StackCube,
  difficulty config-driven), runner `env_runner/maniskill_runner.py` (eval +
  `idql_run` + fixed-initial-state eval + sparse-reward rollout collector),
  `dataset/maniskill_dataset.py` (decoupled from `PushTDataset`), task configs
  `config/task/maniskill_{pickcube,stackcube}.yaml`, demo→zarr tools in
  `tools/maniskill/`, flywheel driver `scripts/maniskill/run_flywheel.sh`.
- **AM-Q δ-gate** (paper Eq.9): `training/offline_rl.py` + `unio4.ope_gate` /
  `unio4.ope_gate_delta` (default off). Reverts a BPPO candidate when OPE does not
  improve by `delta·|AM-Q|`; upstream BPPO lacked this. See the δ-gate section.
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

### Harder task: StackCube (added) + single-round offline-RL knife-edge

- Added `StackCubeRL100-v1` (subclass of stock `StackCubeEnv`: our gemini camera,
  `goal_pos=cubeB` so agent_pos stays 19-dim and the pipeline is reused; spawn
  confined to the crop and configurable). Uses stock `solveStackCube` for demos.
- **BC scales steeply with #demos** (chunk config): 25→0.05, 50→**0.47**, 100→0.93.
  So 50 demos is the headroom sweet spot (BC ~0.47).
- **Single-round offline BPPO on StackCube/50-demo is a knife-edge**:
  `bppo_lr=1e-6` → inert (stays ~0.47, no gain); `1e-5` → collapses to 0. No clean
  improvement window. The BPPO loss is non-trivial (~0.05), i.e. the policy moves
  but the critic's advantage (from 50 demos) doesn't point to higher-success actions.
- **OPE is not a reliable selector here**: at `lr=1e-5` it read *higher* (1.10 vs
  0.95 baseline) exactly as env success crashed to ~0. So we cannot honestly pick a
  checkpoint from OPE on this limited-data sparse-reward setting.
- **Insight**: on limited data, single-round offline RL beating BC is unreliable;
  the lever that demonstrably moves success is **data growth** (the BC-vs-#demos
  table). → the **flywheel** is the thing to demonstrate, not single-round BPPO.

### AM-Q δ-gate (paper Eq.9): protects the BC, does not bootstrap it

The official released code (`bcb457f`) ships a *simplified* BPPO: it only advances
the behavior policy on OPE improvement (`is_update_old_policy`) and **never reverts**
a degrading candidate — the paper's Eq.9 AM-Q acceptance gate (reject + revert when
AM-Q doesn't improve by `delta=0.05|AM-Q|`) is absent. This gap is in upstream, not
introduced by our refactor. We added it as `unio4.ope_gate` (default off).

Test: BPPO from the **100-demo BC (real success 0.933)**, identical
`lr=2.83e-6, rollout=5, 1000 steps`, only the gate differing (20 evals, 30 ep each):

| | real success curve | mean | gate |
| --- | --- | --- | --- |
| ungated (shipped) | 0.933 -> drifts to 0.867-0.90 | ~0.89 | n/a |
| gated (Eq.9)      | **0.933 flat, all 20 evals**  | 0.933 | ACCEPT 1 / REJECT 19 |

- **The gate works**: it pins success at the BC level (revert kills the drift). The
  ungated run never returns to 0.933 after step ~150; the gated run never drops.
- **But it never climbs above BC.** The single early ACCEPT was a spurious OPE spike;
  afterwards nothing beat `best+delta`, so it reverts every candidate -> holds BC.
- **Why no improvement: the OPE is uninformative on StackCube sparse reward.** The
  rollout-Q mean is noisy (1.3-1.8 across evals) and the model-rollout success-frac
  is ~0.065 while real success is 0.93; crucially the OPE curves of the *degrading*
  (ungated) and *held* (gated) policies are nearly identical -> OPE cannot tell them
  apart. With no signal pointing uphill, the gate's only achievable role is
  protection, not bootstrap.
- **Verdict on "is offline RL broken?"**: no catastrophic bug -- the chunk PPO math
  and pipeline are sound (policy resumes at 0.933, updates run). Single-round offline
  RL simply provides **no genuine gain** here because the advantage/OPE signal from
  limited sparse-reward data doesn't identify higher-success actions. The lever that
  demonstrably moves success remains **data growth** (BC-vs-#demos; the flywheel).

### Flywheel (off2off, all-merge): low-saturation, not a spiral

Built the sim flywheel entirely on existing code (driver
`scripts/maniskill/run_flywheel.sh`): per round = deploy current BC with
`train.py eval=True data_collect=True` (runner writes h5) → official
`data_prepare.py --mode extend_zarr` merge → IL re-train BC on the bigger zarr.

all-merge, 4 rounds from the 50-demo StackCube (round-0 BC 0.47):

| round | r0 | r1 | r2 | r3 | r4 |
| --- | --- | --- | --- | --- | --- |
| episodes | 50 | 110 | 170 | 230 | 290 |
| BC success | 0.47 | 0.51 | **0.57** | 0.50 | 0.50 |

- **Saturates ~0.5 — not a collapse/spiral.** Mechanism: stochastic
  `data_collect` success is only ~25% (VIB + denoising noise drops a 0.5 policy to
  0.25), so each round adds ~15 success + ~45 fail → dataset success-fraction
  falls (100→60→49→42→37%) → BC tracks dataset quality, capped at ~0.5. The
  failures are near-misses (reach/grasp/almost-stack), so they dilute rather than
  poison.
- **Gain is bounded by collection success.** Expert demos scaled BC 50→0.47,
  100→0.93; 25%-success rollouts cap it ~0.5. So to climb, the dataset must stay
  high-quality.
- **Bug found & fixed** (was producing a fake "spiral"): `checkpoint.save_ckpt`
  defaults False → the BC re-train saved no checkpoint → the next round's collect
  deployed an *untrained* policy (0% collection) → failures flooded the buffer.
  Fix: BC re-train with `checkpoint.save_ckpt=True`, and the collect step uses the
  deployed policy's own training zarr for the normalizer (else a re-fit mismatch
  silently breaks the policy).
- **`only_success` merge also saturates ~0.50** (tested, 4 rounds): adding only
  the ~15 successes/round keeps the dataset clean but the volume added is tiny
  relative to the expert demos, and you only ever collect successes on cases the
  policy *already* solves — so it adds no new competence. Same ceiling as all-merge.
- **Why the flywheel saturates (root cause)**: self-collected data can only be as
  good as the deploying policy. To climb you need *higher-quality* additions than
  the current policy produces — i.e. either better demos (expert scaling works:
  50→0.47, 100→0.93) or an RL step that genuinely improves the policy before
  collection. The latter is what offline RL is supposed to provide — and on this
  data it doesn't (see the δ-gate section).

## Running the original (non-ManiSkill) tasks on RTX 5090 — triage

We checked whether any of the upstream task zoo (`config/task/`: adroit, dmc,
metaworld, dexart, pusht, ...) can be stood up on the Blackwell desktop. Result:

| task | sim on 5090 | dataset present? | verdict |
| --- | --- | --- | --- |
| **pusht** | runs (pymunk, CPU; smoke-tested reset/step/render) | no | sim OK, but data must be built from scratch |
| **dmc** | physics runs (`dm_control 1.0.41` + `mujoco 3.8.1` + EGL, smoke-tested) | no | blocked: obs path needs `pyrl` (not pip-installable) |
| metaworld | pkg not installed | no | needs install (MuJoCo-based, likely OK) |
| adroit / dexart / realdex / rotate / flipping / pour / cloth | old `mujoco-py` / SAPIEN-hardware | no | high risk on 5090; skip |

Two blocking facts:
- **No upstream task ships a usable dataset.** Every dataset class here expects the
  **point-cloud + off2off schema** (`state, action, point_cloud, next_state,
  next_action, next_point_cloud, reward, done, timeout, return`) — the DP3/off2off
  lineage, generated, not downloadable. (The public Diffusion-Policy pusht zarr is
  `img/state/action/keypoint` — wrong schema.)
- **pusht has no point-cloud and no demo generator.** Its env returns a 5-D state
  only; there is no `gen_demonstration_pusht`. Building pusht data means writing a
  (synthetic) point-cloud generator for the 2-D scene + sourcing demos + the off2off
  conversion — strictly more work than ManiSkill (where SAPIEN gives real point
  clouds and the motion planner gives demos for free), for a 2-D toy.

Conclusion: standing up an upstream task does **not** give a cheap independent
offline-RL validation; it is the same data-gen effort we already did, on a weaker
task. The faster validation is mixed-quality data in the working ManiSkill pipeline.

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

**Done**
- Pipeline reproduced & healthy on the chunk config (BC + IQL critic + dynamics/OPE
  + BPPO all run, resume works). PickCube BC = 100%; StackCube BC scales 50→0.47,
  100→0.93.
- Flywheel built on official code (all-merge + only_success); both saturate ~0.50.
- AM-Q δ-gate (paper Eq.9) implemented (`unio4.ope_gate`, default off) — the
  upstream BPPO omits it. It prevents BPPO from degrading the policy (0.933 held vs
  0.89 ungated) but cannot bootstrap above BC here because the OPE is uninformative.
- Triaged the upstream task zoo on the 5090 (only pusht runs; no datasets shipped).

**The wall we hit**
Single-round offline RL does not beat BC on our data, and the flywheel saturates,
for the *same* reason: our data is **homogeneous (all-expert / self-collected
successes)**. IQL/BPPO advantage only has signal when data is **mixed quality** (bad
+ good), so reward can tell the policy what *not* to imitate. On expert-only data
offline RL ≈ BC. This is expected, not a bug — but it means we have not yet shown
offline RL adding value.

**Next (recommended, cheapest decisive test)**
Run **BC vs offline-RL on the same mixed-quality zarr** (e.g. `maniskill_stackcube
_fw4.zarr`: ~290 ep, expert demos + collected successes + near-miss failures,
~37% success). No new sim or dataset needed; point clouds are real. If offline RL
exceeds BC's ~0.5 there (using reward to skip the failures), the pipeline is
validated and offline RL adds value exactly where theory predicts. If it does not,
the problem is deeper than data composition.

Out of scope (unchanged): online RL, CM distillation, upstream non-ManiSkill tasks.
