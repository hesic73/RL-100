#!/usr/bin/env bash
set -u
# RL-100 sim data flywheel (all-merge). Pure orchestration over EXISTING code:
#   collect  = train.py eval=True data_collect=True   (maniskill_runner writes h5)
#   merge    = tools/teleop_off2off_data/data_prepare.py --mode extend_zarr (official)
#   retrain  = train.py only_bc=True                  (IL re-training on expanded data)
# Per round m: deploy current BC -> collect rollouts -> all-merge into the zarr ->
# retrain BC from scratch on the bigger zarr -> record success. Repeat to saturation.
#
# Usage: START_ZARR=... START_BC=... FIRST=2 LAST=4 bash scripts/maniskill/run_flywheel.sh
cd /home/hsc/26summer/RL-100
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl100
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline WANDB_ERROR_REPORTING=false WANDB_SILENT=true
SCR=/tmp/claude-1001/-home-hsc-26summer-RL-100/6f5433e5-009a-465e-a2f2-0aa4162792e3/scratchpad

TASK=maniskill_stackcube
CUR_ZARR=${START_ZARR:-data/maniskill_stackcube_r1_all.zarr}   # round-1 (110 ep)
PREV_BC=${START_BC:-logs/bc_r1all}                              # round-1 BC dir (~0.63)
FIRST=${FIRST:-2}
LAST=${LAST:-4}
ONLY_SUCCESS=${ONLY_SUCCESS:-false}   # true = merge only successful rollouts
RUN_TAG=${RUN_TAG:-all}               # output namespace (keeps all/succ runs separate)

CHUNK="use_action_embed=False horizon=18 n_action_steps=16 n_obs_steps=3 ft_all_actions=False num_inference_steps=10 \
  policy.model=dp3 policy.encoder_type=dp3vib policy.act=mish \
  policy.encoder_output_dim=64 policy.diffusion_step_embed_dim=256 policy.down_dims=[256,512,1024] \
  policy.ddim_noise_scheduler.num_train_timesteps=100 policy.cm_noise_scheduler.num_train_timesteps=100 \
  policy.scheduler_type=ddim policy.use_agent_pos=True policy.use_vib=True policy.use_recon=True \
  policy.mlp_policy_depth=3 policy.beta_kl=1e-4 \
  chunk_as_single_action=True dynamics_type=mlp dynamics.prediction_mode=full predict_r=True \
  dataloader.batch_size=1024 val_dataloader.batch_size=1024 optimizer.lr=2.83e-4"

echo "flywheel: TASK=$TASK start_zarr=$CUR_ZARR start_bc=$PREV_BC rounds $FIRST..$LAST"
echo "round 0 (50 demos) BC=0.47 ; round 1 (110 ep) BC~0.63"

for m in $(seq $FIRST $LAST); do
  cdir=logs/fw_${RUN_TAG}/r${m}_collect
  bdir=logs/fw_${RUN_TAG}/r${m}_bc
  newz=data/${TASK}_fw${RUN_TAG}${m}.zarr
  echo "================== ROUND $m =================="

  # 1) collect from current BC policy (junk 1-epoch critic/dynamics, unused by collect)
  rm -rf "$cdir"
  python RL-100/train.py --config-name=rl100_3d_epsilon.yaml task=$TASK \
    hydra.run.dir=$(pwd)/$cdir exp_name=fw_r${m}_collect logging.name=fw_r${m}_collect logging.mode=offline \
    training.seed=$((100+m)) training.device=cuda:0 training.resume=True \
    $CHUNK offline=True only_bc=False online=False distill_phase=null kl_annealing=True eval=True data_collect=True \
    training.num_critic_epochs=1 dynamics.dynamics_max_epochs=1 \
    critic.load_pretrain=True +unio4.stage1_resume_dir=$(pwd)/$PREV_BC +unio4.critic_artifact_dir=$(pwd)/$cdir/critic \
    unio4.idql_eval=False unio4.use_ema_eval=True unio4.eval_times=2 \
    critic.omega=0.9 critic.gamma=0.997 critic.q_hidden_dim=1024 critic.v_hidden_dim=512 critic.q_layer_norm=True \
    dynamics.dynamics_hidden_dims=[1024,1024,512,512] \
    task.dataset.zarr_path=$CUR_ZARR task.norm_dataset.zarr_path=$CUR_ZARR \
    task.env_runner.eval_episodes=30 task.env_runner.env_num=1 task.env_runner.seed=$((700+m)) \
    > $SCR/fw_r${m}_collect.log 2>&1
  nh5=$(find $cdir/rollouts -name "episode_*.h5" 2>/dev/null | wc -l)
  echo "  collected $nh5 episodes -> $cdir/rollouts"

  # 2) all-merge into the growing zarr
  cfg=$cdir/data_prepare.yaml   # generated config under logs/ (git-ignored), not the tracked configs dir
  cat > $cfg <<EOF
mode: extend_zarr
overwrite: true
only_success: ${ONLY_SUCCESS}
num_points: 512
base_zarr_path: ${CUR_ZARR}
zarr_output_path: ${newz}
rollout_sources:
  - {name: fw_r${m}, type: h5_rollout_dir, path: ${cdir}/rollouts, enabled: true}
EOF
  python tools/teleop_off2off_data/data_prepare.py --config $cfg > $SCR/fw_r${m}_merge.log 2>&1
  neps=$(python -c "import zarr; print(len(zarr.open('${newz}','r')['meta']['episode_ends']))" 2>/dev/null)
  echo "  merged -> $newz ($neps episodes)"

  # 3) IL re-training (BC from scratch) on the expanded zarr
  rm -rf "$bdir"
  python RL-100/train.py --config-name=rl100_3d_epsilon.yaml task=$TASK \
    hydra.run.dir=$(pwd)/$bdir exp_name=fw_r${m}_bc logging.name=fw_r${m}_bc logging.mode=offline \
    training.seed=42 training.device=cuda:0 training.resume=False \
    training.num_epochs=500 training.rollout_every=50 training.checkpoint_every=50 checkpoint.save_ckpt=True \
    $CHUNK offline=True only_bc=True online=False distill_phase=null kl_annealing=True \
    task.dataset.zarr_path=$newz task.norm_dataset.zarr_path=$newz \
    task.env_runner.eval_episodes=30 task.env_runner.env_num=1 task.env_runner.seed=42 \
    > $SCR/fw_r${m}_bc.log 2>&1
  curve=$(grep -oE "success_rate: [0-9.]+" $SCR/fw_r${m}_bc.log | grep -v SR_test | grep -oE "[0-9.]+" | awk '{printf "%.2f ",$1}')
  echo "  ROUND $m BC curve: $curve"

  CUR_ZARR=$newz
  PREV_BC=$bdir
done
echo "================== FLYWHEEL DONE =================="
