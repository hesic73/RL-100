#!/usr/bin/env bash
set -euo pipefail

# Faithful single-GPU port of the official
#   scripts/Diffusion/Offline/3D/train_policy_chunk_two_stage.sh
# adapted to task=maniskill_pickcube and our refactored single-GPU train.py.
#
# Chunk policy: n_action_steps=16 chunks treated as one action
# (chunk_as_single_action=True). The point: each OPE/dynamics rollout step
# advances 16 env steps, so a short model rollout (5-10) spans ~80-160 env steps
# > one ~66-step episode -> the rollout reaches the sparse terminal reward, so the
# OPE (rollout-Q) becomes informative (unlike n_action_steps=1 where 5 steps never
# reach success).
#
# Config copied verbatim from the official chunk launcher: dp3/mish backbone,
# gamma=0.997, omega=0.9, predict_r=True, kl_annealing=True, beta_kl=1e-4,
# use_action_embed=False, idql_eval=False, use_ema_eval=True, q_hidden=1024,
# q_layer_norm=True, dynamics_hidden=[1024,1024,512,512], 800/800/350 epochs,
# batch 1024, lr 2.83e-4, sequence_stride=16 on critic/finetune datasets.
# Deviation: only_bc=False (our train.py stops at BC if True, unlike train_ddp.py).
#
# Usage: bash scripts/maniskill/train_policy_chunk_two_stage.sh [seed]

seed=${1:-42}
task_name=${TASK:-maniskill_pickcube}
config_name='rl100_3d_epsilon'
exp_name=${task_name}-rl100-chunk

# Official chunk sweep grid (overridable).
LR_VALUES=${LR_VALUES:-"1e-6 1.42e-6 2.83e-6"}
ROLLOUT_VALUES=${ROLLOUT_VALUES:-"3 5 10"}
CLIP_STD_MAX_VALUES=${CLIP_STD_MAX_VALUES:-"0.1 null"}
BPPO_STEPS=${BPPO_STEPS:-5000}

RUN_STAGE1=${RUN_STAGE1:-true}
RUN_SWEEP=${RUN_SWEEP:-true}

# Smoke overrides (tiny epochs) so the full chain can be validated cheaply.
NUM_EPOCHS=${NUM_EPOCHS:-800}
NUM_CRITIC_EPOCHS=${NUM_CRITIC_EPOCHS:-800}
DYN_EPOCHS=${DYN_EPOCHS:-350}
EVAL_EPISODES=${EVAL_EPISODES:-30}

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_ERROR_REPORTING=false

root_run_dir="$(pwd)/logs/chunk/${exp_name}_seed${seed}"
stage1_run_dir="${root_run_dir}/stage1"
critic_artifact_dir="${root_run_dir}/critic_c16_f16"
mkdir -p "${root_run_dir}"

common_params() {
    local run_dir=$1 lr=$2 rollout_length=$3 clip_std_max=$4
    echo "task=${task_name} \
        hydra.run.dir=${run_dir} \
        training.debug=False training.seed=${seed} training.device=cuda:0 \
        exp_name=${exp_name} logging.mode=offline checkpoint.save_ckpt=True use_wandb=True \
        unio4.bppo_lr=${lr} unio4.rollout_length=${rollout_length} clip_std_max=${clip_std_max} \
        training.resume=True \
        policy._target_=rl_100.policy.rl100_3d.RL1003D \
        policy.ddim_noise_scheduler.num_train_timesteps=100 \
        policy.cm_noise_scheduler.num_train_timesteps=100 \
        use_action_embed=False horizon=18 n_action_steps=16 n_obs_steps=3 \
        ft_all_actions=False num_inference_steps=10 \
        policy.model=dp3 policy.encoder_type=dp3vib policy.act=mish \
        policy.encoder_output_dim=64 policy.diffusion_step_embed_dim=256 policy.down_dims=[256,512,1024] \
        policy.scheduler_type='ddim' policy.use_agent_pos=True policy.use_vib=True policy.use_recon=True \
        policy.mlp_policy_depth=3 policy.beta_kl=1e-4 \
        offline=True only_bc=False online=False distill_phase=null kl_annealing=True \
        unio4.idql_eval=False unio4.use_ema_eval=True unio4.eval_times=1 \
        critic.omega=0.9 critic.gamma=0.997 critic.q_hidden_dim=1024 critic.v_hidden_dim=512 critic.q_layer_norm=True \
        dynamics_type='mlp' dynamics.prediction_mode='full' predict_r=True \
        dynamics.dynamics_hidden_dims=[1024,1024,512,512] \
        chunk_as_single_action=True bppo_chunk_level_ratio=True \
        offline_chunk_ratio_mode=scalar offline_chunk_adv_mode=scalar_iql \
        training.num_epochs=${NUM_EPOCHS} training.num_critic_epochs=${NUM_CRITIC_EPOCHS} dynamics.dynamics_max_epochs=${DYN_EPOCHS} \
        dataloader.batch_size=1024 val_dataloader.batch_size=1024 \
        optimizer.lr=2.83e-4 critic.q_lr=2.83e-4 critic.v_lr=2.83e-4 dynamics.dynamics_lr=5.66e-4 \
        task.critic_dataset.sequence_stride=16 task.finetune_dataset.sequence_stride=16 \
        task.env_runner.eval_episodes=${EVAL_EPISODES} task.env_runner.env_num=1 task.env_runner.seed=${seed}"
}

if [ "${RUN_STAGE1}" = true ]; then
    echo "=== Stage 1 (chunk): BC + IQL critic + dynamics (bppo_steps=0) ==="
    python RL-100/train.py --config-name="${config_name}.yaml" \
        $(common_params "${stage1_run_dir}" 1e-6 3 0.1) \
        unio4.bppo_steps=0 \
        +unio4.critic_artifact_dir="${critic_artifact_dir}"
fi

if [ "${RUN_SWEEP}" = true ]; then
    echo "=== Stage 2 (chunk): BPPO sweep, reuse stage-1 artifacts ==="
    for lr in ${LR_VALUES}; do
      for rl in ${ROLLOUT_VALUES}; do
        for csm in ${CLIP_STD_MAX_VALUES}; do
            ts=$(date +"%Y%m%d-%H%M%S")
            sweep_dir="${root_run_dir}/sweep/${ts}-lr_${lr}_rollout_${rl}_clip_${csm}"
            mkdir -p "${sweep_dir}"
            echo "--- chunk sweep: lr=${lr} rollout=${rl} clip_std_max=${csm} ---"
            python RL-100/train.py --config-name="${config_name}.yaml" \
                $(common_params "${sweep_dir}" "${lr}" "${rl}" "${csm}") \
                unio4.bppo_steps="${BPPO_STEPS}" \
                critic.load_pretrain=True \
                +unio4.stage1_resume_dir="${stage1_run_dir}" \
                +unio4.critic_artifact_dir="${critic_artifact_dir}" \
                +unio4.global_best_dir="${root_run_dir}/best"
        done
      done
    done
fi
