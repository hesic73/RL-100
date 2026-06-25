#!/usr/bin/env bash
set -euo pipefail

# Faithful single-GPU port of the official
#   scripts/Diffusion/Offline/3D/train_policy_two_stage.sh
# adapted to task=maniskill_pickcube and our refactored single-GPU entry
# (RL-100/train.py instead of the official multi-GPU train_ddp.py).
#
# Config values are copied verbatim from the official launcher (gamma=0.997,
# num_critic_epochs=500, clip_std_max sweep, idql_eval=True, batch 1024,
# lr 2.83e-4, ...). The ONLY intentional deviation: only_bc=False, because our
# refactored train.py treats only_bc=True as "stop after BC" (the official
# train_ddp.py does not), and we need the offline-RL stage to run.
#
# Two stages:
#   1) one stage-1 pass (bppo_steps=0) -> materialize BC / IQL critic / dynamics
#   2) BPPO sweep over lr x rollout_length x clip_std_max, each reusing the
#      stage-1 artifacts and sharing one global best dir.
#
# Usage: bash scripts/maniskill/train_policy_two_stage.sh [seed]

seed=${1:-42}
task_name=maniskill_pickcube
config_name='rl100_3d_epsilon'
exp_name=${task_name}-rl100-twostage

# Official non-chunk sweep grid (overridable). NOTE: n_action_steps=1 here, so any
# rollout_length is valid (must be a multiple of n_action_steps).
LR_VALUES=${LR_VALUES:-"1e-6 2e-6 4e-6 1e-5"}
ROLLOUT_VALUES=${ROLLOUT_VALUES:-"3 5 10 15 20"}
CLIP_STD_MAX_VALUES=${CLIP_STD_MAX_VALUES:-"0.1 0.8"}
BPPO_STEPS=${BPPO_STEPS:-5000}

RUN_STAGE1=${RUN_STAGE1:-true}
RUN_SWEEP=${RUN_SWEEP:-true}

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_ERROR_REPORTING=false

root_run_dir="$(pwd)/logs/two_stage/${exp_name}_seed${seed}"
stage1_run_dir="${root_run_dir}/stage1"
mkdir -p "${root_run_dir}"

# Common params copied from the official train_policy_two_stage.sh get_common_params.
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
        use_action_embed=True horizon=3 n_action_steps=1 n_obs_steps=3 \
        ft_all_actions=False num_inference_steps=10 \
        policy.model=skipnet policy.encoder_type=dp3vib policy.act=relu \
        policy.encoder_output_dim=64 policy.diffusion_step_embed_dim=256 policy.down_dims='[256,512,1024]' \
        policy.scheduler_type='ddim' policy.use_agent_pos=True policy.use_vib=True policy.use_recon=True \
        policy.mlp_policy_depth=3 policy.beta_kl=1e-5 \
        offline=True only_bc=False online=False distill_phase=null kl_annealing=False \
        unio4.idql_eval=True unio4.eval_times=1 \
        critic.omega=0.7 critic.gamma=0.997 \
        dynamics_type='mlp' dynamics.prediction_mode=full \
        training.num_epochs=600 training.num_critic_epochs=500 dynamics.dynamics_max_epochs=150 \
        dataloader.batch_size=1024 val_dataloader.batch_size=1024 \
        optimizer.lr=2.83e-4 critic.q_lr=2.83e-4 critic.v_lr=2.83e-4 dynamics.dynamics_lr=5.66e-4 \
        task.env_runner.eval_episodes=30 task.env_runner.env_num=1 task.env_runner.seed=${seed}"
}

if [ "${RUN_STAGE1}" = true ]; then
    echo "=== Stage 1: BC + IQL critic + dynamics (bppo_steps=0) ==="
    python RL-100/train.py --config-name="${config_name}.yaml" \
        $(common_params "${stage1_run_dir}" 1e-6 3 0.1) \
        unio4.bppo_steps=0
fi

if [ "${RUN_SWEEP}" = true ]; then
    echo "=== Stage 2: BPPO sweep (reuse stage-1 artifacts, shared global best) ==="
    for lr in ${LR_VALUES}; do
      for rl in ${ROLLOUT_VALUES}; do
        for csm in ${CLIP_STD_MAX_VALUES}; do
            ts=$(date +"%Y%m%d-%H%M%S")
            sweep_dir="${root_run_dir}/sweep/${ts}-lr_${lr}_rollout_${rl}_clip_${csm}"
            mkdir -p "${sweep_dir}"
            echo "--- sweep job: lr=${lr} rollout=${rl} clip_std_max=${csm} ---"
            python RL-100/train.py --config-name="${config_name}.yaml" \
                $(common_params "${sweep_dir}" "${lr}" "${rl}" "${csm}") \
                unio4.bppo_steps="${BPPO_STEPS}" \
                critic.load_pretrain=True \
                +unio4.stage1_resume_dir="${stage1_run_dir}" \
                +unio4.global_best_dir="${root_run_dir}/best"
        done
      done
    done
    echo "=== sweep complete. global best under ${root_run_dir}/best ==="
fi
