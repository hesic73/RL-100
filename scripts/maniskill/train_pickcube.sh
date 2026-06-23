#!/usr/bin/env bash
# Offline BC training on ManiSkill PickCube (3D point-cloud policy).
# Run from the repo root: bash scripts/maniskill/train_pickcube.sh
# No arguments -- edit the settings below.
set -e

# ===== edit these =====
exp_info=run0          # experiment name (appears in the log dir / wandb)
seed=42
num_epochs=400
eval_episodes=16       # episodes per eval
rollout_every=100       # eval + checkpoint every this many epochs
# ======================

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=online
export WANDB_ERROR_REPORTING=false   # disable wandb's sentry telemetry (silences its DeprecationWarning)

python RL-100/train.py --config-name=rl100_3d_epsilon.yaml \
    task=maniskill_pickcube \
    hydra.run.dir=$(pwd)/logs/maniskill_pickcube_${exp_info}_seed${seed} \
    exp_name=maniskill_pickcube_${exp_info} \
    logging.name=maniskill_pickcube_${exp_info}_seed${seed} \
    training.seed=${seed} \
    training.device="cuda:0" \
    training.resume=False \
    training.num_epochs=${num_epochs} \
    training.num_critic_epochs=400 \
    training.rollout_every=${rollout_every} \
    training.checkpoint_every=${rollout_every} \
    logging.mode=online \
    use_wandb=True \
    wandb=True \
    checkpoint.save_ckpt=True \
    horizon=5 \
    n_action_steps=4 \
    n_obs_steps=2 \
    ft_all_actions=False \
    use_action_embed=True \
    num_inference_steps=10 \
    policy.model=skipnet \
    policy.encoder_type=dp3vib \
    policy.act=relu \
    policy.encoder_output_dim=64 \
    policy.diffusion_step_embed_dim=256 \
    policy.down_dims="[256,512,1024]" \
    policy.ddim_noise_scheduler.num_train_timesteps=100 \
    policy.cm_noise_scheduler.num_train_timesteps=50 \
    policy.scheduler_type='ddim' \
    policy.use_agent_pos=True \
    policy.use_vib=True \
    policy.use_recon=True \
    policy.mlp_policy_depth=3 \
    policy.beta_kl=1e-5 \
    offline=True \
    only_bc=True \
    unio4.idql_eval=False \
    critic.omega=0.7 \
    critic.gamma=0.99 \
    dynamics_type='mlp' \
    dynamics.dynamics_max_epochs=150 \
    distill_phase=null \
    kl_annealing=False \
    task.env_runner.eval_episodes=${eval_episodes} \
    task.env_runner.env_num=1
