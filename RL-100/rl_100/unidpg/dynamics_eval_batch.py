
import random

import gym
# import d4rl
#import neorl
from typing import Dict, Union, Tuple
from copy import deepcopy
from collections import defaultdict
import numpy as np
import torch
import hydra

import os
from rl_100.unidpg.transition_model.models.dynamics_model import EnsembleDynamicsModel
from rl_100.unidpg.transition_model.models.ensemble_diffusion_dynamics import EnsembleDiffusionDynamicsModel
from rl_100.unidpg.transition_model.dynamics import EnsembleDynamics_batch
from rl_100.unidpg.transition_model.utils.scaler import StandardScaler
from rl_100.unidpg.transition_model.utils.termination_fns import get_termination_fn
from rl_100.unidpg.transition_model.utils.load_dataset import qlearning_dataset
from rl_100.unidpg.transition_model.utils.buffer_ import ReplayBuffer
from rl_100.unidpg.transition_model.utils.logger import Logger, make_log_dirs




def rollout(
        policy,
        dynamics,
        Q,
        iql,
        init_obss: np.ndarray,
        rollout_length: int,
        args,
        mean,
        std
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        if args.is_iql:
            q_eval = iql.minQ
        else:
            q_eval = Q
        num_transitions = 0
        rewards_arr = np.array([])
        total_q = np.array([])
        rollout_transitions = defaultdict(list)
        # rollout
        observations = init_obss
        length = 0
        discount_return, discount = 0, 1
        for _ in range(rollout_length):
            
            if not args.is_eval_state_norm:
                if args.is_state_norm:
                    s = (observations - torch.FloatTensor(mean).to(args.device)) / torch.FloatTensor(std).to(args.device)
                else:
                    s = observations
            else:
                s = observations

            actions = policy.sample_action(s, get_np = False)

            Q_value = q_eval(s, actions)
            next_observations, rewards, terminals, info, discount_return, discount = dynamics.step(observations.cpu().data.numpy(), actions.cpu().data.numpy(), discount_return, discount)

            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())
            total_q = np.append(total_q, Q_value.cpu().data.numpy().flatten())
            nonterm_mask = (~terminals).flatten()
            length += 1
            if nonterm_mask.sum() == 0:
                print('terminal length: {}'.format(length))
                break

            observations = torch.FloatTensor(next_observations[nonterm_mask]).to(args.device)

        return total_q.mean(), rewards_arr.mean()
    
def train_dynamics(env, normalizer, dynamics_encoder, dynamics_save_path, cfg, feature_dim, action_dim, chunk_as_single_action=False, n_action_steps=1, n_obs_steps=1, device='cuda'):
    """
    Args:
        prediction_mode: 从 cfg.dynamics.prediction_mode 获取
            - "last": last obs prediction
            - "full": whole obss window prediction
    """
    # create dynamics
    prediction_mode = getattr(cfg.dynamics, 'prediction_mode', 'last')
    
    if chunk_as_single_action:
        action_dim = action_dim * n_action_steps
    
    if prediction_mode == "full":
        output_obs_dim = feature_dim * n_obs_steps
        print(f'==========================Dynamics prediction mode: FULL (output dim: {output_obs_dim})==========================')
    else:  # "last"
        output_obs_dim = feature_dim
        print(f'==========================Dynamics prediction mode: LAST (output dim: {output_obs_dim})==========================')
    
    if cfg.dynamics_type == 'diffusion':
        # Instantiate LDDM and wrap it with EnsembleDiffusionDynamicsModel
        lddm_model = hydra.utils.instantiate(cfg.lddm)
        lddm_model.set_device(device)
        
        # Use ensemble wrapper for compatibility with EnsembleDynamics_batch
        use_true_ensemble = getattr(cfg.dynamics, 'use_true_ensemble', False)
        dynamics_model = EnsembleDiffusionDynamicsModel(
            lddm_model=lddm_model,
            obs_dim=output_obs_dim,
            action_dim=action_dim,
            num_ensemble=cfg.dynamics.n_ensemble,
            num_elites=cfg.dynamics.n_elites,
            with_reward=cfg.predict_r,
            device=device,
            use_true_ensemble=use_true_ensemble,
            cfg=cfg,
        )
        print(f'==========================Diffusion Dynamics: ensemble={cfg.dynamics.n_ensemble}, true_ensemble={use_true_ensemble}==========================')
    else:
        dynamics_model = EnsembleDynamicsModel(
            obs_dim=output_obs_dim,
            action_dim=action_dim,
            hidden_dims=cfg.dynamics.dynamics_hidden_dims,
            num_ensemble=cfg.dynamics.n_ensemble,
            num_elites=cfg.dynamics.n_elites,
            weight_decays=cfg.dynamics.dynamics_weight_decay,
            device=device,
            cfg=cfg,
            with_reward=cfg.predict_r,
        )

    if not cfg.dynamics.fix_encoder:
        dynamics_optim = hydra.utils.instantiate(
            cfg.optimizer, params=list(dynamics_model.parameters()) + list(dynamics_encoder.parameters()))
    else:   
        print('==========================fix encoder==========================')
        # for param in dynamics_encoder.parameters():
        #     param.requires_grad = False
        dynamics_optim = hydra.utils.instantiate(
            cfg.optimizer, params=dynamics_model.parameters())
        # dynamics_optim = torch.optim.Adam(
        #     dynamics_model.parameters(),
        #     lr=cfg.dynamics.dynamics_lr
        # )

    termination_fn = get_termination_fn(task=cfg.task_name)
    dynamics = EnsembleDynamics_batch(
        dynamics_model,
        dynamics_optim,
        termination_fn,
        env,
        normalizer,
        dynamics_encoder,
        cfg=cfg,
        action_dim=action_dim,
        gamma=cfg.critic.gamma,
        device=device,
        chunk_as_single_action=chunk_as_single_action,
        n_action_steps=n_action_steps,
        prediction_mode=prediction_mode,
    )

    os.makedirs(dynamics_save_path, exist_ok=True)
    log_dirs = make_log_dirs(
        cfg.task_name, cfg.name, cfg.training.seed, None,
        record_params=None, #["penalty_coef", "rollout_length"]
        root_dir=os.path.join(dynamics_save_path, 'dynamics_logs'),
    )
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboardX"
    }
    logger = Logger(log_dirs, output_config)
    dynamics.set_logger(logger)
    # logger.log_hyperparameters(vars(cfg.dynamics))
    # dynamics.train(
    #     critic_dataset,
    #     max_epochs_since_update=cfg.dynamics.max_epochs_since_update,
    #     max_epochs=cfg.dynamics.dynamics_max_epochs
    # )
    # dynamics.save(dynamics_save_path)

    return dynamics

def dynamics_eval(args, policy, Q, iql, dynamics, replay_buffer, env, mean = 0., std = 1.):
    s, _, _, _, _, _, _, _ = replay_buffer.sample(args.rollout_batch_size)
    #s = replay_buffer.sample_aug_state(args.rollout_batch_size)
    Q_mean, reward_mean = rollout(policy, dynamics, Q, iql, s, args.rollout_length, args, mean, std)
    return Q_mean, reward_mean

def get_args():
    from transition_model.configs import loaded_args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mobile")
    parser.add_argument("--env", type=str, default="walker2d-medium-expert-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--is_state_norm", default=False, type=bool)
    parser.add_argument("--is_eval_state_norm", default=False, type=bool)
    known_args, _ = parser.parse_known_args()
    default_args = loaded_args[known_args.env]
    for arg_key, default_value in default_args.items():
        parser.add_argument(f'--{arg_key}', default=default_value, type=type(default_value))

    return parser.parse_args()
if __name__ == "__main__":
    from buffer import OfflineReplayBuffer
    args = get_args()
    env = gym.make(args.env)

    # seed
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # dim of state and action
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # device
    args.device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"


    # offline dataset to replay buffer
    dataset = env.get_dataset()
    replay_buffer = OfflineReplayBuffer(args.device, state_dim, action_dim, len(dataset['actions']) - 1, percentage=1)
    replay_buffer.load_dataset(dataset=dataset)
    replay_buffer.compute_return(args.gamma)



    if args.is_state_norm:
        mean, std = replay_buffer.normalize_state()
    else:
        mean, std = 0., 1.
    replay_buffer.augmentaion()

    train_dynamics(args, env, replay_buffer)
