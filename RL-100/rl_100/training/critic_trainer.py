import os
import time
import tqdm
import torch
import hydra
import numpy as np
from termcolor import cprint
from rl_100.common.pytorch_util import dict_apply
from rl_100.dataset.base_dataset import BaseDataset
from torch.utils.data import DataLoader
from rl_100.unidpg.dynamics_eval_batch import train_dynamics

def train_critic_and_dynamics(workspace, train_dataloader, val_dataloader, wandb_run, env_runner, device):
    cfg = workspace.cfg
    normalizer = workspace.model.normalizer
    
    workspace.train_dataloader = train_dataloader
    workspace.model.set_critic_normalizer(normalizer)
    workspace.model.to(device)
    
    # for Uni-O4 fine-tuning
    iql, Q_bc, value = workspace.model.initialize_critic(
        device=device,
        q_hidden_dim=cfg.critic.q_hidden_dim,
        q_depth=cfg.critic.q_depth,
        q_lr=cfg.critic.q_lr,
        target_update_freq=cfg.critic.target_update_freq,
        tau=cfg.critic.tau,
        gamma=cfg.critic.gamma,
        v_hidden_dim=cfg.critic.v_hidden_dim,
        v_depth=cfg.critic.v_depth,
        v_lr=cfg.critic.v_lr,
        omega=cfg.critic.omega,
        is_double_q=cfg.critic.is_double_q,
        is_iql=cfg.critic.is_iql,
        is_share_encoder=cfg.critic.is_share_encoder,
        use_action_embed=cfg.use_action_embed,
        fix_encoder=cfg.critic.fix_encoder,
        chunk_as_single_action=cfg.chunk_as_single_action,
        n_action_steps=cfg.n_action_steps,
        use_conv_action_embed=getattr(cfg, 'use_conv_action_embed', False),
        conv_hidden_dims=getattr(cfg, 'conv_hidden_dims', [128, 256]),
        conv_latent_cz=getattr(cfg, 'conv_latent_cz', 32),
        conv_kernel_size=getattr(cfg, 'conv_kernel_size', 5),
        conv_n_groups=getattr(cfg, 'conv_n_groups', 8),
        action_recon_beta=getattr(cfg, 'action_recon_beta', 0.5),
        q_layer_norm=getattr(cfg.critic, 'q_layer_norm', False),
        action_embed_layer_norm=getattr(cfg.critic, 'action_embed_layer_norm', False),
        action_scale_norm=getattr(cfg.critic, 'action_scale_norm', False),
    )
    
    stage1_artifact_dir = workspace.get_stage1_artifact_dir()
    critic_artifact_dir = workspace.get_critic_artifact_dir()
    os.makedirs(critic_artifact_dir, exist_ok=True)
    Q_bc_path = os.path.join(critic_artifact_dir, 'Q_bc_20.pt')
    value_path = os.path.join(critic_artifact_dir, 'value_20.pt')
    
    if cfg.critic.is_iql:
        if cfg.critic.is_share_encoder:
            encoder_path = os.path.join(critic_artifact_dir, 'encoder.pt')
        else:
            encoder_path = None
            
        if os.path.exists(Q_bc_path):
            if cfg.critic.load_pretrain:
                iql.load(Q_bc_path, value_path, encoder_path)
                iql.eval()
                iql.obs_encoder.eval()
                
        if not os.path.exists(Q_bc_path) or cfg.off2off:
            if cfg.offline and cfg.chunk_as_single_action and hasattr(cfg.task, 'critic_dataset'):
                critic_dataset = hydra.utils.instantiate(cfg.task.critic_dataset)
                cprint(f'Critic dataset: {len(critic_dataset)} samples '
                       f'(stride={getattr(critic_dataset, "sequence_stride", 1)})', 'cyan')
                critic_dataloader = DataLoader(critic_dataset, **cfg.dataloader)
            else:
                critic_dataloader = train_dataloader
                
            for local_epoch_idx in range(cfg.training.num_critic_epochs):
                critic_step_log = dict()
                q_train_losses, v_train_losses = list(), list()
                with tqdm.tqdm(critic_dataloader, desc=f"Training epoch {workspace.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        Q_bc_loss, value_loss = iql.update(batch=batch)
                        q_train_losses.append(Q_bc_loss)
                        v_train_losses.append(value_loss)
                q_loss_mean, v_loss_mean = np.mean(q_train_losses), np.mean(v_train_losses)
                print('Step: {}, Q loss: {}, Value loss: {}'.format(local_epoch_idx, q_loss_mean, v_loss_mean))
                wandb_run.log({'Q_loss': q_loss_mean, 'value_loss': v_loss_mean})
            iql.save(Q_bc_path, value_path, encoder_path)
        q_eval = iql.minQ
        
    # load dynamics parameters
    prediction_mode = getattr(cfg.dynamics, 'prediction_mode', 'last')
    if cfg.chunk_as_single_action and prediction_mode != "full":
        raise ValueError(
            "chunk_as_single_action=True requires dynamics.prediction_mode='full'. "
            "A chunk dynamics step advances the whole observation window, so "
            "'last' mode would mix stale observations with the predicted chunk endpoint."
        )
        
    dynamics_encoder = workspace.model.get_dynamics_encoder()
    if cfg.dynamics_type == "diffusion":
        dynamics_path = os.path.join(stage1_artifact_dir, f'saved_models_diffusion_{prediction_mode}')
    else:
        dynamics_path = os.path.join(stage1_artifact_dir, f'saved_models_{prediction_mode}')
        
    # set dynamics parameters
    if prediction_mode == "full" and cfg.n_obs_steps > 1:
        cfg.lddm.encoder_output_dim = workspace.model.obs_feature_dim * cfg.n_obs_steps
    else:
        cfg.lddm.encoder_output_dim = workspace.model.obs_feature_dim
        
    if getattr(cfg, 'use_conv_action_embed', False):
        from rl_100.model.action_ae import ActionChunkEncoder
        conv_encoder = ActionChunkEncoder(
            action_dim=workspace.model.action_dim,
            hidden_dims=list(getattr(cfg, 'conv_hidden_dims', [128, 256])),
            latent_cz=getattr(cfg, 'conv_latent_cz', 32),
            kernel_size=getattr(cfg, 'conv_kernel_size', 5),
            n_groups=getattr(cfg, 'conv_n_groups', 8),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.n_action_steps, workspace.model.action_dim)
            cfg.lddm.action_embed_dim = conv_encoder(dummy).reshape(1, -1).shape[-1]
    elif prediction_mode == "full" and cfg.n_obs_steps > 1:
        cfg.lddm.action_embed_dim = workspace.model.obs_feature_dim
        
    dynamics = train_dynamics(
        env_runner.env, 
        workspace.model.normalizer, 
        dynamics_encoder, 
        dynamics_path, 
        cfg, 
        workspace.model.obs_feature_dim, 
        workspace.model.action_dim,
        chunk_as_single_action=cfg.chunk_as_single_action,
        n_action_steps=cfg.n_action_steps,
        n_obs_steps=cfg.n_obs_steps,
        device=device,
    )
    
    if (not os.path.exists(os.path.join(dynamics_path, "dynamics.pth")) and cfg.offline) or cfg.off2off:
        epoch = 0
        step_log = dict()
        dynamics_losses = list()
        for local_epoch_idx in range(cfg.dynamics.dynamics_max_epochs):
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {epoch}", 
                    leave=False, mininterval=cfg.dynamics.tqdm_interval_sec) as tepoch:
                epoch += 1
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if cfg.chunk_as_single_action:
                        nobs_features = dynamics.obs2latent(batch['obs'])
                        next_nobs_features = dynamics.next_obs2latent(batch['next_obs'])
                        single_nob_features = nobs_features[:, -1, :]
                        single_next_nob_features = next_nobs_features[:, -1, :]
                    else:
                        nobs_features = dynamics.obs2latent(batch['obs'])
                        next_nobs_features = dynamics.obs2latent(batch['next_obs'])
                        single_nob_features = nobs_features[:, -1, :]
                        single_next_nob_features = next_nobs_features[:, -1, :]
                
                    if prediction_mode == "full":
                        batch_size = nobs_features.shape[0]
                        train_nobs = nobs_features.reshape(batch_size, -1)
                        train_next_nobs = next_nobs_features.reshape(batch_size, -1)
                    else:
                        train_nobs = single_nob_features
                        train_next_nobs = single_next_nob_features
                    
                    dynamics_loss = dynamics.learn(batch=batch, nobs_features=train_nobs, next_nobs_features=train_next_nobs)
                    dynamics.optimize(dynamics_loss)
                    dynamics_losses.append(dynamics_loss.item())
            if (local_epoch_idx + 1) % 10 == 0:
                print('dynamics loss: {}'.format(np.array(dynamics_losses[-10:]).mean()))
            
            # Batched validation
            with torch.no_grad():
                val_losses_all = []
                for val_batch in val_dataloader:
                    val_batch = dict_apply(val_batch, lambda x: x.to(device, non_blocking=True))
                    if cfg.chunk_as_single_action:
                        val_nobs_features = dynamics.obs2latent(val_batch['obs'])
                        val_next_nobs_features = dynamics.next_obs2latent(val_batch['next_obs'])
                        val_single_nob_features = val_nobs_features[:, -1, :]
                        val_single_next_nob_features = val_next_nobs_features[:, -1, :]
                    else:
                        val_nobs_features = dynamics.obs2latent(val_batch['obs'])
                        val_next_nobs_features = dynamics.obs2latent(val_batch['next_obs'])
                        val_single_nob_features = val_nobs_features[:, -1, :]
                        val_single_next_nob_features = val_next_nobs_features[:, -1, :]
                    
                    if prediction_mode == "full":
                        batch_size = val_nobs_features.shape[0]
                        val_nobs = val_nobs_features.reshape(batch_size, -1)
                        val_next_nobs = val_next_nobs_features.reshape(batch_size, -1)
                    else:
                        val_nobs = val_single_nob_features
                        val_next_nobs = val_single_next_nob_features
                    
                    val_inputs, val_targets = dynamics.format_samples_for_training(val_batch, val_nobs, val_next_nobs)
                    batch_val_losses = dynamics.validate(val_inputs, val_targets)
                    val_losses_all.append(batch_val_losses)
                
                val_losses_all = np.array(val_losses_all)
                new_holdout_losses = val_losses_all.mean(axis=0).tolist()
                
            dynamics._update_holdout_and_log(new_holdout_losses, np.mean(dynamics_losses), wandb_run, epoch)
            if (dynamics.cnt >= cfg.dynamics.max_epochs_since_update) or (cfg.dynamics.dynamics_max_epochs and (epoch >= cfg.dynamics.dynamics_max_epochs)):
                break
        dynamics.post_well_learned()
        dynamics.save(dynamics_path)
    elif cfg.offline:
        dynamics.load(dynamics_path)

    return iql, Q_bc, value, dynamics, critic_artifact_dir, Q_bc_path, value_path
