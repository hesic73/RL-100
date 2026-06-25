import os
import tqdm
import wandb
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

def offline_finetune(workspace, dynamics, Q, value, iql):
    cfg = workspace.cfg
    ema = workspace.ema
    policy_refs = [workspace.model, workspace.unio4._policy, workspace.unio4._old_policy]
    aug_restore = []
    disabled_aug = False
    offline_use_aug = getattr(cfg, 'offline_use_aug', False)
    
    if not offline_use_aug:
        for policy_ref in policy_refs:
            if hasattr(policy_ref, 'module'):
                policy_ref = policy_ref.module
            if hasattr(policy_ref, 'use_aug'):
                aug_restore.append((policy_ref, policy_ref.use_aug))
                if policy_ref.use_aug:
                    policy_ref.use_aug = False
                    disabled_aug = True
        if disabled_aug:
            print('Disabled image augmentation for offline RL finetuning stage')

    # Create a new dataloader with configurable batch size for finetuning
    if cfg.offline and cfg.chunk_as_single_action and hasattr(cfg.task, 'finetune_dataset'):
        from hydra.utils import instantiate
        finetune_dataset = instantiate(cfg.task.finetune_dataset)
        print(f'Finetune dataset: {len(finetune_dataset)} samples (stride={getattr(finetune_dataset, "sequence_stride", 1)})')
    else:
        finetune_dataset = workspace.dataset

    finetune_batch_size = getattr(cfg.unio4, 'finetune_batch_size', cfg.dataloader.batch_size)
    finetune_dataloader_cfg = OmegaConf.to_container(cfg.dataloader)
    finetune_dataloader_cfg['batch_size'] = finetune_batch_size
    finetune_dataloader_cfg['pin_memory'] = False
    finetune_dataloader_cfg['persistent_workers'] = False
    finetune_dataloader_cfg['num_workers'] = min(finetune_dataloader_cfg.get('num_workers', 8), 2)
    
    workspace.finetune_dataloader = DataLoader(finetune_dataset, **finetune_dataloader_cfg)
    workspace.finetune_dataloader_iter = iter(workspace.finetune_dataloader)
    print(f'Finetuning with batch size: {finetune_batch_size}')
    
    workspace.unio4.set_old_policy()
    if cfg.unio4.fix_encoder:
        workspace.unio4._policy.obs_encoder.eval()
        workspace.unio4._old_policy.obs_encoder.eval()
        
    best_bppo_path = workspace.unio4_output_dir
    os.makedirs(best_bppo_path, exist_ok=True)
    best_saved_scores = float('-inf')
    run_idql_eval = bool(cfg.unio4.idql_eval)
    run_ema_eval = bool(cfg.training.use_ema and workspace.ema_model is not None)
    
    if run_idql_eval:
        idql_log_data = workspace.unio4_eval(
            idql_eval=True,
            dynamics=dynamics,
            first_action=cfg.unio4.first_action,
            get_np=True,
            iql=iql,
            Q=Q,
            repeat_num=128,
            eval_times=cfg.unio4.eval_times,
            eval_name='IDQL Eval',
        )
    else:
        idql_log_data = None
        
    if run_ema_eval:
        ema_log_data = workspace.eval(
            eval_times=cfg.unio4.eval_times,
            policy_override=workspace.ema_model,
            eval_name='EMA Eval',
        )
        _, is_updated_ema = workspace.maybe_update_global_best_ema(ema_log_data['test_mean_score'])
        if is_updated_ema:
            print('------------saved best EMA model----------------')
    else:
        ema_log_data = None
        
    normal_log_data = workspace.eval(
        eval_times=cfg.unio4.eval_times,
        policy_override=workspace.unio4._policy,
        eval_name='Policy Eval',
    )
    
    best_bppo_scores = normal_log_data['test_mean_score']
    best_saved_scores, is_updated = workspace.maybe_update_global_best(best_bppo_scores)
    if is_updated:
        print('------------saved best model----------------')
        
    best_mean_qs = dynamics.rollout(
        workspace.unio4._policy, 
        Q, 
        iql, 
        workspace.sample_finetune_batch(), 
        rollout_length=cfg.unio4.rollout_length, 
        is_iql=cfg.critic.is_iql,
        use_gae=cfg.unio4.use_gae,
        first_action=cfg.dynamics.first_action,
    )
    print('rollout trajectory q mean:{}'.format(best_mean_qs))
    
    update_num = 0
    scores, opes = [], []
    idql_scores, ema_scores, normal_scores = [], [], []
    scores.append(best_bppo_scores)
    if run_idql_eval:
        idql_scores.append(idql_log_data['test_mean_score'])
    if run_ema_eval:
        ema_scores.append(ema_log_data['test_mean_score'])
    normal_scores.append(normal_log_data['test_mean_score'])
    opes.append(best_mean_qs[0].detach().cpu().numpy())
    
    init_log_data = {
        'current_bppo_scores': best_bppo_scores, 
        'current_mean_qs': best_mean_qs,
        'normal_eval_scores': normal_log_data['test_mean_score']
    }
    if run_ema_eval:
        init_log_data['ema_eval_scores'] = ema_log_data['test_mean_score']
    if run_idql_eval:
        init_log_data['idql_eval_scores'] = idql_log_data['test_mean_score']
    wandb.log(init_log_data)
    
    for step in tqdm.tqdm(range(int(cfg.unio4.bppo_steps)), desc='bppo updating ......'):
        if cfg.unio4.is_linear_decay:
            bppo_lr_now = cfg.unio4.bppo_lr * (1 - step / cfg.unio4.bppo_steps)
            clip_ratio_now = cfg.unio4.clip_ratio * (1 - step / cfg.unio4.bppo_steps)
        else:
            bppo_lr_now = None
            clip_ratio_now = None
            
        if step > 200:
            cfg.unio4.is_clip_decay = False
            cfg.unio4.is_bppo_lr_decay = False
            
        losses = workspace.unio4.update_distribution(
            batch=workspace.sample_finetune_batch(),
            value=value,
            Q=Q,
            iql=iql,
            is_clip_decay=cfg.unio4.is_clip_decay,
            is_lr_decay=cfg.unio4.is_bppo_lr_decay,
            is_linear_decay=cfg.unio4.is_linear_decay,
            bppo_lr_now=bppo_lr_now,
            clip_ratio_now=clip_ratio_now,
            is_bc_loss=getattr(cfg.unio4, 'is_bc_loss', False),
            dynamics=dynamics,
            use_gae=cfg.unio4.use_gae,
            fix_encoder=cfg.unio4.fix_encoder,
            final_reward=cfg.unio4.final_reward,
            gamma=cfg.critic.gamma,
            lamda=cfg.ppo.lamda,
        )
        
        if cfg.training.use_ema and workspace.ema is not None:
            workspace.ema.step(workspace.unio4._policy)
            
        wandb.log({'dpg_loss': losses})
        
        if (step+1) % cfg.unio4.eval_freq == 0:
            if run_idql_eval:
                idql_log_data = workspace.unio4_eval(
                    idql_eval=True,
                    dynamics=dynamics,
                    first_action=cfg.unio4.first_action,
                    get_np=True,
                    iql=iql,
                    Q=Q,
                    repeat_num=128,
                    eval_times=cfg.unio4.eval_times,
                    eval_name='IDQL Eval',
                )
                idql_current_scores = idql_log_data['test_mean_score']
                idql_scores.append(idql_current_scores)

            if run_ema_eval:
                ema_log_data = workspace.eval(
                    eval_times=cfg.unio4.eval_times,
                    policy_override=workspace.ema_model,
                    eval_name='EMA Eval',
                )
                ema_current_scores = ema_log_data['test_mean_score']
                ema_scores.append(ema_current_scores)
                _, is_updated_ema = workspace.maybe_update_global_best_ema(ema_current_scores)
                if is_updated_ema:
                    print('------------saved best EMA model----------------')
            else:
                ema_current_scores = None
                
            normal_log_data = workspace.eval(
                eval_times=cfg.unio4.eval_times,
                policy_override=workspace.unio4._policy,
                eval_name='Policy Eval',
            )
            normal_current_scores = normal_log_data['test_mean_score']
            normal_scores.append(normal_current_scores)
            
            current_bppo_scores = normal_current_scores
            best_saved_scores, is_updated = workspace.maybe_update_global_best(current_bppo_scores)
            if is_updated:
                print('------------saved best model----------------')
            else:
                os.makedirs(os.path.join(best_bppo_path, 'score_{}'.format(step)), exist_ok=True)
                workspace.unio4.save(os.path.join(best_bppo_path, 'score_{}'.format(step)))
                print('------------saved {} model----------------'.format(current_bppo_scores))
            scores.append(current_bppo_scores)
            
            score_parts = [f"Step: {step}"]
            if run_idql_eval:
                score_parts.append(f"IDQL Score: {idql_current_scores}")
            if run_ema_eval:
                score_parts.append(f"EMA Score: {ema_current_scores}")
            score_parts.append(f"Normal Score: {normal_current_scores}")
            score_parts.append(f"Selected Score: {current_bppo_scores}")
            print(", ".join(score_parts))
            
            eval_log_data = {
                'current_bppo_scores': current_bppo_scores,
                'normal_eval_scores': normal_current_scores
            }
            if run_ema_eval:
                eval_log_data['ema_eval_scores'] = ema_current_scores
            if run_idql_eval:
                eval_log_data['idql_eval_scores'] = idql_current_scores
            wandb.log(eval_log_data)
            
        if (step+1) % cfg.unio4.eval_step == 0:
            current_mean_qs = dynamics.rollout(
                workspace.unio4._policy, 
                Q, 
                iql, 
                workspace.sample_finetune_batch(), 
                rollout_length=cfg.unio4.rollout_length, 
                is_iql=cfg.critic.is_iql,
                use_gae=cfg.unio4.use_gae,
                first_action=cfg.dynamics.first_action,
            )
            wandb.log({'current_mean_qs': current_mean_qs})
            print('rollout trajectory q mean:{}'.format(current_mean_qs))
            print(f"Step: {step}, Loss: ", losses)
            if cfg.unio4.is_update_old_policy:
                if current_mean_qs > best_mean_qs:
                    best_mean_qs = current_mean_qs
                    workspace.unio4.set_old_policy()
                    print('------------------------------update behavior policy----------------------------------------')
            opes.append(current_mean_qs[0].detach().cpu().numpy())
            
        np.savetxt(os.path.join(best_bppo_path, 'each_ope_score.csv'), opes, fmt='%f', delimiter=',') 
        np.savetxt(os.path.join(best_bppo_path, 'each_scores.csv'), scores, fmt='%f', delimiter=',')
        if run_idql_eval and len(idql_scores) > 0:
            np.savetxt(os.path.join(best_bppo_path, 'each_idql_eval_scores.csv'), idql_scores, fmt='%f', delimiter=',')
        if run_ema_eval and len(ema_scores) > 0:
            np.savetxt(os.path.join(best_bppo_path, 'each_ema_eval_scores.csv'), ema_scores, fmt='%f', delimiter=',')
        if len(normal_scores) > 0:
            np.savetxt(os.path.join(best_bppo_path, 'each_normal_eval_scores.csv'), normal_scores, fmt='%f', delimiter=',')
            
    np.savetxt(os.path.join(best_bppo_path, 'last_ope_score.csv'), opes, fmt='%f', delimiter=',')
    if run_idql_eval and len(idql_scores) > 0:
        np.savetxt(os.path.join(best_bppo_path, 'last_idql_eval_scores.csv'), idql_scores, fmt='%f', delimiter=',')
    if run_ema_eval and len(ema_scores) > 0:
        np.savetxt(os.path.join(best_bppo_path, 'last_ema_eval_scores.csv'), ema_scores, fmt='%f', delimiter=',')
    if len(normal_scores) > 0:
        np.savetxt(os.path.join(best_bppo_path, 'last_normal_eval_scores.csv'), normal_scores, fmt='%f', delimiter=',')
        
    os.makedirs(os.path.join(workspace.output_dir, 'last'), exist_ok=True)
    workspace.unio4.save(os.path.join(workspace.output_dir, 'last'))
    
    for policy_ref, use_aug in aug_restore:
        policy_ref.use_aug = use_aug
    if disabled_aug:
        print('Restored image augmentation setting after offline RL finetuning stage')
    workspace.unio4.flush_ratio_logs(force=True)
