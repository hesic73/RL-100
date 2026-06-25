import os
import time
import tqdm
import torch
import numpy as np
from rl_100.common.pytorch_util import dict_apply

def train_bc(workspace, train_dataloader, val_dataloader, wandb_run, env_runner, device):
    cfg = workspace.cfg
    latest_path = workspace.get_stage1_checkpoint_path(tag='latest')
    if not os.path.exists(latest_path) or cfg.training.resume == False or (cfg.off2off and not cfg.off2off_no_bc):
        # VIB module beta kl annealing
        if hasattr(workspace.model.obs_encoder, 'beta_kl'):
            target_beta_kl = workspace.model.obs_encoder.beta_kl
        
        train_sampling_batch = None
        for local_epoch_idx in range(cfg.training.num_epochs):
            # KL annealing
            if cfg.kl_annealing and hasattr(workspace.model.obs_encoder, 'beta_kl'):
                progress = local_epoch_idx / max(cfg.training.num_epochs - 1, 1)
                workspace.model.obs_encoder.beta_kl = target_beta_kl * progress
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {workspace.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = workspace.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    t1_2 = time.time()

                    # step optimizer
                    if workspace.global_step % cfg.training.gradient_accumulate_every == 0:
                        workspace.optimizer.step()
                        workspace.optimizer.zero_grad()
                        workspace.lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema and workspace.ema is not None:
                        workspace.ema.step(workspace.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': workspace.global_step,
                        'epoch': workspace.epoch,
                        'lr': workspace.lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if workspace.verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        wandb_run.log(step_log, step=workspace.global_step)
                        workspace.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.eval()

            # run rollout
            if (workspace.epoch % cfg.training.rollout_every) == 0 and workspace.RUN_ROLLOUT and env_runner is not None:
                setattr(env_runner, "current_epoch", workspace.epoch)
                runner_log = env_runner.run(policy)
                step_log.update(runner_log)
            # run validation
            if (workspace.epoch % cfg.training.val_every) == 0 and workspace.RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {workspace.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = workspace.model.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (workspace.epoch % cfg.training.sample_every) == 0 and train_sampling_batch is not None:
                with torch.no_grad():
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    if cfg.no_pre_action:
                        gt_action = gt_action[:, cfg.n_obs_steps - 1 :]
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()

            if env_runner is None:
                step_log['test_mean_score'] = - train_loss
                
            # checkpoint
            if (workspace.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                if cfg.checkpoint.save_last_ckpt:
                    workspace.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    workspace.save_snapshot()

                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                
                topk_ckpt_path = workspace.topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None:
                    workspace.save_checkpoint(path=topk_ckpt_path)
                if cfg.only_bc:
                    workspace.unio4.set_policy(workspace.model)
                    workspace.unio4.set_old_policy()
                    os.makedirs(os.path.join(workspace.output_dir, 'bc'), exist_ok=True)
                    workspace.unio4.save(os.path.join(workspace.output_dir, 'bc'))
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            wandb_run.log(step_log, step=workspace.global_step)
            workspace.global_step += 1
            workspace.epoch += 1
            del step_log
