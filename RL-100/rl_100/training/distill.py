import os
import time
import tqdm
import torch
import numpy as np
from copy import deepcopy
import hydra
from rl_100.common.pytorch_util import dict_apply
from rl_100.common.checkpoint_util import TopKCheckpointManager
from rl_100.model.diffusion.ema_model import EMAModel
from rl_100.model.common.lr_scheduler import get_scheduler
from rl_100.model.common.cm_util import update_ema
from termcolor import cprint

def distill_cm(workspace, train_dataloader, val_dataloader, wandb_run, env_runner, phase: str = 'after_dp'):
    cprint('start distill to cm {}'.format(phase), 'green')
    cfg = workspace.cfg
    
    if phase == 'after_dp':
        model_to_optimize = workspace.model
        latest_cm_path = workspace.get_checkpoint_path(tag='latest_cm')
    elif phase == 'after_offline':
        model_to_optimize = workspace.unio4._policy
        latest_cm_path = os.path.join(workspace.offline_best_path, 'last', 'distilled_model.pt')
    else:
        raise RuntimeError(f"Unsupported distill phase: {phase}")

    if cfg.training.resume and os.path.exists(latest_cm_path):
        cprint(f'resume=True and found existing distill checkpoint at {latest_cm_path}; skip distill2cm', 'yellow')
        return

    model_to_optimize.set_target()
    cm_optimizer = torch.optim.AdamW(
        model_to_optimize.distilled_model.parameters(),
        lr=cfg.optimizer.lr,
        betas=(cfg.optimizer.betas[0], cfg.optimizer.betas[1]),
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.eps)

    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=cm_optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=(
            len(train_dataloader) * cfg.training.num_epochs) \
                // cfg.training.gradient_accumulate_every
    )
    
    topk_manager = TopKCheckpointManager(
        save_dir=os.path.join(workspace.output_dir, 'checkpoints'),
        **cfg.checkpoint.topk
    )
    workspace.global_step = 0
    workspace.epoch = 0
    train_sampling_batch = None
    ema_model = deepcopy(model_to_optimize)
    ema = None
    if cfg.training.use_ema:
        ema = hydra.utils.instantiate(
            cfg.ema,
            model=ema_model)
            
    if not os.path.exists(latest_cm_path):
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {workspace.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    batch = dict_apply(batch, lambda x: x.to(workspace.device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    t1_1 = time.time()
                    if getattr(model_to_optimize, 'is_flow', False):
                        raw_loss, loss_dict = model_to_optimize.compute_flow_distill_loss(
                            batch, distill2mean=cfg.distill2mean)
                    elif cfg.distill_loss_type == 'back_up':
                        raw_loss, loss_dict = model_to_optimize.compute_ddim2cm_loss(batch, distill2mean=cfg.distill2mean)
                    elif cfg.distill_loss_type == 'action':
                        raw_loss, loss_dict = model_to_optimize.compute_ddim2cm_loss_action(batch, distill2mean=cfg.distill2mean)
                    elif cfg.distill_loss_type == 'action_same_noise':
                        raw_loss, loss_dict = model_to_optimize.compute_ddim2cm_loss_action_same_noise(batch, distill2mean=cfg.distill2mean)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    t1_2 = time.time()

                    if workspace.global_step % cfg.training.gradient_accumulate_every == 0:
                        torch.nn.utils.clip_grad_norm_(model_to_optimize.distilled_model.parameters(), cfg.training.max_grad_norm)
                        cm_optimizer.step()
                        lr_scheduler.step()
                        cm_optimizer.zero_grad(set_to_none=True)
                        if not getattr(model_to_optimize, 'is_flow', False):
                            update_ema(model_to_optimize.target_model.parameters(), model_to_optimize.distilled_model.parameters(), cfg.training.ema_decay)
                    t1_3 = time.time()
                    if cfg.training.use_ema and ema is not None:
                        ema.step(model_to_optimize)
                    t1_4 = time.time()
                    
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': workspace.global_step,
                        'epoch': workspace.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
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

            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            policy = model_to_optimize
            policy.eval()

            if (workspace.epoch % cfg.training.rollout_every) == 0 and workspace.RUN_ROLLOUT and env_runner is not None:
                runner_log = env_runner.run(policy, use_cm=True, distill2mean=cfg.distill2mean)
                step_log.update(runner_log)  
            if (workspace.epoch % cfg.training.val_every) == 0 and workspace.RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {workspace.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(workspace.device, non_blocking=True))
                            loss, loss_dict = model_to_optimize.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        step_log['val_loss'] = val_loss

            if (workspace.epoch % cfg.training.sample_every) == 0 and train_sampling_batch is not None:
                with torch.no_grad():
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(workspace.device, non_blocking=True))
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
                
            if (workspace.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                if phase == 'after_dp':
                    if cfg.checkpoint.save_last_ckpt:
                        workspace.save_checkpoint(tag='latest_cm')
                    if cfg.checkpoint.save_last_snapshot:
                        workspace.save_snapshot(tag='latest_cm')

                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                        metric_dict['type'] = 'cm'
                    
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        workspace.save_checkpoint(path=topk_ckpt_path)
                    if cfg.only_bc:
                        workspace.unio4.set_policy(workspace.model)
                        workspace.unio4.set_old_policy()
                        os.makedirs(os.path.join(workspace.output_dir, 'best_cm'), exist_ok=True)
                        workspace.unio4.save(os.path.join(workspace.output_dir, 'best_cm'))
                os.makedirs(os.path.join(workspace.offline_best_path, '_{}'.format(str(workspace.epoch))), exist_ok=True)
                model_to_optimize.save(os.path.join(os.path.join(workspace.offline_best_path, '_{}'.format(str(workspace.epoch)))))
                os.makedirs(os.path.join(workspace.offline_best_path, 'last'), exist_ok=True)
                model_to_optimize.save(os.path.join(workspace.offline_best_path, 'last'))

            policy.train()
            wandb_run.log(step_log, step=workspace.global_step)
            workspace.global_step += 1
            workspace.epoch += 1
            del step_log
        
        if phase == 'after_dp':
            workspace.save_checkpoint(tag='latest_cm')
        os.makedirs(os.path.join(workspace.offline_best_path, 'last'), exist_ok=True)
        model_to_optimize.save(os.path.join(workspace.offline_best_path, 'last'))
        
    if getattr(model_to_optimize, 'is_flow', False):
        model_to_optimize.promote_distilled_model()
        os.makedirs(os.path.join(workspace.offline_best_path, 'last'), exist_ok=True)
        model_to_optimize.save(os.path.join(workspace.offline_best_path, 'last'))
        cprint('saved promoted student checkpoint to {}'.format(
            os.path.join(workspace.offline_best_path, 'last')), 'green')
