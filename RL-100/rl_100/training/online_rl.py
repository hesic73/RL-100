import os
import time
import tqdm
import torch
import random
import copy
from collections import deque
import numpy as np
from termcolor import cprint
from rl_100.common.pytorch_util import dict_apply
from rl_100.unidpg.online_buffer import ReplayBuffer, IqlBuffer
from rl_100.unidpg.online_buffer_vec import ReplayBuffer as VecReplayBuffer
from rl_100.unidpg.uni_ppo import compute_gae_per_env

def _iqlft_snapshot_rng():
    snap = {
        "torch_cpu": torch.random.get_rng_state().clone(),
        "numpy": np.random.get_state(legacy=True),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        snap["cuda_all"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return snap

def _iqlft_restore_rng(snap):
    torch.random.set_rng_state(snap["torch_cpu"])
    if torch.cuda.is_available() and "cuda_all" in snap:
        torch.cuda.set_rng_state_all(snap["cuda_all"])
    np.random.set_state(snap["numpy"])
    random.setstate(snap["random"])

def _prepare_offline_iql_batch_for_online(workspace, offline_batch):
    cfg = workspace.cfg
    start = cfg.n_obs_steps - 1

    if getattr(cfg, 'chunk_as_single_action', False):
        end = start + cfg.n_action_steps
        action_len = offline_batch['action'].shape[1]
        if action_len < end:
            raise ValueError(
                f"offline IQL batch action horizon {action_len} is shorter "
                f"than required chunk slice [{start}:{end}]")

        offline_batch['obs'] = dict_apply(
            offline_batch['obs'],
            lambda x: x[:, :cfg.n_obs_steps])
        offline_batch['next_obs'] = dict_apply(
            offline_batch['next_obs'],
            lambda x: x[:, -cfg.n_obs_steps:])

        offline_batch['action'] = offline_batch['action'][:, start:end]
        if cfg.action_norm:
            offline_batch['action'] = workspace.model.normalizer['action'].normalize(
                offline_batch['action'])

        reward_chunk = offline_batch['reward'][:, start:end]
        if reward_chunk.shape[-1] == 1:
            reward_chunk = reward_chunk.squeeze(-1)
        gamma = float(getattr(cfg, 'gamma', cfg.critic.gamma))
        gamma_weights = torch.pow(
            torch.tensor(gamma, device=reward_chunk.device, dtype=reward_chunk.dtype),
            torch.arange(
                cfg.n_action_steps,
                device=reward_chunk.device,
                dtype=reward_chunk.dtype,
            ),
        )
        offline_batch['reward'] = (
            reward_chunk * gamma_weights.reshape(1, -1)
        ).sum(dim=1).reshape(-1, 1, 1)
        offline_batch['not_done'] = offline_batch['not_done'][:, end - 1:end]
        return offline_batch

    offline_batch['obs'] = dict_apply(
        offline_batch['obs'], lambda x: x[:, :cfg.n_obs_steps])
    offline_batch['next_obs'] = dict_apply(
        offline_batch['next_obs'], lambda x: x[:, :cfg.n_obs_steps])
    offline_batch['action'] = offline_batch['action'][:, start:]
    offline_batch['reward'] = offline_batch['reward'][:, start:]
    offline_batch['not_done'] = offline_batch['not_done'][:, start:]
    if cfg.action_norm:
        offline_batch['action'] = workspace.model.normalizer['action'].normalize(
            offline_batch['action'])
    return offline_batch

def _next_offline_iql_batch_for_online(workspace):
    offline_iter = getattr(workspace, '_online_iql_offline_iter', None)
    if offline_iter is None:
        offline_iter = iter(workspace.train_dataloader)
        workspace._online_iql_offline_iter = offline_iter

    try:
        offline_batch = next(offline_iter)
    except StopIteration:
        offline_iter = iter(workspace.train_dataloader)
        workspace._online_iql_offline_iter = offline_iter
        offline_batch = next(offline_iter)

    offline_batch = dict_apply(
        offline_batch,
        lambda x: x.to(workspace.device, non_blocking=True))
    return _prepare_offline_iql_batch_for_online(workspace, offline_batch)

def online_finetune(workspace, dynamics, Q, value, iql, iql_online, copy_encoder, wandb, ema):
    cfg = workspace.cfg
    device = workspace.device
    use_vec_env = getattr(cfg.ppo, 'use_vec_env_online', False)
    enable_force_stochastic = getattr(cfg.ppo, 'force_stochastic_online', True)

    def _set_force_stochastic(encoder, val):
        if hasattr(encoder, 'force_stochastic'):
            encoder.force_stochastic = val

    def _set_iql_deterministic(iql_ref):
        if iql_ref is None:
            return
        encoders = [getattr(iql_ref, 'obs_encoder', None)]
        for net in [iql_ref._Q, iql_ref._target_Q, iql_ref._value]:
            encoders.append(getattr(net, '_obs_encoder', None))
        for encoder in encoders:
            if encoder is not None:
                encoder.eval()
                _set_force_stochastic(encoder, False)

    _set_force_stochastic(workspace.model.obs_encoder, enable_force_stochastic)
    _set_force_stochastic(workspace.unio4._policy.obs_encoder, enable_force_stochastic)
    _set_iql_deterministic(iql)
    _set_iql_deterministic(iql_online)

    if cfg.distill_phase == 'online':
        workspace.unio4._policy.set_target()
        distilled_path = os.path.join(workspace.offline_best_path, 'last/distilled_model.pt')
        if cfg.ppo.load_online_cp:
            cprint(
                'skip offline distilled model load because ppo.load_online_cp=True; '
                'online checkpoint will restore distilled model',
                'yellow'
            )
        elif os.path.exists(distilled_path):
            workspace.unio4._policy.distilled_model.load_state_dict(torch.load(distilled_path))
            print('load distilled model from {} for online distill successfully'.format(distilled_path))
        else:
            raise RuntimeError(
                "distill_phase='online' requires offline distill first, but "
                f"{distilled_path} does not exist."
            )
        cm_optimizer, cm_lr_scheduler = workspace.get_distill_optimizer()
    else:
        cm_optimizer, cm_lr_scheduler = None, None

    online_ft_path = os.path.join(workspace.output_dir, 'online_ft', time.strftime("%Y-%m-%d-%H-%M-%S"))
    config_dict = vars(cfg)

    def write_dict(f, d, indent=0):
        for key, val in d.items():
            if isinstance(val, dict):
                f.write(f"{' ' * indent}{key}:\n")
                write_dict(f, val, indent + 4)
            else:
                f.write(f"{' ' * indent}{key:20} : {val}\n")

    os.makedirs(online_ft_path, exist_ok=True)
    config_path = os.path.join(online_ft_path, 'config.txt')
    with open(config_path, 'w') as f:
        write_dict(f, config_dict)

    reward_scaler = None
    if cfg.ppo.scale_strategy == 'dynamic' or cfg.ppo.scale_strategy == 'number':
        from hydra.utils import instantiate
        from rl_100.unidpg.critic import ValueLearner
        critic_dataset = instantiate(cfg.task.critic_dataset)
        critic_dataloader = DataLoader(critic_dataset, **cfg.dataloader)
        online_value_encoder = workspace.unio4._policy.obs_encoder
        value = ValueLearner(
            device, 
            workspace.model.global_cond_dim, 
            cfg.critic.v_hidden_dim, 
            cfg.critic.v_depth, 
            cfg.critic.v_lr, 
            workspace.model.normalizer, 
            online_value_encoder, 
            workspace.model.n_obs_steps, 
            workspace.model.use_pc_color,
            share_encoder=cfg.ppo.share_encoder,
        )
        if cfg.ppo.share_encoder:
            v_path = os.path.join(workspace.output_dir, 'value_{}_{}.pt'.format(cfg.ppo.scale_strategy, cfg.ppo.share_encoder))
        else:
            v_path = os.path.join(workspace.output_dir, 'value_{}.pt'.format(cfg.ppo.scale_strategy))
            
        scale_dataset = instantiate(cfg.task.scale_dataset)
        scale_dataloader = DataLoader(scale_dataset, **cfg.dataloader)
        reward_scaler = scale_dataset.reward_norm
        cprint('start training value network with dynamic reward scaling', 'green')
        
        if os.path.exists(v_path):
            value.load(v_path)
        elif cfg.ppo.scale_strategy == 'number':
            epoch = 0
            for local_epoch_idx in range(cfg.ppo.num_critic_epochs):
                v_train_losses = list()
                epoch += 1
                with tqdm.tqdm(critic_dataloader, desc=f"Training epoch {epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch['reward'], batch['return'] = batch['reward'] * 0.1, batch['return'] * 0.1
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        value_loss = value.update(batch)
                        v_train_losses.append(value_loss)
                if local_epoch_idx % int(10) == 0:
                    print('Step: {}, Value loss: {}'.format(local_epoch_idx, np.mean(v_train_losses)))
            value.save(v_path)
        elif cfg.ppo.scale_strategy == 'dynamic':
            epoch = 0
            for local_epoch_idx in range(cfg.ppo.num_critic_epochs):
                v_train_losses = list()
                epoch += 1
                with tqdm.tqdm(scale_dataloader, desc=f"Training epoch {epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        value_loss = value.update(batch)
                        v_train_losses.append(value_loss)
                if local_epoch_idx % int(10) == 0:
                    print('Step: {}, Value loss: {}'.format(local_epoch_idx, np.mean(v_train_losses)))
            value.save(v_path)

        value_net = value._value
    else:
        value_net = iql.get_online_value_buget(cfg)

    if getattr(workspace.unio4._policy, 'is_flow', False):
        active_steps = workspace.unio4._policy.flow_inference_steps
        if active_steps != cfg.ppo.num_inference_steps:
            cprint(f'syncing num_inference_steps: {cfg.ppo.num_inference_steps} -> {active_steps}', 'yellow')
            cfg.ppo.num_inference_steps = active_steps
            cfg.num_inference_steps = active_steps
            cfg.policy.num_inference_steps = active_steps

    replay_buffer = ReplayBuffer(args=cfg.ppo, shape_info=workspace.shape_info, device=device)
    if cfg.ppo.iql_ft or cfg.update_phase == 'outloop':
        iql_buffer = IqlBuffer(None, args=cfg.ppo, shape_info=workspace.shape_info, device=device)
        iql = iql_online
        
    if cfg.ppo.load_online_cp:
        import glob
        online_cp_path = os.path.join(workspace.output_dir, 'online_ft')
        dirs = glob.glob(f"{online_cp_path}/*")
        logdir = sorted(dirs)[-1]
        iql, value_net = workspace.load_online_checkpoints(logdir, iql, value_net, ema)
        
    workspace.unio4.transfer2online(critic=value_net, dynamics=dynamics, cfg=cfg, cm_optimizer=cm_optimizer, cm_lr_scheduler=cm_lr_scheduler)

    if cfg.training.use_ema and workspace.ema_model is not None and ema is not None:
        if not cfg.ppo.load_online_cp:
            ema_state = workspace.ema_model.state_dict()
            policy_state = workspace.unio4._policy.state_dict()
            filtered_state = {k: v for k, v in policy_state.items() if k in ema_state}
            workspace.ema_model.load_state_dict(filtered_state, strict=False)
            ema.optimization_step = 0

    if use_vec_env:
        _online_ft_vec(workspace, dynamics, Q, iql, iql_online, wandb, online_ft_path, cm_optimizer, cm_lr_scheduler,
                        ema=ema, reward_scaler_template=reward_scaler if cfg.ppo.scale_strategy == 'dynamic' else None)
        return

    # start training and data collection
    total_steps = 0
    env_runner = workspace.env_runner
    env = env_runner.env
    all_success_rates, all_returns = [], []
    cm_all_success_rates, cm_all_returns = [], []
    all_idql_success_rates, all_idql_returns = [], []
    all_ema_success_rates, all_ema_returns = [], []
    
    if cfg.ppo.idql_eval:
        idql_log_data = workspace.unio4_eval(
            idql_eval=True,
            dynamics=dynamics,
            first_action=cfg.unio4.first_action,
            get_np=True,
            use_gae=cfg.unio4.use_gae,
            iql=iql,
            Q=Q,
            repeat_num=128,
            eval_times=cfg.unio4.eval_times,
        )
        all_idql_success_rates.append(idql_log_data['test_mean_score'])
        all_idql_returns.append(idql_log_data['mean_returns'])
        log_data = workspace.eval(eval_times=cfg.unio4.eval_times, online=True)
        if cfg.distill_phase == 'online':
            cm_log_data = workspace.eval(
                online=True, eval_times=cfg.unio4.eval_times,
                use_cm=True, distill2mean=cfg.distill2mean)
            cm_all_success_rates.append(cm_log_data['test_mean_score'])
            cm_all_returns.append(cm_log_data['mean_returns'])
        else:
            cm_all_success_rates.append(0)
            cm_all_returns.append(0)
    else:
        log_data = workspace.eval(eval_times=cfg.unio4.eval_times, online=True)
        if cfg.distill_phase == 'online':
            cm_log_data = workspace.eval(online=True, eval_times=cfg.unio4.eval_times, use_cm=True, distill2mean=cfg.distill2mean)
            cm_all_success_rates.append(cm_log_data['test_mean_score'])
            cm_all_returns.append(cm_log_data['mean_returns'])
        else:
            cm_all_success_rates.append(0)
            cm_all_returns.append(0)
        all_idql_success_rates.append(0)
        all_idql_returns.append(0)
        
    all_success_rates.append(log_data['test_mean_score'])
    all_returns.append(log_data['mean_returns'])
    
    ema_log_data = None
    if cfg.training.use_ema and workspace.ema_model is not None:
        ema_log_data = workspace.eval(online=True, eval_times=cfg.unio4.eval_times,
                                 policy_override=workspace.ema_model, eval_name='Online EMA Eval')
        all_ema_success_rates.append(ema_log_data['test_mean_score'])
        all_ema_returns.append(ema_log_data['mean_returns'])
        _, is_updated_ema = workspace.maybe_update_online_best_ema(ema_log_data['test_mean_score'])
        if is_updated_ema:
            print('------------saved online best EMA model----------------')
    else:
        all_ema_success_rates.append(0)
        all_ema_returns.append(0)
        
    cprint('start online finetuning, initial policy SR: {}, EMA SR: {}'.format(
        log_data['test_mean_score'],
        ema_log_data['test_mean_score'] if ema_log_data else 'N/A'), 'green')
        
    wandb.log({
        'online ppo success rates': log_data['test_mean_score'], 
        'cm success rates': cm_all_success_rates, 
        'cm returns': cm_all_returns,
        'online ppo returns': log_data['mean_returns'],
        'online ema success rates': ema_log_data['test_mean_score'] if ema_log_data else 0,
        'online ema returns': ema_log_data['mean_returns'] if ema_log_data else 0,
    })
    
    actor_losses, critic_losses, bc_losses, distill_losses = [], [], [], []
    q_train_losses, v_train_losses = [], []
    total_episode_r = deque(maxlen=10)
    episode_reward = 0
    episode_steps = 0
    update_num = 0
    time1 = 0
    
    while total_steps < cfg.ppo.max_train_steps:
        obs = env.reset()
        done = False
        if cfg.ppo.scale_strategy == 'dynamic':
            reward_scaler.reset()
        print('episode reward: {}, episode length: {}'.format(episode_reward, episode_steps))
        total_episode_r.append(episode_reward)
        episode_steps = 0
        episode_reward = 0
        
        if cfg.ppo.clip_std_decay:
            decay_value = workspace.value_decay(initial_value=cfg.clip_std_max, total_steps=total_steps, max_train_steps=cfg.ppo.max_train_steps)
            workspace.unio4._policy.noise_scheduler.clip_std_max = decay_value
            
        while not done:
            episode_steps += 1
            np_obs_dict = dict(obs)
            obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
            obs_dict_input = {}
            obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
            obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
            if 'dexart' in cfg.task_name:
                obs_dict_input['imagin_robot'] = obs_dict['imagin_robot'].unsqueeze(0)
            if 'image' in obs_dict:
                obs_dict_input['image'] = (obs_dict['image'].unsqueeze(0)).to(torch.float)
                
            if cfg.ppo.idql_rollout:
                action, all_x, a_logprob = workspace.unio4._policy.sample_action_with_logprob(
                    obs_dict_input, dynamics=dynamics, first_action=cfg.unio4.first_action, 
                    use_gae=cfg.unio4.use_gae, iql=iql, Q=Q, repeat_num=128
                )
            else:
                action, all_x, a_logprob = workspace.unio4._policy.all_step_action_logprob(
                    obs_dict_input, fix_encoder=cfg.ppo.fix_encoder
                )

            all_x = all_x.squeeze(1).detach().to('cpu').numpy()
            a_logprob = a_logprob.squeeze(1).detach().to('cpu').numpy()
            
            next_obs, reward, done, info = env.step(action.squeeze(0).detach().to('cpu').numpy(), reward_agg_method='discounted_sum', gamma=cfg.gamma)
            
            if done and episode_steps != cfg.task.env_runner.max_steps:
                dw = True
            else:
                dw = False
            episode_reward += reward
            
            obs_dict = dict_apply(obs_dict, lambda x: x.detach().to('cpu').numpy())
            if cfg.ppo.scale_strategy == 'number':
                replay_buffer.store(obs_dict, all_x, a_logprob, reward * 0.1, next_obs, done, dw)
            elif cfg.ppo.scale_strategy == 'dynamic':
                scaled_r = reward_scaler(reward)[0]
                replay_buffer.store(obs_dict, all_x, a_logprob, scaled_r, next_obs, done, dw)
            else:
                replay_buffer.store(obs_dict, all_x, a_logprob, reward, next_obs, done, dw)

            if cfg.ppo.iql_ft or cfg.update_phase == 'outloop':
                iql_buffer.store(obs=obs_dict, action=all_x[-1], reward=reward, next_obs=next_obs, done=done)

            if cfg.update_phase == 'outloop':
                alpha = 0.8 + (1 - 0.8) * (total_steps / cfg.ppo.max_train_steps)
                idql_bs = int(getattr(cfg.ppo, 'idql_batch_size', 256))
                online_sample_size = int(alpha * idql_bs)
                offline_sample_size = idql_bs - online_sample_size
                online_batch = iql_buffer.sample(batch_size=online_sample_size)
                offline_batch = workspace.sample_batch(batch_size=offline_sample_size)
                offline_batch = _prepare_offline_iql_batch_for_online(workspace, offline_batch)
                merged_batch = iql_buffer.merge(online_batch, offline_batch)
                distill_loss = workspace.unio4.distill_update(merged_batch, online=True)
                distill_losses.append(distill_loss)
                
            obs = next_obs
            total_steps += 1
            
            if replay_buffer.count == cfg.ppo.batch_size:
                update_num += 1
                if cfg.ppo.iql_ft:
                    if total_steps > cfg.ppo.online_start_training:
                        print('start online iql training')
                        for _ in range(cfg.ppo.iql_steps):
                            alpha = cfg.ppo.data_ratio + (1 - cfg.ppo.data_ratio) * (total_steps / cfg.ppo.max_train_steps)
                            idql_bs = int(getattr(cfg.ppo, 'idql_batch_size', 256))
                            online_sample_size = int(alpha * idql_bs)
                            offline_sample_size = idql_bs - online_sample_size
                            online_batch = iql_buffer.sample(batch_size=online_sample_size)
                            offline_batch = _next_offline_iql_batch_for_online(workspace)
                            merged_batch = iql_buffer.merge(online_batch, offline_batch)
                            merged_batch = dict_apply(merged_batch, lambda x: x[:idql_bs])
                            Q_bc_loss, value_loss = iql.update(batch=merged_batch, online=True, pre_cut=True, online_recon=cfg.ppo.online_iql_recon)
                        if total_steps % cfg.ppo.evaluate_freq == 0:
                            print('Step: {}, Q loss: {}, Value loss: {}'.format(total_steps, Q_bc_loss, value_loss))
                            wandb.log({'online iql Q_loss': Q_bc_loss, 'online iql value value_loss': value_loss})
                        q_train_losses.append(Q_bc_loss); v_train_losses.append(value_loss)
                    if cfg.ppo.fix_encoder:
                        if cfg.ppo.iql_q_encoder:
                            workspace.unio4._policy.obs_encoder.load_state_dict(iql._Q._obs_encoder.state_dict())
                        elif cfg.ppo.iql_v_encoder:
                            workspace.unio4._policy.obs_encoder.load_state_dict(iql._value._obs_encoder.state_dict())
                            
                time2 = time.time()
                pre_training_time = time.time()
                actor_loss, critic_loss, bc_loss, distill_loss = workspace.unio4.dp_align_update_no_share(replay_buffer, total_steps)
                if distill_loss != 0:
                    distill_losses.append(distill_loss)
                post_training_time = time.time()
                print('pure policy updated time: {}'.format(post_training_time - pre_training_time))
                time3 = time.time()
                
                if cfg.training.use_ema and ema is not None:
                    ema.step(workspace.unio4._policy)
                    
                ppo_elapsed = getattr(workspace.unio4, 'last_ppo_elapsed', None)
                ppo_time_str = f'; ppo loop: {ppo_elapsed:.2f}s' if ppo_elapsed is not None else ''
                print('step {}; collecting data time: {}; update time: {}{}'.format(total_steps, time2 - time1, time3 - time2, ppo_time_str))
                replay_buffer.count = 0
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                bc_losses.append(bc_loss)
                time1 = time.time()
                
                if cfg.ppo.save_online_cp and update_num % cfg.ppo.online_cp_save_freq == 0:
                    workspace.save_online_checkpoints(online_ft_path, update_num, iql, ema)

            if total_steps % cfg.ppo.evaluate_freq == 0:
                evaluate_num += 1
                if cfg.ppo.idql_eval:
                    idql_log_data = workspace.unio4_eval(
                        idql_eval=True,
                        dynamics=dynamics,
                        first_action=cfg.unio4.first_action,
                        get_np=True,
                        use_gae=cfg.unio4.use_gae,
                        iql=iql,
                        Q=Q,
                        repeat_num=128,
                        eval_times=cfg.unio4.eval_times,
                    )
                    log_data = workspace.eval(online=True, eval_times=cfg.unio4.eval_times)
                    all_idql_success_rates.append(idql_log_data['test_mean_score'])
                    all_idql_returns.append(idql_log_data['mean_returns'])
                    if cfg.distill_phase == 'online':
                        cm_log_data = workspace.eval(
                            online=True, eval_times=cfg.unio4.eval_times,
                            use_cm=True, distill2mean=cfg.distill2mean)
                        cm_all_success_rates.append(cm_log_data['test_mean_score'])
                        cm_all_returns.append(cm_log_data['mean_returns'])
                    else:
                        cm_all_success_rates.append(0)
                        cm_all_returns.append(0)
                else:
                    log_data = workspace.eval(online=True, eval_times=cfg.unio4.eval_times)
                    if cfg.distill_phase == 'online':
                        cm_log_data = workspace.eval(online=True, eval_times=cfg.unio4.eval_times, use_cm=True, distill2mean=cfg.distill2mean)
                        cm_all_success_rates.append(cm_log_data['test_mean_score'])
                        cm_all_returns.append(cm_log_data['mean_returns'])
                    else:
                        cm_all_success_rates.append(0)
                        cm_all_returns.append(0)
                    all_idql_success_rates.append(0)
                    all_idql_returns.append(0)

                all_success_rates.append(log_data['test_mean_score'])
                all_returns.append(log_data['mean_returns'])

                ema_log_data = None
                if cfg.training.use_ema and workspace.ema_model is not None:
                    ema_log_data = workspace.eval(online=True, eval_times=cfg.unio4.eval_times,
                                             policy_override=workspace.ema_model, eval_name='Online EMA Eval')
                    all_ema_success_rates.append(ema_log_data['test_mean_score'])
                    all_ema_returns.append(ema_log_data['mean_returns'])
                    _, is_updated_ema = workspace.maybe_update_online_best_ema(ema_log_data['test_mean_score'])
                    if is_updated_ema:
                        print('------------saved online best EMA model----------------')
                else:
                    all_ema_success_rates.append(0)
                    all_ema_returns.append(0)

                cprint(
                    'timestep {}: collecting performance: {} evaluate success rates: {}; evaluate returns: {}  actor_loss: {}; critic_loss: {}; bc_loss: {}; distill_loss: {}; cm_SR: {}; cm_ret: {}; idql_SR: {}; idql_ret: {}; ema_SR: {}; ema_ret: {};'.format(
                        total_steps,
                        np.mean(total_episode_r),
                        log_data['test_mean_score'],
                        log_data['mean_returns'],
                        np.mean(actor_losses[int(-cfg.ppo.evaluate_freq):]),
                        np.mean(critic_losses[int(-cfg.ppo.evaluate_freq):]),
                        np.mean(bc_losses[int(-cfg.ppo.evaluate_freq):]),
                        np.mean(distill_losses[int(-cfg.ppo.evaluate_freq):]),
                        cm_log_data['test_mean_score'] if cfg.distill_phase == 'online' else 0,
                        cm_log_data['mean_returns'] if cfg.distill_phase == 'online' else 0,
                        idql_log_data['test_mean_score'] if idql_log_data else 0,
                        idql_log_data['mean_returns'] if idql_log_data else 0,
                        ema_log_data['test_mean_score'] if ema_log_data else 0,
                        ema_log_data['mean_returns'] if ema_log_data else 0,
                    ),
                    'green'
                )
                wandb.log({
                    'online ppo success rates': log_data['test_mean_score'], 
                    'online ppo returns': log_data['mean_returns'],
                    'online ppo collect returns': np.mean(total_episode_r),
                    'online actor_loss': np.mean(actor_losses[int(-cfg.ppo.evaluate_freq):]), 
                    'online critic_loss': np.mean(critic_losses[int(-cfg.ppo.evaluate_freq):]), 
                    'online bc_loss': np.mean(bc_losses[int(-cfg.ppo.evaluate_freq):]),
                    'online distill_loss': np.mean(distill_losses[int(-cfg.ppo.evaluate_freq):]),
                    'cm_success rates': cm_log_data['test_mean_score'] if cfg.distill_phase == 'online' else 0,
                    'cm_returns': cm_log_data['mean_returns'] if cfg.distill_phase == 'online' else 0,
                    'idql_success rates': idql_log_data['test_mean_score'] if idql_log_data else 0,
                    'idql_returns': idql_log_data['mean_returns'] if idql_log_data else 0,
                    'online ema success rates': ema_log_data['test_mean_score'] if ema_log_data else 0,
                    'online ema returns': ema_log_data['mean_returns'] if ema_log_data else 0,
                })
                
                os.makedirs(online_ft_path, exist_ok=True)
                np.savetxt(os.path.join(online_ft_path, 'success_rates.csv'), all_success_rates, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'returns.csv'), all_returns, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'idql_success_rates.csv'), all_idql_success_rates, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'idql_returns.csv'), all_idql_returns, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'cm_success_rates.csv'), cm_all_success_rates, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'cm_returns.csv'), cm_all_returns, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'ema_success_rates.csv'), all_ema_success_rates, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'ema_returns.csv'), all_ema_returns, fmt='%f', delimiter=',')
                
    os.makedirs(os.path.join(online_ft_path, 'online_last'), exist_ok=True)
    workspace.unio4.save(os.path.join(online_ft_path, 'online_last'))
    if cfg.training.use_ema and workspace.ema_model is not None:
        os.makedirs(os.path.join(online_ft_path, 'online_last_ema'), exist_ok=True)
        workspace.ema_model.save(os.path.join(online_ft_path, 'online_last_ema'))
    workspace.unio4.flush_ratio_logs(force=True)

def _online_ft_vec(workspace, dynamics, Q, iql, iql_online, wandb, online_ft_path, cm_optimizer, cm_lr_scheduler, ema=None, reward_scaler_template=None):
    cfg = workspace.cfg
    device = workspace.device
    
    assert getattr(cfg, 'update_phase', 'inloop') != 'outloop', \
        'vec_env v1 does not support update_phase=outloop'
    assert not getattr(cfg.ppo, 'iql_adv', False), \
        'vec_env v1 does not support ppo.iql_adv=True'
    assert not getattr(cfg.ppo, 'idql_rollout', False), \
        'vec_env v1 does not support ppo.idql_rollout=True'

    train_env_num = getattr(cfg.ppo, 'train_env_num', 1)
    env_runner = workspace.env_runner
    steps_per_update = cfg.ppo.batch_size // train_env_num
    assert cfg.ppo.batch_size % train_env_num == 0, \
        f'batch_size ({cfg.ppo.batch_size}) must be divisible by train_env_num ({train_env_num})'

    use_subproc_vec_rollout = (
        getattr(cfg, 'feature_type', None) == '2D'
        and hasattr(env_runner, 'make_subproc_vec_env')
    )
    vec_env = None
    if use_subproc_vec_rollout:
        vec_env = env_runner.make_subproc_vec_env(
            train_env_num,
            record_video_first=False,
            reward_agg_method='discounted_sum',
            gamma=cfg.gamma,
        )
        envs = None
    else:
        envs = [env_runner.make_env(record_video=False) for _ in range(train_env_num)]
    max_steps = cfg.task.env_runner.max_steps

    if cfg.ppo.scale_strategy == 'dynamic':
        if reward_scaler_template is None:
            raise RuntimeError('vec dynamic reward scaling requires a non-null reward_scaler_template')
        import copy as copy_module_std
        reward_scalers = [copy_module_std.deepcopy(reward_scaler_template) for _ in range(train_env_num)]
        for scaler in reward_scalers:
            scaler.reset()

    replay_buffer = VecReplayBuffer(
        args=cfg.ppo, shape_info=workspace.shape_info,
        device=device, env_num=train_env_num,
        steps_per_update=steps_per_update)
    replay_buffer.reset()

    iql_ft = getattr(cfg.ppo, 'iql_ft', False)
    if iql_ft:
        iql_buffer = IqlBuffer(None, args=cfg.ppo, shape_info=workspace.shape_info, device=device)

    obs_debug_printed = False

    def stack_obs_dicts(obs_list):
        nonlocal obs_debug_printed
        if len(obs_list) == 0:
            raise RuntimeError('vec rollout received an empty obs_list')

        expected_keys = tuple(workspace.shape_info['obs'].keys())
        reference_keys = tuple(obs_list[0].keys())
        missing_from_first = [key for key in expected_keys if key not in reference_keys]
        if missing_from_first:
            raise KeyError(f"vec rollout obs is missing required keys {missing_from_first}; available keys: {sorted(reference_keys)}")

        batched = {}
        for key in expected_keys:
            missing_envs = [idx for idx, obs in enumerate(obs_list) if key not in obs]
            if missing_envs:
                raise KeyError(f"vec rollout obs key '{key}' missing from env indices {missing_envs}")

            try:
                stacked = np.stack([obs[key] for obs in obs_list], axis=0)
            except ValueError as exc:
                shapes = [np.asarray(obs[key]).shape for obs in obs_list]
                raise ValueError(f"vec rollout obs key '{key}' has inconsistent shapes across envs: {shapes}") from exc

            if stacked.size == 0:
                raise ValueError(f"vec rollout obs key '{key}' produced an empty batch")

            batched[key] = torch.from_numpy(stacked).to(device=device, dtype=torch.float)

        if not obs_debug_printed:
            print(f'vec rollout obs keys: {list(batched.keys())}')
            obs_debug_printed = True
        return batched

    def unstack_obs_batch(obs_batch_np):
        keys = list(obs_batch_np.keys())
        batch_size = obs_batch_np[keys[0]].shape[0]
        return [{k: obs_batch_np[k][i] for k in keys} for i in range(batch_size)]

    all_success_rates, all_returns = [], []
    cm_all_success_rates, cm_all_returns = [], []
    all_idql_success_rates, all_idql_returns = [], []
    all_ema_success_rates, all_ema_returns = [], []
    
    if cfg.ppo.idql_eval:
        idql_log_data = workspace.unio4_eval(
            idql_eval=True, dynamics=dynamics,
            first_action=cfg.unio4.first_action, get_np=True,
            use_gae=cfg.unio4.use_gae, iql=iql, Q=Q,
            repeat_num=128, eval_times=cfg.unio4.eval_times)
        all_idql_success_rates.append(idql_log_data['test_mean_score'])
        all_idql_returns.append(idql_log_data['mean_returns'])
        log_data = workspace.eval(eval_times=cfg.unio4.eval_times, online=True)
        if cfg.distill_phase == 'online':
            cm_log_data = workspace.eval(
                online=True, eval_times=cfg.unio4.eval_times,
                use_cm=True, distill2mean=cfg.distill2mean)
            cm_all_success_rates.append(cm_log_data['test_mean_score'])
            cm_all_returns.append(cm_log_data['mean_returns'])
        else:
            cm_all_success_rates.append(0)
            cm_all_returns.append(0)
    else:
        log_data = workspace.eval(eval_times=cfg.unio4.eval_times, online=True)
        if cfg.distill_phase == 'online':
            cm_log_data = workspace.eval(online=True, eval_times=cfg.unio4.eval_times, use_cm=True, distill2mean=cfg.distill2mean)
            cm_all_success_rates.append(cm_log_data['test_mean_score'])
            cm_all_returns.append(cm_log_data['mean_returns'])
        else:
            cm_all_success_rates.append(0)
            cm_all_returns.append(0)
        all_idql_success_rates.append(0)
        all_idql_returns.append(0)
        
    all_success_rates.append(log_data['test_mean_score'])
    all_returns.append(log_data['mean_returns'])
    
    ema_log_data = None
    if cfg.training.use_ema and workspace.ema_model is not None:
        ema_log_data = workspace.eval(online=True, eval_times=cfg.unio4.eval_times,
                                 policy_override=workspace.ema_model, eval_name='Online EMA Eval')
        all_ema_success_rates.append(ema_log_data['test_mean_score'])
        all_ema_returns.append(ema_log_data['mean_returns'])
        _, is_updated_ema = workspace.maybe_update_online_best_ema(ema_log_data['test_mean_score'])
        if is_updated_ema:
            print('------------saved online best EMA model----------------')
    else:
        all_ema_success_rates.append(0)
        all_ema_returns.append(0)
        
    cprint('start vec online finetuning, env_num={}, initial policy SR: {}, EMA SR: {}'.format(
        train_env_num, log_data['test_mean_score'],
        ema_log_data['test_mean_score'] if ema_log_data else 'N/A'), 'green')
        
    wandb.log({
        'online ppo success rates': log_data['test_mean_score'],
        'online ppo returns': log_data['mean_returns'],
        'cm_success rates': cm_all_success_rates[-1] if cm_all_success_rates else 0,
        'cm_returns': cm_all_returns[-1] if cm_all_returns else 0,
        'idql_success rates': all_idql_success_rates[-1] if all_idql_success_rates else 0,
        'idql_returns': all_idql_returns[-1] if all_idql_returns else 0,
        'online ema success rates': ema_log_data['test_mean_score'] if ema_log_data else 0,
        'online ema returns': ema_log_data['mean_returns'] if ema_log_data else 0,
    })

    total_steps = 0
    evaluate_num = 0
    next_eval_at = cfg.ppo.evaluate_freq
    actor_losses, critic_losses, bc_losses, distill_losses = [], [], [], []
    q_train_losses, v_train_losses = [], []
    total_episode_r = deque(maxlen=10)
    episode_rewards = [0.0] * train_env_num
    episode_steps_per_env = [0] * train_env_num
    update_num = 0
    time1 = time.time()
    idql_log_data = None

    if use_subproc_vec_rollout:
        obs_list = unstack_obs_batch(vec_env.reset())
    else:
        obs_list = [envs[i].reset() for i in range(train_env_num)]

    while total_steps < cfg.ppo.max_train_steps:
        if getattr(cfg.ppo, 'clip_std_decay', False):
            decay_value = workspace.value_decay(
                initial_value=cfg.clip_std_max,
                total_steps=total_steps,
                max_train_steps=cfg.ppo.max_train_steps)
            workspace.unio4._policy.noise_scheduler.clip_std_max = decay_value

        obs_before_step = [dict(obs) for obs in obs_list]
        obs_dict_input = stack_obs_dicts(obs_list)
        with torch.no_grad():
            action, all_x, a_logprob = workspace.unio4._policy.all_step_action_logprob(
                obs_dict_input, fix_encoder=cfg.ppo.fix_encoder)

        all_x_np = all_x.detach().cpu().numpy()
        a_logprob_np = a_logprob.detach().cpu().numpy()
        action_np = action.detach().cpu().numpy()

        next_obs_list = [None] * train_env_num
        step_rewards = np.zeros(train_env_num)
        step_dones = np.zeros(train_env_num)
        step_dws = np.zeros(train_env_num)

        if use_subproc_vec_rollout:
            obs_after_step_np, reward_batch, done_batch, info_batch = vec_env.step(action_np)
            reset_obs_list = unstack_obs_batch(obs_after_step_np)
            for i in range(train_env_num):
                reward = float(reward_batch[i])
                done = bool(done_batch[i])
                info = info_batch[i]

                episode_rewards[i] += reward
                episode_steps_per_env[i] += 1
                dw = done and episode_steps_per_env[i] != max_steps

                if done and 'terminal_observation' in info:
                    next_obs_list[i] = info['terminal_observation']
                else:
                    next_obs_list[i] = reset_obs_list[i]

                if cfg.ppo.scale_strategy == 'number':
                    step_rewards[i] = reward * 0.1
                elif cfg.ppo.scale_strategy == 'dynamic':
                    step_rewards[i] = reward_scalers[i](reward)[0]
                else:
                    step_rewards[i] = reward

                step_dones[i] = float(done)
                step_dws[i] = float(dw)

                if iql_ft:
                    iql_buffer.store(obs=obs_before_step[i], action=all_x_np[-1, i],
                                     reward=reward, next_obs=next_obs_list[i],
                                     done=step_dones[i])

                if done:
                    total_episode_r.append(episode_rewards[i])
                    print(f'env {i} episode reward: {episode_rewards[i]:.2f}, steps: {episode_steps_per_env[i]}')
                    episode_rewards[i] = 0.0
                    episode_steps_per_env[i] = 0
                    if cfg.ppo.scale_strategy == 'dynamic':
                        reward_scalers[i].reset()
                obs_list[i] = reset_obs_list[i]
        else:
            for i in range(train_env_num):
                next_obs, reward, done, info = envs[i].step(
                    action_np[i], reward_agg_method='discounted_sum', gamma=cfg.gamma)

                episode_rewards[i] += reward
                episode_steps_per_env[i] += 1
                dw = done and episode_steps_per_env[i] != max_steps

                if cfg.ppo.scale_strategy == 'number':
                    step_rewards[i] = reward * 0.1
                elif cfg.ppo.scale_strategy == 'dynamic':
                    step_rewards[i] = reward_scalers[i](reward)[0]
                else:
                    step_rewards[i] = reward

                step_dones[i] = float(done)
                step_dws[i] = float(dw)
                next_obs_list[i] = next_obs

                if iql_ft:
                    iql_buffer.store(obs=obs_before_step[i], action=all_x_np[-1, i],
                                     reward=reward, next_obs=next_obs_list[i],
                                     done=step_dones[i])
                if done:
                    total_episode_r.append(episode_rewards[i])
                    print(f'env {i} episode reward: {episode_rewards[i]:.2f}, steps: {episode_steps_per_env[i]}')
                    episode_rewards[i] = 0.0
                    episode_steps_per_env[i] = 0
                    if cfg.ppo.scale_strategy == 'dynamic':
                        reward_scalers[i].reset()
                    obs_list[i] = envs[i].reset()
                else:
                    obs_list[i] = next_obs

        obs_keys = list(obs_before_step[0].keys())
        obs_batch_np = {k: np.stack([obs_before_step[i][k] for i in range(train_env_num)], axis=0) for k in obs_keys}
        next_obs_batch_np = {k: np.stack([next_obs_list[i][k] for i in range(train_env_num)], axis=0) for k in obs_keys}

        all_x_for_buffer = np.moveaxis(all_x_np, 1, 0) if all_x_np.ndim > 2 and all_x_np.shape[1] == train_env_num else all_x_np
        a_logprob_for_buffer = np.moveaxis(a_logprob_np, 1, 0) if a_logprob_np.ndim > 2 and a_logprob_np.shape[1] == train_env_num else a_logprob_np

        replay_buffer.store(obs_batch_np, all_x_for_buffer, a_logprob_for_buffer,
                            step_rewards, next_obs_batch_np, step_dones, step_dws)
        total_steps += train_env_num

        if replay_buffer.count == steps_per_update:
            update_num += 1
            if iql_ft:
                if total_steps > cfg.ppo.online_start_training:
                    rng_snapshot = _iqlft_snapshot_rng() if _IQLFT_RESTORE_RNG else None
                    print('start online iql training')
                    for _ in range(cfg.ppo.iql_steps):
                        alpha = cfg.ppo.data_ratio + (1 - cfg.ppo.data_ratio) * (total_steps / cfg.ppo.max_train_steps)
                        idql_bs = int(getattr(cfg.ppo, 'idql_batch_size', 256))
                        online_sample_size = int(alpha * idql_bs)
                        offline_sample_size = idql_bs - online_sample_size
                        online_batch = iql_buffer.sample(batch_size=online_sample_size)
                        offline_batch = _next_offline_iql_batch_for_online(workspace)
                        merged_batch = iql_buffer.merge(online_batch, offline_batch)
                        merged_batch = dict_apply(merged_batch, lambda x: x[:idql_bs])
                        Q_bc_loss, value_loss = iql.update(batch=merged_batch, online=True, pre_cut=True, online_recon=cfg.ppo.online_iql_recon)
                    if total_steps >= next_eval_at - cfg.ppo.evaluate_freq + train_env_num:
                        print('Step: {}, Q loss: {}, Value loss: {}'.format(total_steps, Q_bc_loss, value_loss))
                        wandb.log({'online iql Q_loss': Q_bc_loss, 'online iql value value_loss': value_loss})
                    q_train_losses.append(Q_bc_loss); v_train_losses.append(value_loss)
                if cfg.ppo.fix_encoder:
                    if getattr(cfg.ppo, 'iql_q_encoder', False):
                        workspace.unio4._policy.obs_encoder.load_state_dict(iql._Q._obs_encoder.state_dict())
                    elif getattr(cfg.ppo, 'iql_v_encoder', False):
                        workspace.unio4._policy.obs_encoder.load_state_dict(iql._value._obs_encoder.state_dict())
                if _IQLFT_RESTORE_RNG and total_steps > cfg.ppo.online_start_training:
                    _iqlft_restore_rng(rng_snapshot)

            s_vec, a_vec, a_logprob_vec, r_vec, s_vec_, dw_vec, done_vec = replay_buffer.numpy_to_tensor_vec()
            with torch.no_grad():
                flat_s = dict_apply(s_vec, lambda x: x.reshape(-1, *x.shape[2:]))
                flat_s_ = dict_apply(s_vec_, lambda x: x.reshape(-1, *x.shape[2:]))
                if workspace.unio4.args.share_encoder:
                    flat_vs, flat_vs_ = workspace.unio4._compute_critic_values_in_chunks(flat_s, flat_s_, use_obs2latent=True)
                else:
                    flat_vs, flat_vs_ = workspace.unio4._compute_critic_values_in_chunks(flat_s, flat_s_, use_obs2latent=False)
                vs = flat_vs.reshape(steps_per_update, train_env_num, 1)
                vs_ = flat_vs_.reshape(steps_per_update, train_env_num, 1)
                adv, v_target = compute_gae_per_env(r_vec, done_vec, dw_vec, vs, vs_, cfg.ppo.gamma, cfg.ppo.lamda, cfg.n_action_steps)

            flat_args = copy.copy(cfg.ppo)
            flat_args.batch_size = steps_per_update * train_env_num
            flat_replay = FlatReplayBuffer(args=flat_args, shape_info=workspace.shape_info, device=device)

            if not replay_buffer.wo_visual:
                flat_replay.point_cloud = replay_buffer.point_cloud[:steps_per_update].reshape(-1, *replay_buffer.point_cloud.shape[2:])
                flat_replay.image = replay_buffer.image[:steps_per_update].reshape(-1, *replay_buffer.image.shape[2:])
                if replay_buffer.use_imagin_robot:
                    flat_replay.imagin_robot = replay_buffer.imagin_robot[:steps_per_update].reshape(-1, *replay_buffer.imagin_robot.shape[2:])
            flat_replay.agent_pos = replay_buffer.agent_pos[:steps_per_update].reshape(-1, *replay_buffer.agent_pos.shape[2:])
            flat_replay.action = replay_buffer.action[:steps_per_update].reshape(-1, *replay_buffer.action.shape[2:])
            flat_replay.a_logprob = replay_buffer.a_logprob[:steps_per_update].reshape(-1, *replay_buffer.a_logprob.shape[2:])
            flat_replay.reward = replay_buffer.reward[:steps_per_update].reshape(-1, 1)
            if not replay_buffer.wo_visual:
                flat_replay.next_point_cloud = replay_buffer.next_point_cloud[:steps_per_update].reshape(-1, *replay_buffer.next_point_cloud.shape[2:])
                flat_replay.next_image = replay_buffer.next_image[:steps_per_update].reshape(-1, *replay_buffer.next_image.shape[2:])
                if replay_buffer.use_imagin_robot:
                    flat_replay.next_imagin_robot = replay_buffer.next_imagin_robot[:steps_per_update].reshape(-1, *replay_buffer.next_imagin_robot.shape[2:])
            flat_replay.next_agent_pos = replay_buffer.next_agent_pos[:steps_per_update].reshape(-1, *replay_buffer.next_agent_pos.shape[2:])
            flat_replay.done = replay_buffer.done[:steps_per_update].reshape(-1, 1)
            flat_replay.dw = replay_buffer.dw[:steps_per_update].reshape(-1, 1)
            flat_replay.count = steps_per_update * train_env_num

            precomputed = {
                'adv': adv,
                'v_target': v_target,
                'vs': flat_vs.reshape(-1, 1),
            }

            time2 = time.time()
            actor_loss, critic_loss, bc_loss, distill_loss = workspace.unio4.dp_align_update_no_share(flat_replay, total_steps, precomputed=precomputed)
            if distill_loss != 0:
                distill_losses.append(distill_loss)
            time3 = time.time()
            if cfg.training.use_ema and ema is not None:
                ema.step(workspace.unio4._policy)
            print(f'step {total_steps}; collecting data time: {time2 - time1:.2f}; update time: {time3 - time2:.2f}')

            replay_buffer.reset()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            bc_losses.append(bc_loss)
            time1 = time.time()

            if getattr(cfg.ppo, 'save_online_cp', False) and update_num % getattr(cfg.ppo, 'online_cp_save_freq', 100) == 0:
                workspace.save_online_checkpoints(online_ft_path, update_num, iql, ema)

        if total_steps >= next_eval_at:
            next_eval_at += cfg.ppo.evaluate_freq
            evaluate_num += 1
            if cfg.ppo.idql_eval:
                idql_log_data = workspace.unio4_eval(
                    idql_eval=True, dynamics=dynamics,
                    first_action=cfg.unio4.first_action, get_np=True,
                    use_gae=cfg.unio4.use_gae, iql=iql, Q=Q,
                    repeat_num=128, eval_times=cfg.unio4.eval_times)
                log_data = workspace.eval(online=True, eval_times=cfg.unio4.eval_times)
                all_idql_success_rates.append(idql_log_data['test_mean_score'])
                all_idql_returns.append(idql_log_data['mean_returns'])
                if cfg.distill_phase == 'online':
                    cm_log_data = workspace.eval(
                        online=True, eval_times=cfg.unio4.eval_times,
                        use_cm=True, distill2mean=cfg.distill2mean)
                    cm_all_success_rates.append(cm_log_data['test_mean_score'])
                    cm_all_returns.append(cm_log_data['mean_returns'])
                else:
                    cm_all_success_rates.append(0)
                    cm_all_returns.append(0)
            else:
                log_data = workspace.eval(online=True, eval_times=cfg.unio4.eval_times)
                if cfg.distill_phase == 'online':
                    cm_log_data = workspace.eval(online=True, eval_times=cfg.unio4.eval_times, use_cm=True, distill2mean=cfg.distill2mean)
                    cm_all_success_rates.append(cm_log_data['test_mean_score'])
                    cm_all_returns.append(cm_log_data['mean_returns'])
                else:
                    cm_all_success_rates.append(0)
                    cm_all_returns.append(0)
                all_idql_success_rates.append(0)
                all_idql_returns.append(0)

            all_success_rates.append(log_data['test_mean_score'])
            all_returns.append(log_data['mean_returns'])

            ema_log_data = None
            if cfg.training.use_ema and workspace.ema_model is not None:
                ema_log_data = workspace.eval(online=True, eval_times=cfg.unio4.eval_times,
                                         policy_override=workspace.ema_model, eval_name='Online EMA Eval')
                all_ema_success_rates.append(ema_log_data['test_mean_score'])
                all_ema_returns.append(ema_log_data['mean_returns'])
                _, is_updated_ema = workspace.maybe_update_online_best_ema(ema_log_data['test_mean_score'])
                if is_updated_ema:
                    print('------------saved online best EMA model----------------')
            else:
                all_ema_success_rates.append(0)
                all_ema_returns.append(0)

            cprint(
                'timestep {}: collecting performance: {} evaluate success rates: {}; evaluate returns: {}  actor_loss: {}; critic_loss: {}; bc_loss: {}; distill_loss: {}; cm_SR: {}; cm_ret: {}; idql_SR: {}; idql_ret: {}; ema_SR: {}; ema_ret: {};'.format(
                    total_steps,
                    np.mean(total_episode_r),
                    log_data['test_mean_score'],
                    log_data['mean_returns'],
                    np.mean(actor_losses[int(-cfg.ppo.evaluate_freq):]),
                    np.mean(critic_losses[int(-cfg.ppo.evaluate_freq):]),
                    np.mean(bc_losses[int(-cfg.ppo.evaluate_freq):]),
                    np.mean(distill_losses[int(-cfg.ppo.evaluate_freq):]),
                    cm_log_data['test_mean_score'] if cfg.distill_phase == 'online' else 0,
                    cm_log_data['mean_returns'] if cfg.distill_phase == 'online' else 0,
                    idql_log_data['test_mean_score'] if idql_log_data else 0,
                    idql_log_data['mean_returns'] if idql_log_data else 0,
                    ema_log_data['test_mean_score'] if ema_log_data else 0,
                    ema_log_data['mean_returns'] if ema_log_data else 0,
                ), 'green'
            )
            wandb.log({
                'online ppo success rates': log_data['test_mean_score'],
                'online ppo returns': log_data['mean_returns'],
                'online ppo collect returns': np.mean(total_episode_r),
                'online actor_loss': np.mean(actor_losses[int(-cfg.ppo.evaluate_freq):]),
                'online critic_loss': np.mean(critic_losses[int(-cfg.ppo.evaluate_freq):]),
                'online bc_loss': np.mean(bc_losses[int(-cfg.ppo.evaluate_freq):]),
                'online distill_loss': np.mean(distill_losses[int(-cfg.ppo.evaluate_freq):]),
                'cm_success rates': cm_log_data['test_mean_score'] if cfg.distill_phase == 'online' else 0,
                'cm_returns': cm_log_data['mean_returns'] if cfg.distill_phase == 'online' else 0,
                'idql_success rates': idql_log_data['test_mean_score'] if idql_log_data else 0,
                'idql_returns': idql_log_data['mean_returns'] if idql_log_data else 0,
                'online ema success rates': ema_log_data['test_mean_score'] if ema_log_data else 0,
                'online ema returns': ema_log_data['mean_returns'] if ema_log_data else 0,
            })
            
            os.makedirs(online_ft_path, exist_ok=True)
            np.savetxt(os.path.join(online_ft_path, 'success_rates.csv'), all_success_rates, fmt='%f', delimiter=',')
            np.savetxt(os.path.join(online_ft_path, 'returns.csv'), all_returns, fmt='%f', delimiter=',')
            np.savetxt(os.path.join(online_ft_path, 'idql_success_rates.csv'), all_idql_success_rates, fmt='%f', delimiter=',')
            np.savetxt(os.path.join(online_ft_path, 'idql_returns.csv'), all_idql_returns, fmt='%f', delimiter=',')
            np.savetxt(os.path.join(online_ft_path, 'cm_success_rates.csv'), cm_all_success_rates, fmt='%f', delimiter=',')
            np.savetxt(os.path.join(online_ft_path, 'cm_returns.csv'), cm_all_returns, fmt='%f', delimiter=',')
            np.savetxt(os.path.join(online_ft_path, 'ema_success_rates.csv'), all_ema_success_rates, fmt='%f', delimiter=',')
            np.savetxt(os.path.join(online_ft_path, 'ema_returns.csv'), all_ema_returns, fmt='%f', delimiter=',')

    os.makedirs(os.path.join(online_ft_path, 'online_last'), exist_ok=True)
    workspace.unio4.save(os.path.join(online_ft_path, 'online_last'))
    if cfg.training.use_ema and workspace.ema_model is not None:
        os.makedirs(os.path.join(online_ft_path, 'online_last_ema'), exist_ok=True)
        workspace.ema_model.save(os.path.join(online_ft_path, 'online_last_ema'))
    workspace.unio4.flush_ratio_logs(force=True)
