import os
import time
import fcntl
import hydra
import torch
import dill
import inspect
import pathlib
import random
import copy
from copy import deepcopy
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import threading
from collections import deque
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rl_100.policy.rl100_3d import RL1003D
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.env_runner.base_runner import BaseRunner
from rl_100.common.checkpoint_util import TopKCheckpointManager
from rl_100.common.pytorch_util import dict_apply, optimizer_to
from rl_100.model.diffusion.ema_model import EMAModel
from rl_100.model.common.lr_scheduler import get_scheduler
from rl_100.unidpg.uni_ppo import BehaviorProximalPolicyOptimization

def _copy_to_cpu(state_dict):
    """Copy state dict tensors to CPU for async saving."""
    return {k: v.cpu().clone() for k, v in state_dict.items()}

# Snapshot/restore RNG around the online iql_ft constructor (and its checkpoint
# load) so the extra IQL network does not perturb torch-cpu / cuda / numpy /
# random RNG streams. Toggled by IQLFT_RESTORE_RNG_AFTER_IQL (default on).
_IQLFT_RESTORE_RNG = os.environ.get("IQLFT_RESTORE_RNG_AFTER_IQL", "1") == "1"
_IQLFT_RESTORE_RNG_AT_CONSTRUCT = (
    _IQLFT_RESTORE_RNG
    or os.environ.get("IQLFT_RESTORE_RNG_AT_CONSTRUCT", "0") == "1"
)

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

def init_wandb_run(cfg, output_dir):
    logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
    init_timeout = int(logging_cfg.pop("init_timeout", 120))
    retry_init_timeout = int(logging_cfg.pop("retry_init_timeout", max(init_timeout * 2, 300)))
    settings_cfg = logging_cfg.pop("settings", {}) or {}

    def _wandb_init(timeout):
        settings = dict(settings_cfg)
        settings["init_timeout"] = timeout
        return wandb.init(
            dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(**settings),
            **logging_cfg
        )

    try:
        return _wandb_init(init_timeout)
    except wandb.errors.CommError as exc:
        if "timeout" not in str(exc).lower():
            raise
        cprint(
            f"[WandB] init timed out after {init_timeout}s, retrying with {retry_init_timeout}s",
            "yellow",
        )
        return _wandb_init(retry_init_timeout)

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        cfg.ppo.num_inference_steps = cfg.policy.num_inference_steps
        if getattr(cfg.ppo, 'iql_adv', False):
            raise ValueError(
                'ppo.iql_adv=True is no longer supported after removing '
                'dp_align_update_iql_no_share; use the default PPO path instead.'
            )

        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        print('Training workspace initialized 1')
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.device = cfg.training.device
        
        # configure model
        self.model: RL1003D = hydra.utils.instantiate(cfg.policy)

        # load pretrained 2D encoder if configured
        if getattr(cfg, 'use_pretrained_2DEncoder', False):
            if 'channel' in cfg.policy.stage1_model_name:
                print('load pretrained encoder')
                self.model.obs_encoder.load_pretrained_encoder(
                    self.get_pretrained_model_path(cfg.policy.stage1_model_name), device=self.device)
                self.model.obs_encoder.switch_to_RL_stages()

        self.ema_model: RL1003D = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except Exception: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)
                
        # unio4 for finetuning dp3
        self.unio4 = BehaviorProximalPolicyOptimization(
            policy=self.model,
            device=torch.device(cfg.training.device),
            policy_lr=cfg.unio4.bppo_lr,
            clip_ratio=cfg.unio4.clip_ratio,
            entropy_weight=cfg.unio4.entropy_weight,
            decay=cfg.unio4.decay,
            omega=cfg.unio4.omega,
            batch_size=cfg.unio4.bppo_batch_size,
            is_iql=cfg.critic.is_iql,
            temperature=cfg.unio4.temperature,
            ratio_strategy=cfg.unio4.ratio_strategy,
            top_k=cfg.unio4.top_k,
            num_inference_steps=cfg.policy.num_inference_steps,
            fix_encoder=cfg.unio4.fix_encoder,
            cfg=cfg,
        )

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def get_stage1_artifact_dir(self):
        return self.cfg.unio4.get('stage1_resume_dir', None) or self.output_dir

    def get_critic_artifact_dir(self):
        stage1_dir = self.get_stage1_artifact_dir()
        if self.cfg.chunk_as_single_action:
            explicit_dir = self.cfg.unio4.get('critic_artifact_dir', None)
            if explicit_dir:
                return explicit_dir

            inferred_dir = os.path.join(
                stage1_dir,
                f'critic_c{self.cfg.n_action_steps}_f{self.cfg.n_action_steps}',
            )
            if os.path.exists(os.path.join(inferred_dir, 'Q_bc_20.pt')):
                return inferred_dir
        return stage1_dir

    def get_stage1_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.get_stage1_artifact_dir()).joinpath('checkpoints', f'{tag}.ckpt')

    def get_global_best_dir(self):
        return self.cfg.unio4.get('global_best_dir', None) or os.path.join(self.output_dir, 'best')

    def get_global_best_ema_dir(self):
        return self.cfg.unio4.get('global_best_ema_dir', None) or os.path.join(self.output_dir, 'best_ema')

    def get_global_best_score_path(self):
        return os.path.join(self.get_global_best_dir(), 'best_score.csv')

    def get_global_best_ema_score_path(self):
        return os.path.join(self.get_global_best_ema_dir(), 'best_score.csv')

    def get_global_best_lock_path(self):
        best_dir = self.get_global_best_dir()
        return os.path.join(os.path.dirname(best_dir), '.global_best.lock')

    def get_global_best_ema_lock_path(self):
        best_dir = self.get_global_best_ema_dir()
        return os.path.join(os.path.dirname(best_dir), '.global_best_ema.lock')

    def _read_best_score(self, score_path):
        if os.path.exists(score_path):
            best_score = np.loadtxt(score_path, delimiter=',')
            if isinstance(best_score, np.ndarray):
                best_score = float(np.asarray(best_score).reshape(-1)[0])
            else:
                best_score = float(best_score)
            return best_score
        return float('-inf')

    def _maybe_update_best(self, score, best_dir, best_score_path, lock_path, save_fn, eval_name):
        os.makedirs(os.path.dirname(best_dir), exist_ok=True)
        with open(lock_path, 'a+') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            best_saved_scores = self._read_best_score(best_score_path)
            is_updated = score > best_saved_scores
            if is_updated:
                os.makedirs(best_dir, exist_ok=True)
                save_fn(best_dir)
                np.savetxt(best_score_path, [score], fmt='%f', delimiter=',')
                meta_path = os.path.join(best_dir, 'best_meta.txt')
                with open(meta_path, 'w') as f:
                    f.write(f"score: {score}\n")
                    f.write(f"eval_name: {eval_name}\n")
                    f.write(f"source_run_dir: {self.output_dir}\n")
                    f.write(f"timestamp_dir: {self.unio4_output_dir}\n")
                    f.write(f"seed: {self.cfg.training.seed}\n")
                    f.write(f"rollout_length: {self.cfg.unio4.rollout_length}\n")
                    f.write(f"bppo_lr: {self.cfg.unio4.bppo_lr}\n")
            else:
                score = best_saved_scores
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return score, is_updated

    def maybe_update_global_best(self, score):
        return self._maybe_update_best(
            score=score,
            best_dir=self.get_global_best_dir(),
            best_score_path=self.get_global_best_score_path(),
            lock_path=self.get_global_best_lock_path(),
            save_fn=self.unio4.save,
            eval_name='Policy Eval',
        )

    def maybe_update_global_best_ema(self, score):
        if self.ema_model is None:
            return float('-inf'), False
        return self._maybe_update_best(
            score=score,
            best_dir=self.get_global_best_ema_dir(),
            best_score_path=self.get_global_best_ema_score_path(),
            lock_path=self.get_global_best_ema_lock_path(),
            save_fn=self.ema_model.save,
            eval_name='EMA Eval',
        )

    def get_online_best_ema_dir(self):
        return os.path.join(self.output_dir, 'online_best_ema')

    def get_online_best_ema_score_path(self):
        return os.path.join(self.get_online_best_ema_dir(), 'best_score.csv')

    def get_online_best_ema_lock_path(self):
        return os.path.join(os.path.dirname(self.get_online_best_ema_dir()), '.online_best_ema.lock')

    def maybe_update_online_best_ema(self, score):
        if self.ema_model is None:
            return float('-inf'), False
        return self._maybe_update_best(
            score=score,
            best_dir=self.get_online_best_ema_dir(),
            best_score_path=self.get_online_best_ema_score_path(),
            lock_path=self.get_online_best_ema_lock_path(),
            save_fn=self.ema_model.save,
            eval_name='Online EMA Eval',
        )

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.distill_phase is not None and getattr(self.model, 'is_flow', False):
            if cfg.distill_phase not in ('after_dp', 'after_offline', 'online'):
                raise RuntimeError(f"Unsupported distill_phase='{cfg.distill_phase}' for flow mode.")

        copy_encoder = deepcopy(self.model.obs_encoder)

        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        self.verbose = verbose
        RUN_VALIDATION = False # reduce time cost
        self.RUN_ROLLOUT = RUN_ROLLOUT
        self.RUN_VALIDATION = RUN_VALIDATION
        self.output_dir = self._resolve_output_dir()

        self.unio4_output_dir = os.path.join(self.output_dir, time.strftime("%Y-%m-%d-%H-%M-%S"))
        config = vars(cfg)  
        
        for group in self.optimizer.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = group['lr']

        def write_dict(f, d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    f.write(f"{' ' * indent}{key}:\n")
                    write_dict(f, value, indent + 4)  
                else:
                    f.write(f"{' ' * indent}{key:20} : {value}\n")

        os.makedirs(self.unio4_output_dir, exist_ok=True)
        self.unio4.set_ratio_log_dir(os.path.join(self.unio4_output_dir, 'ratio_logs'))
        config_path = os.path.join(self.unio4_output_dir, 'config.txt')

        with open(config_path, 'w') as f:
            write_dict(f, config)
        print('====================================Here==================================')
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_stage1_checkpoint_path(tag='latest')
            lastest_cm_ckpt_path = self.get_stage1_checkpoint_path(tag='latest_cm')
            if lastest_cm_ckpt_path.is_file():
                print(f"Resuming cm model from checkpoint {lastest_cm_ckpt_path}")
                if cfg.distill_phase is not None:
                    self.model.set_target()
                self.load_checkpoint(path=lastest_cm_ckpt_path)
            elif lastest_ckpt_path.is_file():
                print(f"Resuming diffusion model from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
            else:
                print(f"No checkpoint found at {lastest_ckpt_path}")

        device = torch.device(cfg.training.device)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        self.dataset = dataset
        self.shape_info = dataset.get_shape_info(self.cfg.horizon - self.model.start, self.cfg.n_obs_steps)
        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        
        if (self.cfg.off2off and self.cfg.off2off_no_bc) or self.cfg.use_pre_norm:
            norm_dataset = hydra.utils.instantiate(cfg.task.norm_dataset)
            normalizer = norm_dataset.get_normalizer()
            cprint('***********************************reuse the normalizer of pre-dataset***********************************', 'yellow')
        else:
            normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        self.ema = None
        if cfg.training.use_ema:
            self.ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        self.env_runner = env_runner
        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
            
        if self.cfg.use_wandb:
            cfg.logging.name = str(cfg.logging.name)
            cprint("-----------------------------", "yellow")
            cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
            cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
            cprint("-----------------------------", "yellow")
            import logging
            wandb_logger = logging.getLogger("wandb")
            wandb_logger.setLevel(logging.ERROR)
            wandb_run = init_wandb_run(cfg, self.output_dir)
            cprint(f"[WandB] view run: {wandb_run.url}", "cyan")
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                },
                allow_val_change=True
            )
        else:
            import logging
            logging.getLogger("wandb").setLevel(logging.ERROR)
            wandb_run = wandb.init(mode="disabled")

        self.topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Stage 1: BC (Diffusion) Training
        if cfg.eval:
            if getattr(self.cfg, 'load_path', None) is not None:
                checkpoint_dir = self.cfg.load_path
                if os.path.exists(checkpoint_dir):
                    if os.path.exists(os.path.join(checkpoint_dir, 'model.pt')):
                        print(f"Loading unio4 policy from {checkpoint_dir}")
                        self.unio4.load(checkpoint_dir)
                        self.model.model.load_state_dict(self.unio4._policy.model.state_dict())
                        self.model.obs_encoder.load_state_dict(self.unio4._policy.obs_encoder.state_dict())
                        if self.ema_model is not None:
                            self.ema_model.model.load_state_dict(self.unio4._policy.model.state_dict())
                            self.ema_model.obs_encoder.load_state_dict(self.unio4._policy.obs_encoder.state_dict())
                    else:
                        print(f"Loading workspace checkpoint from {checkpoint_dir}")
                        self.load_checkpoint(path=checkpoint_dir)
            if not self.cfg.unio4.idql_eval:
                log_data = self.eval(eval_times=self.cfg.unio4.eval_times)
                score = log_data['test_mean_score']
                return score

        from rl_100.training.bc_trainer import train_bc
        train_bc(self, train_dataloader, val_dataloader, wandb_run, env_runner, device)
        
        self.offline_best_path = self.get_global_best_dir()
        self.offline_last_path = os.path.join(self.output_dir, 'last')
        if self.cfg.only_bc:
            cprint('only_bc=True: BC stage done, stopping before the critic/dynamics/offline-RL stages.', 'green')
            return
            
        if self.cfg.n_obs_steps > 1 and not self.cfg.chunk_as_single_action:
            if getattr(self.cfg.dynamics, 'prediction_mode', 'last') != 'full':
                raise ValueError(
                    f"n_obs_steps={self.cfg.n_obs_steps} (chunk_as_single_action=False) requires "
                    f"dynamics.prediction_mode='full', got '{getattr(self.cfg.dynamics, 'prediction_mode', 'last')}'."
                )

        # Stage 1.5 & 1.6: Critic & Dynamics training
        from rl_100.training.critic_trainer import train_critic_and_dynamics
        iql, Q_bc, value, dynamics, critic_artifact_dir, Q_bc_path, value_path = train_critic_and_dynamics(
            self, train_dataloader, val_dataloader, wandb_run, env_runner, device)

        if self.cfg.distill_phase == 'after_dp':
            from rl_100.training.distill import distill_cm
            distill_cm(self, train_dataloader, val_dataloader, wandb_run, env_runner, phase=self.cfg.distill_phase)

        # Stage 2: Offline RL (BPPO)
        self.unio4.set_policy(self.model)
        self.unio4.set_old_policy()
        if cfg.eval:
            if getattr(self.cfg, 'load_path', None) is not None:
                checkpoint_dir = self.cfg.load_path
                if os.path.exists(checkpoint_dir):
                    if os.path.exists(os.path.join(checkpoint_dir, 'model.pt')):
                        print(f"Loading unio4 policy from {checkpoint_dir}")
                        self.unio4.load(checkpoint_dir)
                        self.model.model.load_state_dict(self.unio4._policy.model.state_dict())
                        self.model.obs_encoder.load_state_dict(self.unio4._policy.obs_encoder.state_dict())
                        if self.ema_model is not None:
                            self.ema_model.model.load_state_dict(self.unio4._policy.model.state_dict())
                            self.ema_model.obs_encoder.load_state_dict(self.unio4._policy.obs_encoder.state_dict())
                    else:
                        print(f"Loading workspace checkpoint from {checkpoint_dir}")
                        self.load_checkpoint(path=checkpoint_dir)
            if self.cfg.unio4.idql_eval:
                log_data = self.unio4_eval(
                    idql_eval=True,
                    dynamics=dynamics,
                    first_action=self.cfg.unio4.first_action,
                    get_np=True,
                    iql=iql,
                    Q=Q_bc,
                    repeat_num=128,
                    eval_times=self.cfg.unio4.eval_times
                )
            else:
                log_data = self.eval(eval_times=self.cfg.unio4.eval_times)
            score = log_data['test_mean_score']
            return score
        else:
            if cfg.offline:
                from rl_100.training.offline_rl import offline_finetune
                offline_finetune(self, dynamics, Q_bc, value, iql)

        # Stage 2.5: CM Distill after Offline RL
        if self.cfg.distill_phase == 'after_offline':
            if self.cfg.offline_cp_timestamp and self.cfg.offline_cp_timestep is not None:
                self.unio4.load(os.path.join(self.output_dir, self.cfg.offline_cp_timestamp, self.cfg.offline_cp_timestep))
            else:
                self.unio4.load(os.path.join(self.offline_best_path))
            from rl_100.training.distill import distill_cm
            distill_cm(self, train_dataloader, val_dataloader, wandb_run, env_runner, phase=self.cfg.distill_phase)

        # Stage 3: Online RL fine-tuning
        if cfg.online:
            if self.cfg.load_bc:
                if self.cfg.distill_phase == 'online' and not self.cfg.ppo.load_online_cp:
                    raise RuntimeError(
                        "distill_phase='online' requires an offline policy checkpoint. "
                        "Disable load_bc or resume from an online checkpoint."
                    )
                self.unio4._policy.model.load_state_dict(self.model.model.state_dict())
                self.unio4._policy.obs_encoder.load_state_dict(self.model.obs_encoder.state_dict())
                self.unio4.set_old_policy()
            elif self.cfg.distill_phase in ('after_dp', 'after_offline'):
                if self.cfg.distill_phase == 'after_dp':
                    self.unio4.load(os.path.join(self.offline_best_path, 'last'))
                if getattr(self.unio4._policy, 'is_flow', False):
                    target_steps = self.unio4._policy.flow_distill_inference_steps
                    self.unio4._policy.flow_inference_steps = target_steps
                    self.unio4._old_policy.flow_inference_steps = target_steps
                    cprint(f'restored flow_inference_steps={target_steps} for promoted model (policy + old_policy)', 'yellow')
            else:
                if self.cfg.offline_cp_timestamp and self.cfg.offline_cp_timestep is not None:
                    self.unio4.load(os.path.join(self.output_dir, self.cfg.offline_cp_timestamp, self.cfg.offline_cp_timestep))
                else:
                    self.unio4.load(os.path.join(self.offline_best_path))
                    
            if self.cfg.distill_phase == 'online' and not self.cfg.ppo.load_online_cp:
                offline_distilled_path = os.path.join(self.offline_best_path, 'last', 'distilled_model.pt')
                if os.path.exists(offline_distilled_path):
                    cprint('found offline distilled model for online distill: {}'.format(offline_distilled_path), 'green')
                else:
                    cprint('offline distilled model not found at {}; running offline distill before online'.format(offline_distilled_path), 'yellow')
                    from rl_100.training.distill import distill_cm
                    distill_cm(train_dataloader, val_dataloader, wandb_run, env_runner, phase='after_offline')
                    if not os.path.exists(offline_distilled_path):
                        raise RuntimeError(f"Offline distill completed but did not create {offline_distilled_path}")
            
            if cfg.ppo.iql_ft:
                from rl_100.unidpg.critic import IQL_Q_V_no
                _iqlft_construct_snapshot = _iqlft_snapshot_rng() if _IQLFT_RESTORE_RNG_AT_CONSTRUCT else None
                online_iql_encoder = deepcopy(self.model.obs_encoder)
                iql_online = IQL_Q_V_no(
                    device=self.device,
                    state_dim=self.model.obs_feature_dim * self.model.n_obs_steps,
                    feature_dim=self.model.obs_feature_dim,
                    action_dim=self.model.action_dim,
                    q_hidden_dim=self.cfg.critic.q_hidden_dim,
                    q_depth=self.cfg.critic.q_depth,
                    Q_lr=self.cfg.ppo.ft_q_lr,
                    target_update_freq=self.cfg.critic.target_update_freq,
                    tau=self.cfg.critic.tau,
                    gamma=self.cfg.critic.gamma,
                    v_hidden_dim=self.cfg.critic.v_hidden_dim,
                    v_depth=self.cfg.critic.v_depth,
                    v_lr=self.cfg.ppo.ft_v_lr,
                    omega=self.cfg.ppo.iql_omega,
                    is_double_q=self.cfg.critic.is_double_q,
                    dp3_normalizer=self.model.normalizer,
                    obs_encoder=online_iql_encoder,
                    n_obs_steps=self.model.n_obs_steps,
                    is_share_encoder=self.cfg.ppo.is_share_iql_encoder,
                    use_pc_color=self.model.use_pc_color,
                    use_action_embed=self.cfg.use_action_embed,
                    fix_encoder=self.cfg.ppo.fix_iql_encoder,
                    encoder_update_with=self.cfg.ppo.iql_encoder_update_with,
                    n_action_steps=self.cfg.n_action_steps,
                    chunk_as_single_action=self.cfg.chunk_as_single_action,
                    use_conv_action_embed=getattr(self.cfg, 'use_conv_action_embed', False),
                    conv_hidden_dims=getattr(self.cfg, 'conv_hidden_dims', [128, 256]),
                    conv_latent_cz=getattr(self.cfg, 'conv_latent_cz', 32),
                    conv_kernel_size=getattr(self.cfg, 'conv_kernel_size', 5),
                    conv_n_groups=getattr(self.cfg, 'conv_n_groups', 8),
                    action_recon_beta=getattr(self.cfg, 'action_recon_beta', 0.5),
                    q_layer_norm=getattr(self.cfg.critic, 'q_layer_norm', False),
                    action_embed_layer_norm=getattr(self.cfg.critic, 'action_embed_layer_norm', False),
                    action_scale_norm=getattr(self.cfg.critic, 'action_scale_norm', False),
                )
                iql_online.eval_with_raw_obs = True
                if os.path.exists(Q_bc_path):
                    if self.cfg.critic.load_pretrain:
                        if self.cfg.critic.is_share_encoder:
                            encoder_path = os.path.join(critic_artifact_dir, 'encoder.pt')
                        else:
                            encoder_path = None
                        online_encoder_path = encoder_path if encoder_path and os.path.exists(encoder_path) else None
                        if not self.cfg.ppo.is_share_iql_encoder:
                            iql_online.load_with_encoder(Q_bc_path, value_path, online_encoder_path)
                        else:
                            iql_online.load(
                                Q_bc_path,
                                value_path,
                                online_encoder_path,
                                force_load=online_encoder_path is not None,
                            )
                        cprint('load Q_bc and value for online iql finetuning successfully', 'green')
                if _IQLFT_RESTORE_RNG_AT_CONSTRUCT:
                    _iqlft_restore_rng(_iqlft_construct_snapshot)
            else:
                iql_online = None
                
            from rl_100.training.online_rl import online_finetune
            online_finetune(self, dynamics, Q_bc, value, iql, iql_online, copy_encoder, wandb, self.ema)

    def eval(self, online=False, eval_times=1, use_cm=False, distill2mean=False, policy_override=None, eval_name='Eval'):
        env_runner = self.env_runner
        if policy_override is not None:
            policy = policy_override
        elif online:
            policy = self.unio4._policy
        else:
            if self.cfg.training.use_ema:
                policy = self.ema_model
            else:
                policy = self.model

        saved_force_stochastic = []
        if hasattr(policy, 'obs_encoder') and hasattr(policy.obs_encoder, 'force_stochastic'):
            saved_force_stochastic.append((policy.obs_encoder, policy.obs_encoder.force_stochastic))
            policy.obs_encoder.force_stochastic = False

        policy.eval()
        eval_env_num = getattr(self.cfg.ppo, 'eval_env_num', 1)
        try:
            run_params = inspect.signature(env_runner.run).parameters
        except (TypeError, ValueError):
            run_params = {}
        run_kwargs = {
            'use_cm': use_cm,
            'distill2mean': distill2mean,
            'eval_env_num': eval_env_num,
        }
        if getattr(self.cfg, 'data_collect', False):
            run_kwargs['data_collect'] = True
            run_kwargs['traj_path'] = getattr(self.cfg, 'traj_path', None) or os.path.join(self.output_dir, 'rollouts')
        run_kwargs = {k: v for k, v in run_kwargs.items() if k in run_params}
        log_data = {'test_mean_score': [], 'mean_returns': []}
        try:
            for _ in range(eval_times):
                runner_log = env_runner.run(policy, **run_kwargs)
                log_data['test_mean_score'].append(runner_log['test_mean_score'])
                log_data['mean_returns'].append(runner_log['mean_returns'])

                cprint(f"---------------- {eval_name} Results --------------", 'magenta')
                for key, value in runner_log.items():
                    if isinstance(value, float):
                        cprint(f"{key}: {value:.4f}", 'magenta')
        finally:
            for encoder, prev_value in reversed(saved_force_stochastic):
                encoder.force_stochastic = prev_value
        log_data['test_mean_score'] = np.mean(log_data['test_mean_score'])
        log_data['mean_returns'] = np.mean(log_data['mean_returns'])
        print(f'{eval_name} average success rates:', np.mean(log_data['test_mean_score']))
        print(f'{eval_name} average rewards:', np.mean(log_data['mean_returns']))
        return log_data

    def unio4_eval(self, idql_eval: bool = False, dynamics = None, first_action = False, get_np = True, use_gae = True, iql = None, Q = None, repeat_num = 100, eval_times: int = 1, use_cm=False, distill2mean=False, eval_name: str = 'IDQL Eval'):
        cfg = copy.deepcopy(self.cfg)
        env_runner = self.env_runner
        policy = self.unio4._policy
        if cfg.training.use_ema:
            if cfg.unio4.use_ema_eval:
                policy = self.ema_model

        saved_force_stochastic = []
        seen_encoders = set()

        def _disable_force_stochastic(encoder):
            if encoder is None or not hasattr(encoder, 'force_stochastic'):
                return
            encoder_id = id(encoder)
            if encoder_id in seen_encoders:
                return
            seen_encoders.add(encoder_id)
            saved_force_stochastic.append((encoder, encoder.force_stochastic))
            encoder.force_stochastic = False

        _disable_force_stochastic(getattr(policy, 'obs_encoder', None))
        if idql_eval and iql is not None:
            _disable_force_stochastic(getattr(iql, 'obs_encoder', None))
            for net_name in ['_Q', '_target_Q', '_value']:
                net = getattr(iql, net_name, None)
                _disable_force_stochastic(getattr(net, '_obs_encoder', None))
        if idql_eval and Q is not None:
            _disable_force_stochastic(getattr(Q, '_obs_encoder', None))
            _disable_force_stochastic(getattr(Q, 'obs_encoder', None))

        policy.eval()
        eval_env_num = getattr(self.cfg.ppo, 'eval_env_num', 1)
        try:
            idql_run_params = inspect.signature(env_runner.idql_run).parameters
        except (TypeError, ValueError):
            idql_run_params = {}
        try:
            run_params = inspect.signature(env_runner.run).parameters
        except (TypeError, ValueError):
            run_params = {}
        log_data = {'test_mean_score': [], 'mean_returns': []}
        try:
            for i in tqdm.tqdm(range(eval_times), desc='evaluating ......'):
                if idql_eval:
                    idql_kwargs = {
                        'dynamics': dynamics,
                        'first_action': first_action,
                        'get_np': get_np,
                        'use_gae': use_gae,
                        'iql': iql,
                        'Q': Q,
                        'repeat_num': repeat_num,
                        'use_cm': use_cm,
                        'distill2mean': distill2mean,
                        'eval_env_num': eval_env_num,
                    }
                    idql_kwargs = {k: v for k, v in idql_kwargs.items() if k in idql_run_params}
                    runner_log = env_runner.idql_run(policy, **idql_kwargs)
                else:
                    run_kwargs = {
                        'use_cm': use_cm,
                        'distill2mean': distill2mean,
                        'eval_env_num': eval_env_num,
                    }
                    run_kwargs = {k: v for k, v in run_kwargs.items() if k in run_params}
                    runner_log = env_runner.run(policy, **run_kwargs)
                cprint(f"---------------- {eval_name} Results --------------", 'magenta')
                for key, value in runner_log.items():
                    if isinstance(value, float):
                        cprint(f"{key}: {value:.4f}", 'magenta')
                log_data['test_mean_score'].append(runner_log['test_mean_score'])
                log_data['mean_returns'].append(runner_log['mean_returns'])
        finally:
            for encoder, prev_value in reversed(saved_force_stochastic):
                encoder.force_stochastic = prev_value
        log_data['test_mean_score'] = np.mean(log_data['test_mean_score'])
        log_data['mean_returns'] = np.mean(log_data['mean_returns'])
        print(f'{eval_name} average success rates:', log_data['test_mean_score'])
        print(f'{eval_name} average rewards:', np.mean(log_data['mean_returns']))
        return log_data

    def value_decay(self, initial_value, total_steps, max_train_steps, min_value=0.1):
        value_now = initial_value * (1 - total_steps / max_train_steps)
        return np.clip(value_now, a_min=min_value, a_max=None)

    def load_online_checkpoints(self, online_ft_path, iql=None, value_net=None, ema=None):
        self.online_update_num_path = os.path.join(online_ft_path, 'update_num.txt')
        update_num = np.loadtxt(self.online_update_num_path, dtype=int)
        self.online_policy_cp_path = os.path.join(online_ft_path, 'policy', 'update_{}'.format(update_num))
        self.online_value_cp_path = os.path.join(online_ft_path, 'value', 'update_{}'.format(update_num))
        self.online_iql_cp_path = os.path.join(online_ft_path, 'iql', 'update_{}'.format(update_num))
        self.online_lr_cp_path = os.path.join(online_ft_path, 'lr', 'update_{}'.format(update_num), 'lr.txt')
        self.online_distilled_cp_path = os.path.join(online_ft_path, 'distilled', 'update_{}'.format(update_num))
        self.online_update_num = np.loadtxt(self.online_update_num_path, dtype=int)
        
        self.unio4.load(self.online_policy_cp_path)
        if hasattr(self.unio4, "critic"):
            self.unio4.load_critic(self.online_value_cp_path)
        else:
            value_net.load_state_dict(torch.load(os.path.join(self.online_value_cp_path, 'critic.pth')))
            cprint('2. load value net from {}'.format(self.online_value_cp_path), 'green')
            
        if self.cfg.ppo.iql_ft:
            online_encoder_path = os.path.join(self.online_iql_cp_path, 'encoder.pth')
            if not os.path.exists(online_encoder_path):
                if not self.cfg.ppo.fix_iql_encoder:
                    raise FileNotFoundError('trainable online IQL encoder checkpoint is required: {}'.format(online_encoder_path))
                online_encoder_path = None
            iql.load(
                v_path=os.path.join(self.online_iql_cp_path, 'value.pth'),
                q_path=os.path.join(self.online_iql_cp_path, 'Q_bc.pth'),
                encoder_path=online_encoder_path,
                force_load=online_encoder_path is not None
            )
            cprint('3. load iql from {}'.format(self.online_iql_cp_path), 'green')
            
        self.unio4._policy.distilled_model.load_state_dict(torch.load(os.path.join(self.online_distilled_cp_path, 'distilled.pth')))
        cprint('4. load distilled model from {}'.format(self.online_distilled_cp_path), 'green')
        lr_a, lr_c = np.loadtxt(self.online_lr_cp_path, dtype=float)
        cprint('5. load learning rate from {}'.format(self.online_lr_cp_path), 'green')
        self.cfg.ppo.lr_a, self.cfg.ppo.lr_c = float(lr_a), float(lr_c)
        cprint('load online checkpoint from {}, and actor and critic learning rate is {} and {}'.format(online_ft_path, update_num, self.cfg.ppo.lr_a, self.cfg.ppo.lr_c), 'green')
        
        if self.cfg.training.use_ema and self.ema_model is not None:
            ema_cp_path = os.path.join(online_ft_path, 'ema', 'update_{}'.format(update_num))
            if os.path.exists(ema_cp_path):
                self.ema_model.load(ema_cp_path)
                cprint('6. load EMA from {}'.format(ema_cp_path), 'green')
                if ema is not None:
                    ema_step_path = os.path.join(ema_cp_path, 'optimization_step.txt')
                    if os.path.exists(ema_step_path):
                        ema.optimization_step = int(np.loadtxt(ema_step_path, dtype=int))
                        cprint('7. load EMA optimization_step from {}'.format(ema_step_path), 'green')
                    else:
                        ema.optimization_step = 0
                        cprint('7. EMA optimization_step not found, reset to 0 for backward compatibility', 'yellow')
            else:
                ema_state = self.ema_model.state_dict()
                policy_state = self.unio4._policy.state_dict()
                filtered_state = {k: v for k, v in policy_state.items() if k in ema_state}
                self.ema_model.load_state_dict(filtered_state, strict=False)
                cprint('6. EMA checkpoint not found, synced from policy', 'yellow')
                
        if value_net is not None:
            return iql, value_net
        else:
            return iql, self.unio4.critic

    def save_online_checkpoints(self, online_ft_path, update_num, iql=None, ema=None):
        self.online_update_num_path = os.path.join(online_ft_path, 'update_num.txt')
        self.online_policy_cp_path = os.path.join(online_ft_path, 'policy', 'update_{}'.format(update_num))
        self.online_value_cp_path = os.path.join(online_ft_path, 'value', 'update_{}'.format(update_num))
        self.online_iql_cp_path = os.path.join(online_ft_path, 'iql', 'update_{}'.format(update_num))
        self.online_distilled_cp_path = os.path.join(online_ft_path, 'distilled', 'update_{}'.format(update_num))
        self.online_lr_cp_path = os.path.join(online_ft_path, 'lr', 'update_{}'.format(update_num))
        
        os.makedirs(self.online_policy_cp_path, exist_ok=True)
        os.makedirs(self.online_value_cp_path, exist_ok=True)
        os.makedirs(self.online_iql_cp_path, exist_ok=True)
        os.makedirs(self.online_lr_cp_path, exist_ok=True)
        os.makedirs(self.online_distilled_cp_path, exist_ok=True)
        
        np.savetxt(self.online_update_num_path, [update_num], fmt='%d', delimiter=',')
        self.unio4.save(self.online_policy_cp_path)
        self.unio4.save_critic(self.online_value_cp_path)
        
        if self.cfg.ppo.iql_ft:
            iql.save(
                v_path=os.path.join(self.online_iql_cp_path, 'value.pth'),
                q_path=os.path.join(self.online_iql_cp_path, 'Q_bc.pth'),
                encoder_path=os.path.join(self.online_iql_cp_path, 'encoder.pth')
            )
        if self.cfg.distill_phase == 'online':
            torch.save(self.unio4._policy.distilled_model.state_dict(), os.path.join(self.online_distilled_cp_path, 'distilled.pth'))
        np.savetxt(os.path.join(self.online_lr_cp_path, 'lr.txt'), [self.unio4.lr_a, self.unio4.lr_c], fmt='%.10f', delimiter=',')
        
        if self.cfg.training.use_ema and self.ema_model is not None:
            ema_cp_path = os.path.join(online_ft_path, 'ema', 'update_{}'.format(update_num))
            os.makedirs(ema_cp_path, exist_ok=True)
            self.ema_model.save(ema_cp_path)
            if ema is not None:
                np.savetxt(os.path.join(ema_cp_path, 'optimization_step.txt'), [ema.optimization_step], fmt='%d', delimiter=',')
        print('save online checkpoint to {}'.format(online_ft_path))

    def _resolve_output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            from hydra.core.hydra_config import HydraConfig
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def sample_batch(self, batch_size: int = 512):
        batch = next(iter(self.train_dataloader))
        return dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

    def sample_finetune_batch(self):
        try:
            batch = next(self.finetune_dataloader_iter)
        except StopIteration:
            self.finetune_dataloader_iter = iter(self.finetune_dataloader)
            batch = next(self.finetune_dataloader_iter)
        return dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

    def save_checkpoint(self, path=None, tag='latest', exclude_keys=None, include_keys=None, use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
                
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
            
        del payload
        torch.cuda.empty_cache()
        print(f"Checkpoint saved to {path}")
        return str(path.absolute())

    def get_pretrained_model_path(self, stage1_model_name):
        data_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'third_party', 'VRL3', 'src', 'vrl3data')
        model_folder_path = os.path.join(data_folder_path, "trained_models")
        model_path = os.path.join(model_folder_path, stage1_model_name + '_checkpoint.pth.tar')
        return model_path

    def get_checkpoint_path(self, tag='latest'):
        if tag == 'latest' or tag == 'latest_cm':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag == 'best':
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys and key in self.__dict__:
                try:
                    self.__dict__[key].load_state_dict(value, **kwargs)
                except (RuntimeError, ValueError) as e:
                    print(f"Warning: Ignoring keys in state_dict for {key}: {e}")
                    if 'optimizer' in key.lower() or 'optim' in key.lower():
                        print(f"Skipping optimizer {key} due to parameter group mismatch")
                    continue
            else:
                print(f"Warning: Skipping key '{key}' - not found in model or excluded.")

        for key in include_keys:
            if key in payload['pickles'] and key in self.__dict__:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
            else:
                print(f"Warning: Skipping pickle '{key}' - not found in payload or model.")

    def load_checkpoint(self, path=None, tag='latest', exclude_keys=None, include_keys=None, **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(cls, path, exclude_keys=None, include_keys=None, **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(payload=payload, exclude_keys=exclude_keys, include_keys=include_keys, **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)

    def get_distill_optimizer(self):
        cfg = self.cfg
        cm_optimizer = torch.optim.AdamW(
            self.unio4._policy.distilled_model.parameters(),
            lr=cfg.optimizer.lr,
            betas=(cfg.optimizer.betas[0], cfg.optimizer.betas[1]),
            weight_decay=cfg.optimizer.weight_decay,
            eps=cfg.optimizer.eps)

        cm_lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=cm_optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(cfg.ppo.max_train_steps * cfg.ppo.K_epochs)
        )
        return cm_optimizer, cm_lr_scheduler
