import torch
import torch.nn.functional as F

from rl_100.unidpg.net import ValueMLP, QMLP, DoubleQMLP
from rl_100.unidpg.buffer import OnlineReplayBuffer

from rl_100.model.vision.pointnet_extractor import DP3Encoder
import torch.nn as nn
from rl_100.common.pytorch_util import dict_apply
from copy import deepcopy
class ValueLearner:
    _device: torch.device
    _value: ValueMLP
    _optimizer: torch.optim
    _batch_size: int

    def __init__(
        self, 
        device: torch.device, 
        state_dim: int, 
        hidden_dim: int, 
        depth: int, 
        value_lr: float, 
        dp3_normalizer,
        obs_encoder: DP3Encoder = None,
        n_obs_steps: int = 2,
        use_pc_color: bool = False,
        share_encoder: bool = False
    ) -> None:
        super().__init__()
        # for dp3
        self.use_pc_color = use_pc_color
        self.obs_encoder = obs_encoder
        self.normalizer = dp3_normalizer
        self.n_obs_steps = n_obs_steps
        self.is_share_encoder = share_encoder
        if share_encoder:
            v_obs_encoder = None
        else:
            v_obs_encoder = deepcopy(obs_encoder)  

        # for v
        self._device = device
        self._value = ValueMLP(v_obs_encoder, state_dim, hidden_dim, depth).to(device)

        self._optimizer = torch.optim.Adam(
            self._value.parameters(), 
            lr=value_lr,
            )

    def __call__(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._value(s)


    def update(
        self, batch: dict
    ) -> float:

        nobs = self.normalizer.normalize(batch['obs'])
        batch_size = nobs['agent_pos'].shape[0]
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))

        s, Return = this_nobs, batch['return'][:, self.n_obs_steps - 1]
        if self.is_share_encoder:
            nobs_features = self.obs_encoder(s).reshape(batch_size, -1)
        else:
            nobs_features = s
        value_loss = F.mse_loss(self._value(nobs_features), Return)

        self._optimizer.zero_grad()
        value_loss.backward()
        self._optimizer.step()

        return value_loss.item()


    def save(
        self, path: str
    ) -> None:
        # Handle DDP model saving
        state_dict = self._value.state_dict()
        # Remove 'module.' prefix if present (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        torch.save(new_state_dict, path)
        print('Value parameters saved in {}'.format(path))


    def load(
        self, path: str
    ) -> None:
        self._value.load_state_dict(torch.load(path, map_location=self._device))
        print('Value parameters loaded')



class QLearner(nn.Module):
    _device: torch.device
    _Q: QMLP
    _optimizer: torch.optim
    _target_Q: QMLP
    _total_update_step: int
    _target_update_freq: int
    _tau: float
    _gamma: float
    _batch_size: int

    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        batch_size: int
    ) -> None:
        super().__init__()
        self._device = device
        self._Q = QMLP(state_dim, action_dim, hidden_dim, depth).to(device)
        self._optimizer = torch.optim.Adam(
            self._Q.parameters(),
            lr=Q_lr,
            )

        self._target_Q = QMLP(state_dim, action_dim, hidden_dim, depth).to(device)
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._total_update_step = 0
        self._target_update_freq = target_update_freq
        self._tau = tau

        self._gamma = gamma
        self._batch_size = batch_size


    def __call__(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        return self._Q(s, a)


    def loss(
        self, replay_buffer: OnlineReplayBuffer, pi
    ) -> torch.Tensor:
        raise NotImplementedError


    def update(
        self, replay_buffer: OnlineReplayBuffer, pi
    ) -> float:
        Q_loss = self.loss(replay_buffer, pi)
        self._optimizer.zero_grad()
        Q_loss.backward()
        self._optimizer.step()

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        return Q_loss.item()


    def save(
        self, path: str
    ) -> None:
        # Handle DDP model saving
        state_dict = self._Q.state_dict()
        # Remove 'module.' prefix if present (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        torch.save(new_state_dict, path)
        print('Q function parameters saved in {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._Q.load_state_dict(torch.load(path, map_location=self._device))
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')


class QSarsaLearner(nn.Module):
    _device: torch.device
    _Q: QMLP
    _optimizer: torch.optim
    _target_Q: QMLP
    _total_update_step: int
    _target_update_freq: int
    _tau: float
    _gamma: float

    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        dp3_normalizer,
        obs_encoder: DP3Encoder = None,
        n_obs_steps: int = 2,
        is_share_encoder: bool = True,
        use_pc_color: bool = False
    ) -> None:
        super().__init__()

        # for dp3
        self.use_pc_color = use_pc_color
        self.obs_encoder = obs_encoder
        self.normalizer = dp3_normalizer
        self.n_obs_steps = n_obs_steps
        self.is_share_encoder = is_share_encoder
        if is_share_encoder:
            q1_obs_encoder = None
            q2_obs_encoder = None
        else:
            q1_obs_encoder = deepcopy(obs_encoder)
            q2_obs_encoder = deepcopy(obs_encoder)

        self._device = device

        self._Q = QMLP(q1_obs_encoder, state_dim, action_dim, hidden_dim, depth).to(device)
        
        if is_share_encoder:
            self._q_optimizer = torch.optim.Adam(
                list(self._Q.parameters()) + list(self.obs_encoder.parameters()),
                lr=Q_lr,
                )
        else:
            self._optimizer = torch.optim.Adam(
                self._Q.parameters(),
                lr=Q_lr,
                )

        self._target_Q = QMLP(q1_obs_encoder, state_dim, action_dim, hidden_dim, depth).to(device)
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._total_update_step = 0
        self._target_update_freq = target_update_freq
        self._tau = tau

        self._gamma = gamma

    def __call__(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        return self._Q(s, a)


    def update(
        self, batch: dict
    ) -> float:
        
        nobs = self.normalizer.normalize(batch['obs'])
        next_nobs = self.normalizer.normalize(batch['next_obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        next_nactions = self.normalizer['next_action'].normalize(batch['next_action'])
        batch_size = nactions.shape[0]
        if not self.use_pc_color and 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            next_nobs['point_cloud'] = next_nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        

        next_this_nobs = dict_apply(next_nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))

        if self.is_share_encoder:
            nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, -1)
            next_nobs_features = self.obs_encoder(next_this_nobs).reshape(batch_size, -1)
        else:
            nobs_features = this_nobs
            next_nobs_features = next_this_nobs

        s, a, r, s_p, a_p, not_done = nobs_features, nactions[:, self.n_obs_steps - 1], \
            batch['reward'], next_nobs_features, next_nactions[:, self.n_obs_steps - 1], batch['not_done']
        
        # s, a, r, s_p, a_p, not_done, _, _ = replay_buffer.sample(self._batch_size)
        with torch.no_grad():
            target_Q_value = r + not_done * self._gamma * self._target_Q(s_p, a_p)
        
        Q = self._Q(s, a)
        Q_loss = F.mse_loss(Q, target_Q_value)
        self._optimizer.zero_grad()
        Q_loss.backward()
        self._optimizer.step()

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        return Q_loss.item()


    def save(
        self, path: str
    ) -> None:
        # Handle DDP model saving
        state_dict = self._Q.state_dict()
        # Remove 'module.' prefix if present (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        torch.save(new_state_dict, path)
        print('Q function parameters saved in {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._Q.load_state_dict(torch.load(path, map_location=self._device))
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')


class QPiLearner(QLearner):
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        batch_size: int
    ) -> None:
        super().__init__(
        device = device,
        state_dim = state_dim,
        action_dim = action_dim,
        hidden_dim = hidden_dim,
        depth = depth,
        Q_lr = Q_lr,
        target_update_freq = target_update_freq,
        tau = tau,
        gamma = gamma,
        batch_size = batch_size
        )


    def loss(
        self, replay_buffer: OnlineReplayBuffer, pi
    ) -> torch.Tensor:
        s, a, r, s_p, _, not_done, _, _ = replay_buffer.sample(self._batch_size)
        a_p = pi.select_action(s_p, is_sample=True)
        with torch.no_grad():
            target_Q_value = r + not_done * self._gamma * self._target_Q(s_p, a_p)
        
        Q = self._Q(s, a)
        loss = F.mse_loss(Q, target_Q_value)
        
        return loss



class IQL_Q_V_no(nn.Module):
    def __init__(
        self,
        device: torch.device,
        state_dim: int, # obs_feature_dim * n_obs_steps
        feature_dim: int,
        action_dim: int, # single action dim
        q_hidden_dim: int,
        q_depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        v_hidden_dim: int,
        v_depth: int,
        v_lr: float,
        omega: float,
        is_double_q: bool,
        dp3_normalizer,
        obs_encoder: DP3Encoder = None,
        n_obs_steps: int = 2,
        is_share_encoder: bool = True,
        use_pc_color: bool = False,
        use_action_embed: bool = False,
        fix_encoder: bool = False,
        action_norm: bool = True,
        encoder_update_with: str = "value",  # 新增参数: "value", "q", "both"
        n_action_steps: int = 1,
        chunk_as_single_action: bool = False,
        use_conv_action_embed: bool = False,
        conv_hidden_dims: list = [128, 256],
        conv_latent_cz: int = 32,
        conv_kernel_size: int = 5,
        conv_n_groups: int = 8,
        action_recon_beta: float = 0.5,
        q_layer_norm: bool = False,
        action_embed_layer_norm: bool = False,
        action_scale_norm: bool = False,
    ) -> None:

        super().__init__()
        self._device = device
        self._omega = omega
        self._is_double_q = is_double_q
        # for dp3
        self.use_pc_color = use_pc_color
        self.obs_encoder = obs_encoder
        self.normalizer = dp3_normalizer
        self.n_obs_steps = n_obs_steps
        self.is_share_encoder = is_share_encoder
        self.fix_encoder = fix_encoder
        self.action_norm = action_norm
        self.encoder_update_with = encoder_update_with  # 新增属性
        self.state_dim = state_dim  # 添加state_dim属性
        self.use_action_embed = use_action_embed  # 添加use_action_embed属性
        self.use_conv_action_embed = use_conv_action_embed
        self.action_recon_beta = action_recon_beta
        self.q_layer_norm = q_layer_norm
        self.action_embed_layer_norm = action_embed_layer_norm
        self.action_scale_norm = action_scale_norm
        self.chunk_as_single_action = chunk_as_single_action
        if chunk_as_single_action:
            self.n_action_steps = n_action_steps
        else:
            self.n_action_steps = 1
        single_action_dim = action_dim
        action_dim = action_dim * self.n_action_steps if chunk_as_single_action else action_dim
        self.action_dim = action_dim
        if is_share_encoder:
            q1_obs_encoder = None
            q2_obs_encoder = None
            v_obs_encoder = None

        else:
            q1_obs_encoder = deepcopy(obs_encoder)
            q2_obs_encoder = deepcopy(obs_encoder)
            v_obs_encoder = deepcopy(obs_encoder)

        # conv action embed kwargs
        conv_kwargs = dict(
            use_conv_action_embed=use_conv_action_embed,
            single_action_dim=single_action_dim,
            n_action_steps=n_action_steps,
            conv_hidden_dims=conv_hidden_dims,
            conv_latent_cz=conv_latent_cz,
            conv_kernel_size=conv_kernel_size,
            conv_n_groups=conv_n_groups,
            q_layer_norm=q_layer_norm,
            action_embed_layer_norm=action_embed_layer_norm,
            action_scale_norm=action_scale_norm,
        )

        #for q
        if is_double_q:
            print('using double q learning')
            self._Q = DoubleQMLP(use_action_embed, q1_obs_encoder, state_dim, feature_dim, action_dim, q_hidden_dim, q_depth, fix_encoder=fix_encoder, **conv_kwargs).to(device)
            self._target_Q = DoubleQMLP(use_action_embed, q2_obs_encoder, state_dim, feature_dim, action_dim, q_hidden_dim, q_depth, fix_encoder=fix_encoder, **conv_kwargs).to(device)
        else:
            self._Q = QMLP(use_action_embed, q1_obs_encoder, state_dim, feature_dim, action_dim, q_hidden_dim, q_depth, fix_encoder=fix_encoder, **conv_kwargs).to(device)
            self._target_Q = QMLP(use_action_embed, q2_obs_encoder, state_dim, feature_dim, action_dim, q_hidden_dim, q_depth, fix_encoder=fix_encoder, **conv_kwargs).to(device)


        
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._total_update_step = 0
        self._target_update_freq = target_update_freq
        self._tau = tau
        self._gamma = gamma
        #for v
        self._value = ValueMLP(v_obs_encoder, state_dim, v_hidden_dim, v_depth, n_obs_steps, fix_encoder=fix_encoder).to(device)
        if is_share_encoder and not self.fix_encoder:
            # 根据encoder_update_with参数决定encoder由哪个优化器更新
            if encoder_update_with == "value":
                print('-----------encoder will be updated by value loss-----------')
                # 用value loss更新encoder
                self._v_optimizer = torch.optim.Adam(
                    list(self._value.parameters()) + list(self.obs_encoder.parameters()), 
                    lr=v_lr,
                )
                self._q_optimizer = torch.optim.Adam(
                    self._Q.parameters(),
                    lr=Q_lr,
                )
            elif encoder_update_with == "q":
                print('-----------encoder will be updated by q loss-----------')
                # 用q loss更新encoder
                self._v_optimizer = torch.optim.Adam(
                    self._value.parameters(), 
                    lr=v_lr,
                )
                self._q_optimizer = torch.optim.Adam(
                    list(self._Q.parameters()) + list(self.obs_encoder.parameters()),
                    lr=Q_lr,
                )
            elif encoder_update_with == "both":
                print('-----------encoder will be updated by both losses using separate optimizer-----------')
                # 创建独立的encoder优化器，可以同时从两个损失获得梯度
                self._v_optimizer = torch.optim.Adam(
                    self._value.parameters(), 
                    lr=v_lr,
                )
                self._q_optimizer = torch.optim.Adam(
                    self._Q.parameters(),
                    lr=Q_lr,
                )
                self._encoder_optimizer = torch.optim.Adam(
                    self.obs_encoder.parameters(),
                    lr=min(v_lr, Q_lr),  # 使用较小的学习率
                )
            else:
                raise ValueError(f"encoder_update_with must be 'value', 'q', or 'both', got {encoder_update_with}")
        else:
            print('-----------not sharing encoder or fix encoder from dp3-----------')
            self._v_optimizer = torch.optim.Adam(
                self._value.parameters(), 
                lr=v_lr,
            )
            self._q_optimizer = torch.optim.Adam(
                self._Q.parameters(),
                lr=Q_lr,
            )
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.fix_encoder:
            if self.is_share_encoder and self.obs_encoder is not None:
                self.obs_encoder.eval()
            elif not self.is_share_encoder:
                if hasattr(self._Q, '_obs_encoder') and self._Q._obs_encoder is not None:
                    self._Q._obs_encoder.eval()
                if hasattr(self._target_Q, '_obs_encoder') and self._target_Q._obs_encoder is not None:
                    self._target_Q._obs_encoder.eval()
                if hasattr(self._value, '_obs_encoder') and self._value._obs_encoder is not None:
                    self._value._obs_encoder.eval()
        return self

    def obs2nobs(self, obs: dict):
        nobs = self.normalizer.normalize(obs)
        batch_size = nobs['agent_pos'].shape[0]
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        return this_nobs
    def minQ(self, s: torch.Tensor, a: torch.Tensor):
        if not self.is_share_encoder:
            if isinstance(s, dict):
                if len(s['point_cloud'].shape) != 3:
                    s = self.obs2nobs(s)
        else:
            if isinstance(s, dict):
                batch_size = s['agent_pos'].shape[0]
                s = self.obs2nobs(s)
                if self.fix_encoder:
                    self.obs_encoder.eval()
                    with torch.no_grad():
                        s = self.obs_encoder(s).reshape(batch_size, -1)
                else:
                    s = self.obs_encoder(s).reshape(batch_size, -1)
        if self.chunk_as_single_action:
            a = a.reshape(-1, self.action_dim)
        Q1, Q2 = self._Q(s, a)
        return torch.min(Q1, Q2)

    def target_minQ(self, s: torch.Tensor, a: torch.Tensor):
        if not self.is_share_encoder:
            if self.chunk_as_single_action:
                a = a.reshape(-1, self.action_dim)
            # 对于不共享encoder，target_Q网络也有自己的encoder
            if isinstance(s, dict):
                s_features = self._target_Q._obs_encoder(s).reshape(-1, self.state_dim)
                a_encoded, _ = self._target_Q.encode_action(a)
                sa = torch.cat([s_features, a_encoded], dim=1)
                Q1 = self._target_Q._net1(sa)
                Q2 = self._target_Q._net2(sa)
            else:
                a_encoded, _ = self._target_Q.encode_action(a)
                sa = torch.cat([s, a_encoded], dim=1)
                Q1 = self._target_Q._net1(sa)
                Q2 = self._target_Q._net2(sa)
        else:
            Q1, Q2 = self._target_Q(s, a)
        return torch.min(Q1, Q2)
    
    def expectile_loss(self, loss: torch.Tensor)->torch.Tensor:
        weight = torch.where(loss > 0, self._omega, (1 - self._omega))
        return weight * (loss**2)

    def _as_column(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.shape[1] != 1:
            raise ValueError(f"{name} must be scalar per sample, got shape {tensor.shape}")
        return tensor

    def update(self, batch: dict, online: bool = False, pre_cut=False, online_recon=False) -> float:
        if self.fix_encoder:
            if self.is_share_encoder and self.obs_encoder is not None:
                self.obs_encoder.eval()
            elif not self.is_share_encoder:
                if hasattr(self._Q, '_obs_encoder') and self._Q._obs_encoder is not None:
                    self._Q._obs_encoder.eval()
                if hasattr(self._target_Q, '_obs_encoder') and self._target_Q._obs_encoder is not None:
                    self._target_Q._obs_encoder.eval()
                if hasattr(self._value, '_obs_encoder') and self._value._obs_encoder is not None:
                    self._value._obs_encoder.eval()
        
        nobs = self.normalizer.normalize(batch['obs'])
        next_nobs = self.normalizer.normalize(batch['next_obs'])
        if online:
            nactions = batch['action']
        else:
            if self.action_norm:
                nactions = self.normalizer['action'].normalize(batch['action'])
            else:
                nactions = batch['action']
        # next_nactions = self.normalizer['next_action'].normalize(batch['next_action'])
        batch_size = nactions.shape[0]
        if not self.use_pc_color and 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            next_nobs['point_cloud'] = next_nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        
        if self.chunk_as_single_action:
            next_this_nobs = dict_apply(next_nobs,
                lambda x: x[:, -self.n_obs_steps:, ...].reshape(-1, *x.shape[2:]))
        else:
            next_this_nobs = dict_apply(next_nobs,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))

        vib_recon_loss = 0
        q_recon_loss = 0
        v_recon_loss = 0
        
        if self.is_share_encoder:
            # 共享encoder的情况 - 原有逻辑
            if self.fix_encoder:
                with torch.no_grad():
                    nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, -1)
                    next_nobs_features = self.obs_encoder(next_this_nobs).reshape(batch_size, -1)
            else:
                if online_recon:
                    cricit_vib_recon_loss, critic_recon_loss_items, nobs_features = self.obs_encoder.Recon_VIB_loss(this_nobs)
                    next_cricit_vib_recon_loss, critic_recon_loss_items, next_nobs_features = self.obs_encoder.Recon_VIB_loss(next_this_nobs)
                    nobs_features, next_nobs_features = nobs_features.reshape(batch_size, -1), next_nobs_features.reshape(batch_size, -1)
                    vib_recon_loss = cricit_vib_recon_loss + next_cricit_vib_recon_loss
                    vib_recon_loss = vib_recon_loss.mean()
                else:
                    nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, -1)
                    next_nobs_features = self.obs_encoder(next_this_nobs).reshape(batch_size, -1)
        else:
            # 不共享encoder的情况 - 新增逻辑
            if self.fix_encoder:
                # 如果fix encoder，使用无梯度计算
                with torch.no_grad():
                    if self._is_double_q:
                        # 对于double Q，使用Q网络的encoder
                        nobs_features_q = self._Q._obs_encoder(this_nobs).reshape(batch_size, -1)
                        next_nobs_features_q = self._Q._obs_encoder(next_this_nobs).reshape(batch_size, -1)
                    else:
                        nobs_features_q = self._Q._obs_encoder(this_nobs).reshape(batch_size, -1)
                        next_nobs_features_q = self._Q._obs_encoder(next_this_nobs).reshape(batch_size, -1)
                    
                    # 对于V网络
                    nobs_features_v = self._value._obs_encoder(this_nobs).reshape(batch_size, -1)
                    next_nobs_features_v = self._value._obs_encoder(next_this_nobs).reshape(batch_size, -1)
                    
                    # 为了保持接口一致，使用Q网络的特征作为主要特征
                    nobs_features = nobs_features_q
                    next_nobs_features = next_nobs_features_q
            else:
                # 如果不fix encoder，并且需要reconstruction loss
                if online_recon:
                    # 计算Q网络encoder的reconstruction loss
                    if self._is_double_q:
                        q_cricit_vib_recon_loss, q_critic_recon_loss_items, nobs_features_q = self._Q._obs_encoder.Recon_VIB_loss(this_nobs)
                        next_q_cricit_vib_recon_loss, next_q_critic_recon_loss_items, next_nobs_features_q = self._Q._obs_encoder.Recon_VIB_loss(next_this_nobs)
                        q_recon_loss = (q_cricit_vib_recon_loss + next_q_cricit_vib_recon_loss).mean()
                    else:
                        q_cricit_vib_recon_loss, q_critic_recon_loss_items, nobs_features_q = self._Q._obs_encoder.Recon_VIB_loss(this_nobs)
                        next_q_cricit_vib_recon_loss, next_q_critic_recon_loss_items, next_nobs_features_q = self._Q._obs_encoder.Recon_VIB_loss(next_this_nobs)
                        q_recon_loss = (q_cricit_vib_recon_loss + next_q_cricit_vib_recon_loss).mean()
                    
                    # 计算V网络encoder的reconstruction loss
                    v_cricit_vib_recon_loss, v_critic_recon_loss_items, nobs_features_v = self._value._obs_encoder.Recon_VIB_loss(this_nobs)
                    next_v_cricit_vib_recon_loss, next_v_critic_recon_loss_items, next_nobs_features_v = self._value._obs_encoder.Recon_VIB_loss(next_this_nobs)
                    v_recon_loss = (v_cricit_vib_recon_loss + next_v_cricit_vib_recon_loss).mean()
                    
                    # 重塑特征维度
                    nobs_features_q = nobs_features_q.reshape(batch_size, -1)
                    next_nobs_features_q = next_nobs_features_q.reshape(batch_size, -1)
                    nobs_features_v = nobs_features_v.reshape(batch_size, -1)
                    next_nobs_features_v = next_nobs_features_v.reshape(batch_size, -1)
                    
                    # 为了保持接口一致，使用Q网络的特征作为主要特征
                    nobs_features = nobs_features_q
                    next_nobs_features = next_nobs_features_q
                else:
                    # 不需要reconstruction loss，直接前向传播
                    if self._is_double_q:
                        nobs_features_q = self._Q._obs_encoder(this_nobs).reshape(batch_size, -1)
                        next_nobs_features_q = self._Q._obs_encoder(next_this_nobs).reshape(batch_size, -1)
                    else:
                        nobs_features_q = self._Q._obs_encoder(this_nobs).reshape(batch_size, -1)
                        next_nobs_features_q = self._Q._obs_encoder(next_this_nobs).reshape(batch_size, -1)
                    
                    nobs_features_v = self._value._obs_encoder(this_nobs).reshape(batch_size, -1)
                    next_nobs_features_v = self._value._obs_encoder(next_this_nobs).reshape(batch_size, -1)
                    
                    nobs_features = nobs_features_q
                    next_nobs_features = next_nobs_features_q
        if not pre_cut:
            if self.chunk_as_single_action:
                action_start = self.n_obs_steps - 1
                action_end = action_start + self.n_action_steps
                if nactions.shape[1] < action_end:
                    raise ValueError(
                        f"chunk critic requires action horizon >= {action_end}, "
                        f"got {nactions.shape[1]}"
                    )

                reward_chunk = batch['reward'][:, action_start:action_end]
                reward_chunk = reward_chunk.reshape(batch_size, self.n_action_steps, -1)
                if reward_chunk.shape[-1] != 1:
                    raise ValueError(
                        f"chunk reward must be scalar per step, got shape {reward_chunk.shape}"
                    )
                reward_chunk = reward_chunk.squeeze(-1)
                
                gamma_weights = torch.pow(self._gamma, torch.arange(self.n_action_steps, device=reward_chunk.device, dtype=reward_chunk.dtype))
                discounted_reward = torch.sum(reward_chunk * gamma_weights, dim=-1, keepdim=True)
                
                s, a, r, s_p, not_done = (
                    nobs_features,
                    nactions[:, action_start:action_end],
                    discounted_reward,
                    next_nobs_features,
                    self._as_column(batch['not_done'][:, action_end - 1], 'not_done'),
                )
            else:
                action_idx = self.n_obs_steps - 1
                s, a, r, s_p, not_done = (
                    nobs_features,
                    nactions[:, action_idx],
                    self._as_column(batch['reward'][:, action_idx], 'reward'),
                    next_nobs_features,
                    self._as_column(batch['not_done'][:, action_idx], 'not_done'),
                )
        else:
            s, a, r, s_p, not_done = nobs_features, nactions, batch['reward'], next_nobs_features, batch['not_done'] # TODO revise chunk_as_single_action for online later
            r = self._as_column(r, 'reward')
            not_done = self._as_column(not_done, 'not_done')
        
        # Compute value loss
        with torch.no_grad():
            self._target_Q.eval()
            if self.is_share_encoder:
                if self._is_double_q:
                    target_q = self.target_minQ(s, a)
                else:
                    target_q = self._target_Q(s, a)
            else:
                if self._is_double_q:
                    target_q = self.target_minQ(this_nobs, a)
                else:
                    target_a = a.reshape(-1, self.action_dim) if self.chunk_as_single_action else a
                    target_features = self._target_Q._obs_encoder(this_nobs).reshape(-1, self.state_dim)
                    target_action, _ = self._target_Q.encode_action(target_a)
                    target_q = self._target_Q._net(torch.cat([target_features, target_action], dim=1))
        
        # 对于不共享encoder的情况，V网络需要使用自己的特征
        if not self.is_share_encoder:
            if not pre_cut:
                s_v, s_p_v = nobs_features_v, next_nobs_features_v
            else:
                s_v, s_p_v = nobs_features_v, next_nobs_features_v
            # 直接调用内部网络，避免重复encoder处理
            value = self._value._net(s_v)
        else:
            value = self._value(s)
            
        value_loss = self.expectile_loss(target_q - value).mean()
        
        # 添加reconstruction loss到value loss
        if online_recon and not self.fix_encoder:
            if self.is_share_encoder and self.encoder_update_with == "value":
                value_loss += vib_recon_loss
            elif not self.is_share_encoder:
                # 对于不共享encoder，总是将V网络的reconstruction loss加到value loss
                value_loss += v_recon_loss

        self._v_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self._v_optimizer.step()

        # Compute critic loss
        with torch.no_grad():
            self._value.eval()
            if not self.is_share_encoder:
                # 直接调用内部网络，避免重复encoder处理
                next_v = self._value._net(s_p_v)
            else:
                next_v = self._value(s_p)
        # Discount must match the transition span. chunk_as_single_action treats the
        # whole chunk as one step (reward is the discounted chunk sum, next_obs is
        # n_action_steps ahead) -> gamma**n_action_steps. The non-chunk path uses a
        # single-step transition (single-step reward, next_obs one step ahead) -> gamma.
        discount_steps = self.n_action_steps if self.chunk_as_single_action else 1
        target_q = r + not_done * (self._gamma ** discount_steps) * next_v
        action_recon_loss = None
        if self._is_double_q:
            # 对于不共享encoder的情况，直接调用内部网络
            if not self.is_share_encoder:
                if self.chunk_as_single_action:
                    a = a.reshape(-1, self.action_dim)
                q_model = self._Q.module if hasattr(self._Q, 'module') else self._Q
                a_encoded, a_recon = q_model.encode_action(a)
                if self.use_conv_action_embed and a_recon is not None:
                    action_recon_loss = F.mse_loss(a_recon, a.reshape(a.shape[0], -1))
                sa = torch.cat([s, a_encoded], dim=1)
                current_q1 = q_model._net1(sa)
                current_q2 = q_model._net2(sa)
            else:
                if self.use_conv_action_embed:
                    current_q1, current_q2, action_recon_loss = self._Q(
                        s, a, return_action_recon_loss=True)
                else:
                    current_q1, current_q2 = self._Q(s, a)
            q_loss = ((current_q1 - target_q)**2 + (current_q2 - target_q)**2).mean()
        else:
            # 对于不共享encoder的情况，直接调用内部网络
            if not self.is_share_encoder:
                if self.chunk_as_single_action:
                    a = a.reshape(-1, self.action_dim)
                q_model = self._Q.module if hasattr(self._Q, 'module') else self._Q
                a_encoded, a_recon = q_model.encode_action(a)
                if self.use_conv_action_embed and a_recon is not None:
                    action_recon_loss = F.mse_loss(a_recon, a.reshape(a.shape[0], -1))
                sa = torch.cat([s, a_encoded], dim=1)
                Q = q_model._net(sa)
            else:
                if self.use_conv_action_embed:
                    Q, action_recon_loss = self._Q(
                        s, a, return_action_recon_loss=True)
                else:
                    Q = self._Q(s, a)
            q_loss = F.mse_loss(Q, target_q)

        # 添加reconstruction loss到q loss
        if online_recon and not self.fix_encoder:
            if self.is_share_encoder and self.encoder_update_with == "q":
                q_loss += vib_recon_loss
            elif not self.is_share_encoder:
                # 对于不共享encoder，总是将Q网络的reconstruction loss加到q loss
                q_loss += q_recon_loss

        # Conv action AE reconstruction loss
        if self.use_conv_action_embed and action_recon_loss is not None:
            q_loss = q_loss + self.action_recon_beta * action_recon_loss

        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        # 如果使用独立的encoder优化器，需要额外更新encoder
        if (self.is_share_encoder and not self.fix_encoder and 
            self.encoder_update_with == "both" and online_recon):
            # 对encoder进行单独的更新
            self._encoder_optimizer.zero_grad()
            # 计算encoder的总损失(包含重构损失)
            encoder_loss = vib_recon_loss
            encoder_loss.backward()
            self._encoder_optimizer.step()

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)


        return q_loss.detach().cpu().numpy(), value_loss.detach().cpu().numpy()
        
    def get_advantage(self, s, a)->torch.Tensor:
        if self._is_double_q:
            # share_encoder + dict: encode once, pass features to both minQ and V
            if self.is_share_encoder and isinstance(s, dict):
                if len(s['point_cloud'].shape) != 3:
                    s = self.obs2nobs(s)
                if self.fix_encoder:
                    self.obs_encoder.eval()
                    with torch.no_grad():
                        s = self.obs_encoder(s).reshape(-1, self.state_dim)
                else:
                    s = self.obs_encoder(s).reshape(-1, self.state_dim)

            q = self.minQ(s, a)

            if isinstance(s, dict):
                # Only reaches here when not is_share_encoder (share_encoder dicts encoded above)
                if len(s['point_cloud'].shape) != 3:
                    s = self.obs2nobs(s)
                if self.fix_encoder:
                    self._value._obs_encoder.eval()
                    with torch.no_grad():
                        s_features = self._value._obs_encoder(s).reshape(-1, self.state_dim)
                else:
                    s_features = self._value._obs_encoder(s).reshape(-1, self.state_dim)
                v = self._value._net(s_features)
            else:
                # s is already features
                if not self.is_share_encoder:
                    v = self._value._net(s)
                else:
                    v = self._value(s)
            return q - v
        else:
            if not self.is_share_encoder:
                # 处理单Q网络的情况
                if self.chunk_as_single_action:
                    a = a.reshape(-1, self.action_dim)
                if isinstance(s, dict):
                    s_q = self._Q._obs_encoder(s).reshape(-1, self.state_dim)
                    s_v = self._value._obs_encoder(s).reshape(-1, self.state_dim)
                    a_encoded, _ = self._Q.encode_action(a)
                    sa = torch.cat([s_q, a_encoded], dim=1)
                    q = self._Q._net(sa)
                    v = self._value._net(s_v)
                else:
                    a_encoded, _ = self._Q.encode_action(a)
                    sa = torch.cat([s, a_encoded], dim=1)
                    q = self._Q._net(sa)
                    v = self._value._net(s)
                return q - v
            else:
                return self._Q(s, a) - self._value(s)
    def get_online_value_buget(self, cfg):
        if not cfg.ppo.share_encoder:
            print('get online value budget with encoder from offline') 
            value_model = nn.Sequential(
                self.obs_encoder,
                self._value
            )
        else:
            print('get online value budget without encoder from offline') 
            value_model = self._value
        return value_model
    def save(
        self, q_path: str, v_path: str, encoder_path: str
    ) -> None:
        # Handle DDP-wrapped models by saving the underlying module's state_dict
        q_to_save = self._Q.module if hasattr(self._Q, 'module') else self._Q
        v_to_save = self._value.module if hasattr(self._value, 'module') else self._value
        encoder_to_save = self.obs_encoder.module if hasattr(self.obs_encoder, 'module') else self.obs_encoder

        torch.save(q_to_save.state_dict(), q_path)
        print(f'Q function parameters saved in {q_path}')
        
        torch.save(v_to_save.state_dict(), v_path)
        print(f'Value parameters saved in {v_path}')
        
        if not self.fix_encoder and encoder_path is not None:
            torch.save(encoder_to_save.state_dict(), encoder_path)
            print(f'Encoder parameters saved in {encoder_path}')

    def load(
        self, q_path: str, v_path: str, encoder_path: str = None, force_load: bool = False
    ) -> None:
        self._Q.load_state_dict(torch.load(q_path, map_location=self._device))
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')
        self._value.load_state_dict(torch.load(v_path, map_location=self._device))
        print('Value parameters loaded')
        if not self.fix_encoder and encoder_path is not None:
            self.obs_encoder.load_state_dict(torch.load(encoder_path, map_location=self._device))
            print('Value parameters loaded {}'.format(encoder_path))
        elif force_load and encoder_path is not None:
            self.obs_encoder.load_state_dict(torch.load(encoder_path, map_location=self._device))
            print('Value parameters loaded {}'.format(encoder_path))
    def load_with_encoder(self, q_path: str, v_path: str, encoder_path: str = None) -> None:
        q_params = torch.load(q_path, map_location=self._device)
        if encoder_path is not None:
            encoder_params = torch.load(encoder_path, map_location=self._device)
        else:
            encoder_params = self.obs_encoder.state_dict()
        modified_encoder_params = {f"_obs_encoder.{k}": v for k, v in encoder_params.items()}

        q_params.update(modified_encoder_params)
        self._Q.load_state_dict(q_params)
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')

        value_params = torch.load(v_path, map_location=self._device)
        value_params.update(modified_encoder_params)

        self._value.load_state_dict(value_params)
        print('Value parameters loaded')


class IQL_Q_V_online(nn.Module):
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        feature_dim: int,
        action_dim: int,
        q_hidden_dim: int,
        q_depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        v_hidden_dim: int,
        v_depth: int,
        v_lr: float,
        omega: float,
        is_double_q: bool,
        dp3_normalizer,
        obs_encoder: DP3Encoder = None,
        n_obs_steps: int = 2,
        is_share_encoder: bool = True,
        use_pc_color: bool = False,
        use_action_embed: bool = False,
        fix_encoder: bool = False,
        encoder_lr_scale: float = 1.0,
        action_norm: bool = True,
    ) -> None:
        
        super().__init__()
        self._device = device
        self._omega = omega
        self._is_double_q = is_double_q
        # for dp3
        self.use_pc_color = use_pc_color
        self.obs_encoder = obs_encoder
        self.normalizer = dp3_normalizer
        self.n_obs_steps = n_obs_steps
        self.is_share_encoder = is_share_encoder
        self.fix_encoder = fix_encoder
        self.encoder_lr_scale = encoder_lr_scale
        self.action_norm = action_norm
        if is_share_encoder:
            q1_obs_encoder = None
            q2_obs_encoder = None
            v_obs_encoder = None
        else:
            q1_obs_encoder = deepcopy(obs_encoder)
            q2_obs_encoder = deepcopy(obs_encoder)
            v_obs_encoder = deepcopy(obs_encoder)     
        #for q
        if is_double_q:
            print('using double q learning')
            self._Q = DoubleQMLP(use_action_embed, q1_obs_encoder, state_dim, feature_dim, action_dim, q_hidden_dim, q_depth).to(device)
            self._target_Q = DoubleQMLP(use_action_embed, q2_obs_encoder, state_dim, feature_dim, action_dim, q_hidden_dim, q_depth).to(device)
        else:
            self._Q = QMLP(use_action_embed, q1_obs_encoder, state_dim, feature_dim, action_dim, q_hidden_dim, q_depth).to(device)
            self._target_Q = QMLP(use_action_embed, q2_obs_encoder, state_dim, feature_dim, action_dim, q_hidden_dim, q_depth).to(device)


        
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._total_update_step = 0
        self._target_update_freq = target_update_freq
        self._tau = tau
        self._gamma = gamma
        #for v
        self._value = ValueMLP(v_obs_encoder, state_dim, v_hidden_dim, v_depth, n_obs_steps).to(device)
        
            
        if is_share_encoder and not self.fix_encoder:
            self._target_encoder = deepcopy(self.obs_encoder)
            if encoder_lr_scale != 1.0:
                self._q_optimizer = torch.optim.Adam(
                    list(self._Q.parameters()),
                    lr=Q_lr,
                    )
                self._encoder_optimizer = torch.optim.Adam(
                    list(self.obs_encoder.parameters()),
                    lr=Q_lr * encoder_lr_scale,
                    )
            else:
                self._q_optimizer = torch.optim.Adam(
                    list(self._Q.parameters()+ list(self.obs_encoder.parameters())),
                    lr=Q_lr,
                    )


            self._v_optimizer = torch.optim.Adam(
                list(self._value.parameters()), 
                lr=v_lr,
                )
        else:
            print('-----------not sharing encoder or fix encoder from dp3-----------')
            self._v_optimizer = torch.optim.Adam(
                self._value.parameters(), 
                lr=v_lr,
                )
            self._q_optimizer = torch.optim.Adam(
                self._Q.parameters(),
                lr=Q_lr,
                )
    def obs2nobs(self, obs: dict):
        nobs = self.normalizer.normalize(obs)
        batch_size = nobs['agent_pos'].shape[0]
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        return this_nobs
    def minQ(self, s: torch.Tensor, a: torch.Tensor):
        if not self.is_share_encoder:
            s = self.obs2nobs(s)
        else:
            
            batch_size = s['agent_pos'].shape[0]
            s = self.obs2nobs(s)
            s = self.obs_encoder(s).reshape(batch_size, -1)
        Q1, Q2 = self._Q(s, a)
        return torch.min(Q1, Q2)

    def target_minQ(self, s: torch.Tensor, a: torch.Tensor):
        Q1, Q2 = self._target_Q(s, a)
        return torch.min(Q1, Q2)
    
    def expectile_loss(self, loss: torch.Tensor)->torch.Tensor:
        weight = torch.where(loss > 0, self._omega, (1 - self._omega))
        return weight * (loss**2)
    def update(self, batch: dict, online: bool = False) -> float:
        nobs = self.normalizer.normalize(batch['obs'])
        next_nobs = self.normalizer.normalize(batch['next_obs'])
        if online:
            nactions = batch['action']
        else:
            if self.action_norm:
                nactions = self.normalizer['action'].normalize(batch['action'])
            else:
                nactions = batch['action']
        # next_nactions = self.normalizer['next_action'].normalize(batch['next_action'])
        batch_size = nactions.shape[0]
        if not self.use_pc_color and 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            next_nobs['point_cloud'] = next_nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        

        next_this_nobs = dict_apply(next_nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        if self.is_share_encoder:
            if self.fix_encoder:
                with torch.no_grad():
                    nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, -1)
                    next_nobs_features = self.obs_encoder(next_this_nobs).reshape(batch_size, -1)
            else:   
                nobs_features = self._target_encoder(this_nobs).reshape(batch_size, -1)
                next_nobs_features = self._target_encoder(next_this_nobs).reshape(batch_size, -1)
        else:
            nobs_features = this_nobs
            next_nobs_features = next_this_nobs

        s, a, r, s_p, not_done = nobs_features, nactions[:, self.n_obs_steps - 1], batch['reward'][:, self.n_obs_steps - 1], next_nobs_features, batch['not_done'][:, self.n_obs_steps - 1]
        # s, a, r, s_p, _, not_done, _, _ = replay_buffer.sample(self._batch_size)
        # Compute value loss
        with torch.no_grad():
            self._target_Q.eval()
            if self._is_double_q:
                target_q = self.target_minQ(s, a)
            else:
                target_q = self._target_Q(s, a)
        value = self._value(s)
        value_loss = self.expectile_loss(target_q - value).mean()

        #update v
        self._v_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self._v_optimizer.step()

        # Compute critic loss
        with torch.no_grad():
            self._value.eval()
            next_v = self._value(s_p)
            
        target_q = r + not_done * self._gamma * next_v
        if self._is_double_q: 
            current_q1, current_q2 = self._Q(s, a)
            q_loss = ((current_q1 - target_q)**2 + (current_q2 - target_q)**2).mean()
        else:
            Q = self._Q(s, a)
            q_loss = F.mse_loss(Q, target_q)

        #update q and target q
        if self.is_share_encoder and not self.fix_encoder:
            if self.encoder_lr_scale != 1.0:
                self._encoder_optimizer.zero_grad()
        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()
        if self.is_share_encoder and not self.fix_encoder:
            if self.encoder_lr_scale != 1.0:
                self._encoder_optimizer.step()

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        if self.is_share_encoder and not self.fix_encoder:
            if self._total_update_step % self._target_update_freq == 0:
                for param, target_param in zip(self.obs_encoder.parameters(), self._target_encoder.parameters()):
                    target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
            

        return q_loss.detach().cpu().numpy(), value_loss.detach().cpu().numpy()
        
    def get_advantage(self, s, a)->torch.Tensor:

        if self._is_double_q:
            return self.minQ(s, a) - self._value(s)
        else:
            return self._Q(s, a) - self._value(s)
    def get_online_value_buget(self, cfg):
        if not cfg.ppo.share_encoder:
            value_model = nn.Sequential(
                self.obs_encoder,
                self._value
            )
        else:
            value_model = self._value
        return value_model
    def save(
        self, q_path: str, v_path: str, encoder_path: str
    ) -> None:
        # Handle DDP-wrapped models by saving the underlying module's state_dict
        q_to_save = self._Q.module if hasattr(self._Q, 'module') else self._Q
        v_to_save = self._value.module if hasattr(self._value, 'module') else self._value
        encoder_to_save = self.obs_encoder.module if hasattr(self.obs_encoder, 'module') else self.obs_encoder

        torch.save(q_to_save.state_dict(), q_path)
        print(f'Q function parameters saved in {q_path}')
        
        torch.save(v_to_save.state_dict(), v_path)
        print(f'Value parameters saved in {v_path}')
        
        if not self.fix_encoder and encoder_path is not None:
            torch.save(encoder_to_save.state_dict(), encoder_path)
            print(f'Encoder parameters saved in {encoder_path}')

    def load(
        self, q_path: str, v_path: str, encoder_path: str = None
    ) -> None:
        self._Q.load_state_dict(torch.load(q_path, map_location=self._device))
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')
        self._value.load_state_dict(torch.load(v_path, map_location=self._device))
        print('Value parameters loaded')
        if not self.fix_encoder and encoder_path is not None:
            self.obs_encoder.load_state_dict(torch.load(encoder_path, map_location=self._device))
            self._target_encoder.load_state_dict(torch.load(encoder_path, map_location=self._device))
            print('Value parameters loaded {}'.format(encoder_path))

    def load_with_encoder(self, q_path: str, v_path: str, encoder_path: str = None) -> None:
        q_params = torch.load(q_path, map_location=self._device)
        encoder_params = self.obs_encoder.state_dict()
        modified_encoder_params = {f"_obs_encoder.{k}": v for k, v in encoder_params.items()}

        q_params.update(modified_encoder_params)
        self._Q.load_state_dict(q_params)
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')

        value_params = torch.load(v_path, map_location=self._device)
        value_params.update(modified_encoder_params)

        self._value.load_state_dict(value_params)
        print('Value parameters loaded')

class IQL_Q_V(nn.Module):
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        feature_dim: int,   
        action_dim: int,
        q_hidden_dim: int,
        q_depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        v_hidden_dim: int,
        v_depth: int,
        v_lr: float,
        omega: float,
        is_double_q: bool,
        dp3_normalizer,
        obs_encoder: DP3Encoder = None,
        n_obs_steps: int = 2,
        is_share_encoder: bool = True,
        use_pc_color: bool = False,
        use_action_embed: bool = False,
        fix_encoder: bool = False,
        action_norm: bool = True,
    ) -> None:
        
        super().__init__()
        self._device = device
        self._omega = omega
        self._is_double_q = is_double_q
        # for dp3
        self.use_pc_color = use_pc_color
        self.normalizer = dp3_normalizer
        self.n_obs_steps = n_obs_steps
        self.action_norm = action_norm
        #for q
        if is_double_q:
            print('using double q learning')
            self._Q = DoubleQMLP(use_action_embed, None, state_dim, feature_dim, action_dim, q_hidden_dim, q_depth).to(device)
            self._target_Q = DoubleQMLP(use_action_embed, None, state_dim, feature_dim, action_dim, q_hidden_dim, q_depth).to(device)
        else:
            self._Q = QMLP(use_action_embed, None, state_dim, action_dim, feature_dim, q_hidden_dim, q_depth).to(device)
            self._target_Q = QMLP(use_action_embed, None, state_dim, feature_dim, action_dim, q_hidden_dim, q_depth).to(device)

        #for v
        self._value = ValueMLP(None, state_dim, v_hidden_dim, v_depth, n_obs_steps).to(device)
        
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._total_update_step = -1
        self._target_update_freq = target_update_freq
        self._tau = tau
        self._gamma = gamma

    def minQ(self, s: torch.Tensor, a: torch.Tensor):
        Q1, Q2 = self._Q(s, a)
        return torch.min(Q1, Q2)

    def target_minQ(self, s: torch.Tensor, a: torch.Tensor):
        Q1, Q2 = self._target_Q(s, a)
        return torch.min(Q1, Q2)
    
    def expectile_loss(self, loss: torch.Tensor)->torch.Tensor:
        weight = torch.where(loss > 0, self._omega, (1 - self._omega))
        return weight * (loss**2)
    def update(self, batch, nobs_features: torch.tensor, next_nobs_features: torch.tensor) -> float:
        if self.action_norm:
            nactions = self.normalizer['action'].normalize(batch['action'])
        else:
            nactions = batch['action']
        # next_nactions = self.normalizer['next_action'].normalize(batch['next_action'])
        batch_size = nactions.shape[0]

        nobs_features = nobs_features.reshape(batch_size, -1)
        next_nobs_features = next_nobs_features.reshape(batch_size, -1)

        s, a, r, s_p, not_done = nobs_features, nactions[:, self.n_obs_steps - 1], batch['reward'][:, self.n_obs_steps - 1], next_nobs_features, batch['not_done'][:, self.n_obs_steps - 1]
        # s, a, r, s_p, _, not_done, _, _ = replay_buffer.sample(self._batch_size)
        # Compute value loss
        with torch.no_grad():
            self._target_Q.eval()
            if self._is_double_q:
                target_q = self.target_minQ(s, a)
            else:
                target_q = self._target_Q(s, a)
        value = self._value(s)
        value_loss = self.expectile_loss(target_q - value).mean()

        #update v
        # self._v_optimizer.zero_grad()
        # value_loss.backward(retain_graph=True)
        # self._v_optimizer.step()

        # Compute critic loss
        with torch.no_grad():
            self._value.eval()
            next_v = self._value(s_p)
            
        target_q = r + not_done * self._gamma * next_v
        if self._is_double_q: 
            current_q1, current_q2 = self._Q(s, a)
            q_loss = ((current_q1 - target_q)**2 + (current_q2 - target_q)**2).mean()
        else:
            Q = self._Q(s, a)
            q_loss = F.mse_loss(Q, target_q)

        #update q and target q
        # self._q_optimizer.zero_grad()
        # q_loss.backward()
        # self._q_optimizer.step()

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)


        return q_loss, value_loss
        
    def get_advantage(self, s, a)->torch.Tensor:
        if self._is_double_q:
            return self.minQ(s, a) - self._value(s)
        else:
            return self._Q(s, a) - self._value(s)
    
    def save(
        self, q_path: str, v_path: str, encoder_path: str
    ) -> None:
        # Handle DDP model saving for Q
        q_state_dict = self._Q.state_dict()
        new_q_state_dict = {}
        for k, v in q_state_dict.items():
            if k.startswith('module.'):
                new_q_state_dict[k[7:]] = v
            else:
                new_q_state_dict[k] = v
        torch.save(new_q_state_dict, q_path)
        print('Q function parameters saved in {}'.format(q_path))
        
        # Handle DDP model saving for Value
        v_state_dict = self._value.state_dict()
        new_v_state_dict = {}
        for k, v in v_state_dict.items():
            if k.startswith('module.'):
                new_v_state_dict[k[7:]] = v
            else:
                new_v_state_dict[k] = v
        torch.save(new_v_state_dict, v_path)
        print('Value parameters saved in {}'.format(v_path))

    def load(
        self, q_path: str, v_path: str, encoder_path: str
    ) -> None:
        self._Q.load_state_dict(torch.load(q_path, map_location=self._device))
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')
        self._value.load_state_dict(torch.load(v_path, map_location=self._device))
        print('Value parameters loaded')
