import os
import hydra
import numpy as np
import torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rl_100.env.maniskill import ManiSkillPointcloudEnv
from rl_100.gym_util.multistep_wrapper_dmc import MultiStepWrapper
from rl_100.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)
CFG = os.path.abspath("RL-100/rl_100/config")
OV = ["task=maniskill_pickcube", "policy.model=skipnet", "policy.encoder_type=dp3vib",
      "policy.act=relu", "horizon=5", "n_obs_steps=2", "n_action_steps=4", "ft_all_actions=False",
      "use_action_embed=True", "policy.encoder_output_dim=64", "policy.use_vib=True",
      "policy.use_recon=True", "policy.use_agent_pos=True", "policy.mlp_policy_depth=3",
      "policy.scheduler_type=ddim", "policy.ddim_noise_scheduler.num_train_timesteps=100",
      "policy.cm_noise_scheduler.num_train_timesteps=50", "policy.diffusion_step_embed_dim=256",
      "policy.down_dims=[256,512,1024]", "num_inference_steps=10", "policy.beta_kl=1e-5"]

STEPS = 4000
device = "cuda"


def main():
    with initialize_config_dir(config_dir=CFG, version_base=None):
        cfg = compose(config_name="rl100_3d_epsilon", overrides=OV)

    ds = hydra.utils.instantiate(cfg.task.dataset)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(ds.get_normalizer())
    policy.to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

    # ---- train BC ----
    policy.train()
    it = iter(loader)
    losses = []
    for step in range(STEPS):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = dict_apply(batch, lambda x: x.to(device))
        opt.zero_grad()
        loss, _ = policy.compute_loss(batch)
        loss.backward(); opt.step()
        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            print(f"step {step+1}: loss={np.mean(losses[-500:]):.4f}")

    policy.eval()

    # ---- in-domain action reproduction ----
    batch = dict_apply(next(iter(loader)), lambda x: x.to(device))
    with torch.no_grad():
        pred = policy.predict_action(
            {'point_cloud': batch['obs']['point_cloud'][:, :2].float(),
             'agent_pos': batch['obs']['agent_pos'][:, :2].float()}, deterministic=True)['action']
    gt = batch['action'][:, :pred.shape[1]].float()
    mae = (pred - gt).abs().mean().item()
    print(f"\n[action repro] pred vs demo MAE={mae:.4f} | demo action std={gt.std().item():.4f}")
    print(f"  per-dim MAE: {(pred-gt).abs().mean((0,1)).cpu().numpy().round(3)}")

    # ---- closed-loop eval ----
    succ = gr = 0
    N = 10
    for ep in range(N):
        base = ManiSkillPointcloudEnv(task_name="PickCubeRL100-v1", control_mode="pd_joint_pos",
                                      num_points=512, max_episode_steps=100, seed=1000 + ep)
        env = MultiStepWrapper(base, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps)
        obs = env.reset(); policy.reset()
        done, eg, es = False, False, False
        while not done:
            od = dict_apply(obs, lambda x: torch.from_numpy(x).to(device).float())
            with torch.no_grad():
                a = policy.predict_action({'point_cloud': od['point_cloud'].unsqueeze(0),
                                           'agent_pos': od['agent_pos'].unsqueeze(0)}, deterministic=True)['action']
            obs, _, done, _ = env.step(a.detach().cpu().numpy().squeeze(0))
            ev = base._env.unwrapped.evaluate()
            if bool(ev.get('is_grasped', [False])[0]): eg = True
            if bool(ev['success'][0]): es = True
        succ += es; gr += eg; env.close()
    print(f"\n[closed-loop eval] grasp {gr}/{N} | success {succ}/{N}")


if __name__ == "__main__":
    main()
