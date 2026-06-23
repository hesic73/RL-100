import argparse
import os
import hydra
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from rl_100.env.maniskill import ManiSkillPointcloudEnv
from rl_100.gym_util.multistep_wrapper_dmc import MultiStepWrapper
from rl_100.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None, help="default: <run_dir>/eval_videos/<ckpt_name>")
    p.add_argument("--num-episodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=100)
    args = p.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ck["cfg"])

    if args.out is None:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))
        stem = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.out = os.path.join(run_dir, "eval_videos", stem)
    os.makedirs(args.out, exist_ok=True)

    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(hydra.utils.instantiate(cfg.task.dataset).get_normalizer())
    policy.to(device).eval()
    policy.load_state_dict(ck["state_dicts"].get("ema_model", ck["state_dicts"]["model"]), strict=False)

    successes = []
    for ep in range(args.num_episodes):
        base = ManiSkillPointcloudEnv(
            task_name=cfg.task.env_runner.task_name, control_mode=cfg.task.env_runner.control_mode,
            num_points=cfg.task.env_runner.num_points, max_episode_steps=args.max_steps,
            seed=args.seed + ep)
        env = MultiStepWrapper(base, n_obs_steps=cfg.n_obs_steps,
                               n_action_steps=cfg.n_action_steps, reward_agg_method='sum')
        obs = env.reset()
        policy.reset()
        frames, done, success, ret = [], False, False, 0.0
        while not done:
            obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device).float())
            with torch.no_grad():
                action = policy.predict_action(
                    {'point_cloud': obs_dict['point_cloud'].unsqueeze(0),
                     'agent_pos': obs_dict['agent_pos'].unsqueeze(0)}, deterministic=True)['action']
            obs, reward, done, info = env.step(action.detach().cpu().numpy().squeeze(0))
            frames.append(base.render())
            ret += reward
            s = np.asarray(info.get('success')).reshape(-1)
            if s.size:
                success = success or bool(s[-1])
        successes.append(success)
        path = os.path.join(args.out, f"ep{ep:02d}_{'success' if success else 'fail'}.mp4")
        imageio.mimwrite(path, frames, fps=10)
        print(f"ep {ep}: success={success} return={ret:.2f}")
        env.close()
    print(f"\nsuccess_rate {np.mean(successes):.2f} ({sum(successes)}/{len(successes)})")
    print(f"videos: {args.out}")


if __name__ == "__main__":
    main()
