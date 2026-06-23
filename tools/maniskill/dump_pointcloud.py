import argparse
import os
import numpy as np
import open3d as o3d

from rl_100.env.maniskill import ManiSkillPointcloudEnv

# Optional: roll out a checkpoint and dump the point cloud along the trajectory
# instead of just initial states.
import hydra
import torch
from omegaconf import OmegaConf
from rl_100.gym_util.multistep_wrapper_dmc import MultiStepWrapper
from rl_100.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)


def save_pcd(pc, path):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc[:, :3].astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(pc[:, 3:6].astype(np.float64))
    o3d.io.write_point_cloud(path, cloud)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/pcd")
    p.add_argument("--num", type=int, default=4, help="number of point clouds to dump")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default=None, help="if set, roll out this policy and dump every --stride steps")
    p.add_argument("--stride", type=int, default=10)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = ManiSkillPointcloudEnv(
        task_name="PickCubeRL100-v1", control_mode="pd_joint_pos", num_points=512,
        seed=args.seed)

    if args.ckpt is None:
        for i in range(args.num):
            obs = env.reset()
            save_pcd(obs["point_cloud"], os.path.join(args.out, f"pc_{i:02d}.pcd"))
            print(f"saved pc_{i:02d}.pcd")
    else:
        ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
        cfg = OmegaConf.create(ck["cfg"])
        policy = hydra.utils.instantiate(cfg.policy)
        policy.set_normalizer(hydra.utils.instantiate(cfg.task.dataset).get_normalizer())
        policy.to("cuda").eval()
        policy.load_state_dict(ck["state_dicts"].get("ema_model", ck["state_dicts"]["model"]), strict=False)
        wenv = MultiStepWrapper(env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps)
        obs = wenv.reset()
        policy.reset()
        done, t, saved = False, 0, 0
        while not done:
            if t % args.stride == 0:
                save_pcd(env._to_obs(env._env.unwrapped.get_obs())["point_cloud"],
                         os.path.join(args.out, f"step_{t:03d}.pcd"))
                saved += 1
            od = dict_apply(obs, lambda x: torch.from_numpy(x).to("cuda").float())
            with torch.no_grad():
                a = policy.predict_action({'point_cloud': od['point_cloud'].unsqueeze(0),
                                           'agent_pos': od['agent_pos'].unsqueeze(0)},
                                          deterministic=True)['action']
            obs, _, done, _ = wenv.step(a.detach().cpu().numpy().squeeze(0))
            t += 1
        print(f"saved {saved} point clouds along a {t}-step rollout")

    env.close()
    print(f"output: {os.path.abspath(args.out)}")
    print(f"view with: python tools/maniskill/view_pcd.py {args.out}/<file>.pcd")


if __name__ == "__main__":
    main()
