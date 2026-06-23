import argparse
import json
import h5py
import numpy as np
import zarr
import gymnasium as gym
from tqdm import tqdm

import mani_skill.envs  # noqa: F401
from mani_skill.trajectory import utils as traj_utils
from rl_100.env.maniskill import process_point_cloud, build_agent_pos  # noqa: F401  also registers PickCubeRL100-v1

GAMMA = 0.99


def compute_return(reward, not_done, gamma=GAMMA):
    ret = np.zeros((len(reward), 1), dtype=np.float32)
    pre = 0.0
    for i in reversed(range(len(reward))):
        ret[i] = reward[i] + gamma * pre * not_done[i]
        pre = ret[i]
    return ret


def obs_to_pc_state(obs, num_points):
    pc = obs["pointcloud"]
    point_cloud = process_point_cloud(
        pc["xyzw"][0].cpu().numpy(), pc["rgb"][0].cpu().numpy(), num_points)
    extra = obs["extra"]
    state = build_agent_pos(obs["agent"]["qpos"][0].cpu().numpy(),
                            extra["tcp_pose"][0].cpu().numpy(),
                            extra["goal_pos"][0].cpu().numpy())
    return point_cloud, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo-h5", required=True, help="ORIGINAL demo h5 (actions + env_states)")
    p.add_argument("--demo-json", required=True)
    p.add_argument("--env-id", default="PickCubeRL100-v1")
    p.add_argument("--control-mode", default="pd_joint_pos")
    p.add_argument("--out", required=True)
    p.add_argument("--num-points", type=int, default=512)
    p.add_argument("--max-episodes", type=int, default=None, help="default: all demos in the h5")
    args = p.parse_args()

    f = h5py.File(args.demo_h5, "r")
    meta = json.load(open(args.demo_json))
    episodes = meta["episodes"]
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]

    env = gym.make(args.env_id, obs_mode="pointcloud", control_mode=args.control_mode,
                   num_envs=1, sim_backend="physx_cpu")

    buf = {k: [] for k in ["state", "action", "point_cloud", "next_state",
                           "next_action", "next_point_cloud", "reward",
                           "done", "timeout", "return"]}
    episode_ends = []
    total = 0

    for ep in tqdm(episodes, desc="state-replay"):
        traj = f[f"traj_{ep['episode_id']}"]
        actions = traj["actions"][:].astype(np.float32)
        T = actions.shape[0]
        success = bool(traj["success"][-1])
        states = traj_utils.dict_to_list_of_dicts(traj["env_states"])

        env.reset(seed=ep["episode_seed"], options=dict(reconfigure=False))
        pcs, sts = [], []
        for t in range(T + 1):
            env.unwrapped.set_state_dict(states[t])
            pc, st = obs_to_pc_state(env.unwrapped.get_obs(), args.num_points)
            pcs.append(pc); sts.append(st)

        reward = np.zeros((T, 1), dtype=np.float32)
        done = np.zeros((T, 1), dtype=np.float32)
        timeout = np.zeros((T, 1), dtype=np.float32)
        reward[-1, 0] = 1.0 if success else 0.0
        done[-1, 0] = 1.0
        timeout[-1, 0] = 0.0 if success else 1.0

        for t in range(T):
            buf["state"].append(sts[t])
            buf["action"].append(actions[t])
            buf["point_cloud"].append(pcs[t])
            buf["next_state"].append(sts[t + 1])
            buf["next_action"].append(actions[t + 1] if t + 1 < T else actions[t])
            buf["next_point_cloud"].append(pcs[t + 1])
        buf["reward"].append(reward)
        buf["done"].append(done)
        buf["timeout"].append(timeout)
        buf["return"].append(compute_return(reward, 1.0 - done))
        total += T
        episode_ends.append(total)
    env.close()

    data = {
        "state": np.stack(buf["state"]), "action": np.stack(buf["action"]),
        "point_cloud": np.stack(buf["point_cloud"]),
        "next_state": np.stack(buf["next_state"]), "next_action": np.stack(buf["next_action"]),
        "next_point_cloud": np.stack(buf["next_point_cloud"]),
        "reward": np.concatenate(buf["reward"]), "done": np.concatenate(buf["done"]),
        "timeout": np.concatenate(buf["timeout"]), "return": np.concatenate(buf["return"]),
    }
    episode_ends = np.array(episode_ends, dtype=np.int64)
    root = zarr.open(args.out, mode="w")
    dg = root.require_group("data"); mg = root.require_group("meta")
    for k, v in data.items():
        dg.array(k, v, chunks=(min(1024, v.shape[0]),) + v.shape[1:], dtype=v.dtype)
    mg.array("episode_ends", episode_ends, chunks=(len(episode_ends),), dtype=np.int64)
    print(f"wrote {args.out}  episodes={len(episode_ends)} transitions={total}")
    for k, v in data.items():
        print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
