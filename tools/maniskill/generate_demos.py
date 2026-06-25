import argparse
import json

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import rl_100.env.maniskill  # noqa: F401  registers PickCubeRL100-v1
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.examples.motionplanning.panda.solutions import solvePickCube

SOLVERS = {"PickCubeRL100-v1": solvePickCube}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="PickCubeRL100-v1")
    p.add_argument("--num-demos", type=int, default=200)
    p.add_argument("--out-dir", default="data/maniskill_demos")
    p.add_argument("--vis", action="store_true", help="open a GUI to watch the solver live")
    p.add_argument("--env-kwargs", default="{}",
                   help="JSON of task-difficulty kwargs, e.g. '{\"goal_z_max\":0.25}'. "
                        "MUST match the eval/training env_runner.env_kwargs and the zarr conversion.")
    args = p.parse_args()

    env_kwargs = json.loads(args.env_kwargs)
    solve = SOLVERS[args.env_id]
    env = gym.make(args.env_id, obs_mode="none", control_mode="pd_joint_pos",
                   num_envs=1, sim_backend="physx_cpu",
                   render_mode="human" if args.vis else "rgb_array", **env_kwargs)
    env = RecordEpisode(env, output_dir=f"{args.out_dir}/{args.env_id}/motionplanning",
                        trajectory_name="trajectory", save_video=False,
                        source_type="motionplanning", record_reward=False, save_on_reset=False)

    seed, saved = 0, 0
    while saved < args.num_demos:
        res = solve(env, seed=seed, debug=False, vis=args.vis)
        if res != -1 and bool(res[-1]["success"].item()):
            env.flush_trajectory()
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.num_demos}")
        else:
            env.flush_trajectory(save=False)
        seed += 1
    env.close()
    print(f"generated {saved} demos from {seed} seeds -> {args.out_dir}/{args.env_id}/motionplanning")


if __name__ == "__main__":
    main()
