import os

import imageio
import numpy as np
import torch
import tqdm
from termcolor import cprint

from rl_100.env.maniskill import ManiSkillPointcloudEnv
from rl_100.gym_util.multistep_wrapper_dmc import MultiStepWrapper
from rl_100.policy.base_policy import BasePolicy
from rl_100.common.pytorch_util import dict_apply
from rl_100.env_runner.base_runner import BaseRunner
import rl_100.common.logger_util as logger_util


class ManiSkillRunner(BaseRunner):
    def __init__(self, output_dir, eval_episodes=20, max_steps=100, n_obs_steps=2,
                 n_action_steps=4, fps=10, task_name="PickCube-v1",
                 control_mode="pd_joint_pos", num_points=512, tqdm_interval_sec=5.0,
                 seed=42, gamma=0.99, record_n_videos=None, **kwargs):
        super().__init__(output_dir)
        self.task_name = task_name
        self.control_mode = control_mode
        self.num_points = num_points
        self.max_steps = max_steps
        self.eval_episodes = eval_episodes
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.fps = fps
        # how many eval episodes to render to mp4 each eval (None = all)
        self.record_n_videos = eval_episodes if record_n_videos is None else record_n_videos
        self.video_dir = os.path.join(output_dir, "eval_videos")
        self._seed = seed
        self._env_counter = 0
        self._eval_count = 0
        self.current_epoch = None  # set by train.py before each run() so videos are named by epoch
        self.env = self.make_env()
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def make_env(self, record_video=True):
        self._env_counter += 1
        env = ManiSkillPointcloudEnv(
            task_name=self.task_name, control_mode=self.control_mode,
            num_points=self.num_points, max_episode_steps=self.max_steps,
            seed=self._seed + self._env_counter)
        return MultiStepWrapper(env, n_obs_steps=self.n_obs_steps,
                                n_action_steps=self.n_action_steps,
                                reward_agg_method='sum')

    def run(self, policy: BasePolicy, use_cm=False, distill2mean=False, eval_env_num=1):
        device = policy.device
        env = self.env
        label = self.current_epoch if self.current_epoch is not None else self._eval_count
        out_dir = os.path.join(self.video_dir, f"epoch_{label:04d}")
        all_returns, successes = [], []
        for ep in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval {self.task_name}",
                            leave=False, mininterval=self.tqdm_interval_sec):
            obs = env.reset()
            policy.reset()
            record = ep < self.record_n_videos
            frames = []
            done, episode_reward, success = False, 0.0, False
            while not done:
                obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device).float())
                with torch.no_grad():
                    action = policy.predict_action(
                        {'point_cloud': obs_dict['point_cloud'].unsqueeze(0),
                         'agent_pos': obs_dict['agent_pos'].unsqueeze(0)},
                        deterministic=True, use_cm=use_cm, distill2mean=distill2mean)['action']
                obs, reward, done, info = env.step(action.detach().cpu().numpy().squeeze(0))
                if record:
                    frames.append(env.render())
                episode_reward += reward
                s = np.asarray(info.get('success')).reshape(-1)
                if s.size:
                    success = success or bool(s[-1])
            all_returns.append(episode_reward)
            successes.append(float(success))
            if record:
                os.makedirs(out_dir, exist_ok=True)
                tag = "success" if success else "fail"
                imageio.mimwrite(os.path.join(out_dir, f"ep{ep:02d}_{tag}.mp4"), frames, fps=self.fps)

        self._eval_count += 1
        sr = float(np.mean(successes))
        cprint(f"mean_returns: {np.mean(all_returns):.3f} | success_rate: {sr:.3f} | videos: {out_dir}", 'green')
        self.logger_util_test.record(sr)
        self.logger_util_test10.record(sr)
        return {
            'test_mean_score': sr,
            'mean_returns': float(np.mean(all_returns)),
            'success_rate': sr,
            'SR_test_L3': self.logger_util_test.average_of_largest_K(),
            'SR_test_L5': self.logger_util_test10.average_of_largest_K(),
        }
