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
                 seed=42, gamma=0.99, record_n_videos=None, env_kwargs=None, **kwargs):
        super().__init__(output_dir)
        self.task_name = task_name
        self.control_mode = control_mode
        self.env_kwargs = dict(env_kwargs) if env_kwargs else {}
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
            seed=self._seed + self._env_counter, env_kwargs=self.env_kwargs)
        return MultiStepWrapper(env, n_obs_steps=self.n_obs_steps,
                                n_action_steps=self.n_action_steps,
                                reward_agg_method='sum')

    def run(self, policy: BasePolicy, data_collect=False, use_cm=False, distill2mean=False, traj_path=None, eval_env_num=1):
        device = policy.device
        env = self.env
        
        if data_collect:
            import h5py
            import copy
            from datetime import datetime
            
            deterministic = False
            file_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            if traj_path is None:
                traj_path = os.path.join(self.output_dir, 'rollouts')
            session_dir = os.path.join(traj_path, f"collect_{file_time}")
            os.makedirs(session_dir, exist_ok=True)
            
            all_returns, successes = [], []
            for ep in tqdm.tqdm(range(self.eval_episodes), desc=f"DataCollect {self.task_name}",
                                leave=False, mininterval=self.tqdm_interval_sec):
                point_cloud_arrays = []
                state_arrays = []
                action_arrays = []
                next_point_cloud_arrays = []
                next_state_arrays = []
                reward_arrays = []
                done_arrays = []
                timeout_arrays = []
                is_success_arrays = []
                
                obs = env.reset()
                policy.reset()
                done = False
                episode_reward = 0.0
                success = False
                
                while not done:
                    # Record current single-step observation BEFORE stepping
                    curr_obs = env.observation
                    point_cloud_arrays.append(curr_obs['point_cloud'])
                    state_arrays.append(curr_obs['agent_pos'])
                    
                    obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device).float())
                    with torch.no_grad():
                        action = policy.predict_action(
                            {'point_cloud': obs_dict['point_cloud'].unsqueeze(0),
                             'agent_pos': obs_dict['agent_pos'].unsqueeze(0)},
                            deterministic=deterministic, use_cm=use_cm, distill2mean=distill2mean)['action']
                    act_np = action.detach().cpu().numpy().squeeze(0) # shape (n_action_steps, action_dim)
                    
                    # Execute the action chunk step-by-step
                    for i in range(self.n_action_steps):
                        if done:
                            break
                        
                        if i > 0:
                            # For subsequent steps in the chunk, record the state BEFORE stepping
                            curr_obs = env.observation
                            point_cloud_arrays.append(curr_obs['point_cloud'])
                            state_arrays.append(curr_obs['agent_pos'])
                        
                        # Record the current action
                        action_arrays.append(act_np[i])
                        
                        # Step the underlying environment
                        next_obs_single, reward, done_single, info = env.env.step(act_np[i])
                        
                        # Update wrapper history manually to keep it in sync
                        env.observation = next_obs_single
                        env.obs.append(next_obs_single)
                        env.reward.append(reward)
                        
                        done = done_single or (env.max_episode_steps is not None and len(env.reward) >= env.max_episode_steps)
                        env.done.append(done)
                        
                        # Record rewards and flags
                        reward_arrays.append(reward)
                        done_arrays.append(done)
                        timeout_arrays.append(done and not info['success'])
                        is_success = bool(info['success'])
                        is_success_arrays.append(is_success)
                        success = success or is_success
                        episode_reward += reward
                        
                        # Record next observation states
                        next_point_cloud_arrays.append(next_obs_single['point_cloud'])
                        next_state_arrays.append(next_obs_single['agent_pos'])
                        
                    # Update wrapped observation for the next policy call
                    obs = env._get_obs(self.n_obs_steps)
                
                # At the end of the episode, compute the final action prediction for next_action alignment
                obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device).float())
                with torch.no_grad():
                    action = policy.predict_action(
                        {'point_cloud': obs_dict['point_cloud'].unsqueeze(0),
                         'agent_pos': obs_dict['agent_pos'].unsqueeze(0)},
                        deterministic=deterministic, use_cm=use_cm, distill2mean=distill2mean)['action']
                act_np = action.detach().cpu().numpy().squeeze(0)
                final_action = act_np[0]
                
                next_action_arrays = copy.deepcopy(action_arrays)
                next_action_arrays.append(final_action)
                next_action_arrays = next_action_arrays[1:]
                
                all_returns.append(episode_reward)
                successes.append(float(success))
                
                # Real-world fidelity: store SPARSE terminal reward (1.0 on the
                # final step iff the episode succeeded, else 0.0) instead of the
                # env's dense reward. This matches the demo dataset
                # (convert_states_to_zarr) and what a real robot can actually
                # provide, so the offline-RL reward signal stays consistent across
                # demos and rollouts.
                sparse_reward = np.zeros(len(reward_arrays), dtype=np.float32)
                if success and len(sparse_reward) > 0:
                    sparse_reward[-1] = 1.0

                # Save the episode data to HDF5
                hdf5_filename = os.path.join(session_dir, f"episode_{ep:04d}.h5")
                with h5py.File(hdf5_filename, 'w') as hdf5_file:
                    hdf5_file.create_dataset('point_cloud', data=np.array(point_cloud_arrays))
                    hdf5_file.create_dataset('state', data=np.array(state_arrays))
                    hdf5_file.create_dataset('action', data=np.array(action_arrays))
                    hdf5_file.create_dataset('next_point_cloud', data=np.array(next_point_cloud_arrays))
                    hdf5_file.create_dataset('next_state', data=np.array(next_state_arrays))
                    hdf5_file.create_dataset('next_action', data=np.array(next_action_arrays))
                    hdf5_file.create_dataset('reward', data=sparse_reward)
                    hdf5_file.create_dataset('done', data=np.array(done_arrays))
                    hdf5_file.create_dataset('timeout', data=np.array(timeout_arrays))
                    hdf5_file.create_dataset('is_success', data=np.array(is_success_arrays))
                print(f"data saved in {hdf5_filename}")
                
            self._eval_count += 1
            sr = float(np.mean(successes))
            cprint(f"mean_returns: {np.mean(all_returns):.3f} | success_rate: {sr:.3f} | rollouts: {session_dir}", 'green')
            self.logger_util_test.record(sr)
            self.logger_util_test10.record(sr)
            return {
                'test_mean_score': sr,
                'mean_returns': float(np.mean(all_returns)),
                'success_rate': sr,
                'SR_test_L3': self.logger_util_test.average_of_largest_K(),
                'SR_test_L5': self.logger_util_test10.average_of_largest_K(),
            }

        label = self.current_epoch if self.current_epoch is not None else self._eval_count
        out_dir = os.path.join(self.video_dir, f"epoch_{label:04d}")
        # Fair comparison: restart the per-episode seed sequence so EVERY eval
        # call (every checkpoint, every BPPO step, every run with the same
        # env_runner.seed) is scored on the identical set of initial states.
        # env.reset() does self._episode+=1 then seeds with self._seed+self._episode.
        env.env._episode = 0
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

    def idql_run(self, policy: BasePolicy, dynamics, first_action, get_np, use_gae,
                 iql, Q, repeat_num, use_cm=False, distill2mean=False, eval_env_num=1):
        # IDQL-style eval: sample repeat_num candidate actions per state and let
        # the critic (iql/Q, optionally rolled through dynamics) pick the best,
        # instead of the policy's own deterministic action. Same eval loop /
        # fixed-initial-state protocol / metrics as run().
        device = policy.device
        env = self.env

        label = self.current_epoch if self.current_epoch is not None else self._eval_count
        out_dir = os.path.join(self.video_dir, f"epoch_{label:04d}_idql")
        env.env._episode = 0  # fair comparison: identical initial states every eval
        all_returns, successes = [], []
        for ep in tqdm.tqdm(range(self.eval_episodes), desc=f"IDQL Eval {self.task_name}",
                            leave=False, mininterval=self.tqdm_interval_sec):
            obs = env.reset()
            policy.reset()
            record = ep < self.record_n_videos
            frames = []
            done, episode_reward, success = False, 0.0, False
            while not done:
                obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device).float())
                with torch.no_grad():
                    action_dict = policy.sample_action(
                        {'point_cloud': obs_dict['point_cloud'].unsqueeze(0),
                         'agent_pos': obs_dict['agent_pos'].unsqueeze(0)},
                        dynamics=dynamics, first_action=first_action, get_np=get_np,
                        use_gae=use_gae, iql=iql, Q=Q, repeat_num=repeat_num,
                        use_cm=use_cm, distill2mean=distill2mean)
                action = action_dict['action']
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()
                obs, reward, done, info = env.step(np.asarray(action).squeeze(0))
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
        cprint(f"[IDQL] mean_returns: {np.mean(all_returns):.3f} | success_rate: {sr:.3f} | videos: {out_dir}", 'green')
        self.logger_util_test.record(sr)
        self.logger_util_test10.record(sr)
        return {
            'test_mean_score': sr,
            'mean_returns': float(np.mean(all_returns)),
            'success_rate': sr,
            'SR_test_L3': self.logger_util_test.average_of_largest_K(),
            'SR_test_L5': self.logger_util_test10.average_of_largest_K(),
        }

