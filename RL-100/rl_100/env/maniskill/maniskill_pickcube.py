from typing import Any

import gym
import gymnasium
import numpy as np
import sapien
import torch
import fpsample

import mani_skill.envs  # noqa: F401  registers base ManiSkill tasks
import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


# Our own PickCube task: grasp a red cube, move it to a green goal. Cube spawn,
# goal xy and goal height are independent __init__ knobs. Camera is a simulated
# Orbbec Gemini 435Le.
GEMINI_W, GEMINI_H = 640, 400
GEMINI_FOV = np.deg2rad(65)  # vertical; 640/400 aspect -> ~90 deg horizontal
GEMINI_EYE = [0.35, 0.0, 0.55]
GEMINI_TARGET = [0.0, 0.0, 0.08]


def _gemini_camera(uid):
    pose = sapien_utils.look_at(eye=GEMINI_EYE, target=GEMINI_TARGET)
    return CameraConfig(uid, pose, GEMINI_W, GEMINI_H, GEMINI_FOV, 0.01, 100)


@register_env("PickCubeRL100-v1", max_episode_steps=100, override=True)
class PickCubeRL100Env(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02,
                 cube_spawn_half_size=0.06, cube_spawn_center=(0.0, 0.0),
                 goal_xy_half_size=0.06, goal_z_min=0.0, goal_z_max=0.1, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.cube_spawn_half_size = cube_spawn_half_size
        self.cube_spawn_center = cube_spawn_center
        self.goal_xy_half_size = goal_xy_half_size
        self.goal_z_min = goal_z_min
        self.goal_z_max = goal_z_max
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return [_gemini_camera("base_camera")]

    @property
    def _default_human_render_camera_configs(self):
        return [_gemini_camera("render_camera")]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]))
        self.goal_site = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[0, 1, 0, 1], name="goal_site",
            body_type="kinematic", add_collision=False, initial_pose=sapien.Pose())
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            cx, cy = self.cube_spawn_center

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 2 * self.cube_spawn_half_size - self.cube_spawn_half_size
            xyz[:, 0] += cx
            xyz[:, 1] += cy
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal = torch.zeros((b, 3))
            goal[:, :2] = torch.rand((b, 2)) * 2 * self.goal_xy_half_size - self.goal_xy_half_size
            goal[:, 0] += cx
            goal[:, 1] += cy
            goal[:, 2] = torch.rand((b)) * (self.goal_z_max - self.goal_z_min) + self.goal_z_min + self.cube_half_size
            self.goal_site.set_pose(Pose.create_from_pq(goal))

    def _get_obs_extra(self, info: dict):
        obs = dict(is_grasped=info["is_grasped"], tcp_pose=self.agent.tcp_pose.raw_pose,
                   goal_pos=self.goal_site.pose.p)
        if "state" in self.obs_mode:
            obs.update(obj_pose=self.cube.pose.raw_pose,
                       tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                       obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p)
        return obs

    def evaluate(self):
        is_obj_placed = torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1) <= self.goal_thresh
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {"success": is_obj_placed & is_robot_static, "is_obj_placed": is_obj_placed,
                "is_robot_static": is_robot_static, "is_grasped": is_grasped}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj = torch.linalg.norm(self.cube.pose.p - self.agent.tcp_pose.p, axis=1)
        reward = 1 - torch.tanh(5 * tcp_to_obj)
        is_grasped = info["is_grasped"]
        reward += is_grasped
        obj_to_goal = torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
        reward += (1 - torch.tanh(5 * obj_to_goal)) * is_grasped
        qvel = self.agent.robot.get_qvel()[..., :-2]
        reward += (1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))) * info["is_obj_placed"]
        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 5


# Workspace crop + farthest-point sampling, mirroring RL-100's real-robot
# pipeline. Robot is kept in the cloud (no segmentation, as on a real camera).
CROP_MIN = np.array([-0.15, -0.15, 0.005], dtype=np.float32)
CROP_MAX = np.array([0.15, 0.15, 0.35], dtype=np.float32)


def process_point_cloud(xyzw, rgb, num_points):
    xyz = xyzw[:, :3]
    mask = np.all((xyz > CROP_MIN) & (xyz < CROP_MAX), axis=1)
    xyz, rgb = xyz[mask], rgb[mask]
    if xyz.shape[0] == 0:
        return np.zeros((num_points, 6), dtype=np.float32)
    if xyz.shape[0] >= num_points:
        idx = fpsample.bucket_fps_kdtree_sampling(xyz.astype(np.float64), num_points)
    else:
        idx = np.resize(np.arange(xyz.shape[0]), num_points)
    pc = np.concatenate([xyz, rgb.astype(np.float32) / 255.0], axis=-1)
    return pc[idx].astype(np.float32)


def build_agent_pos(qpos, tcp_pose, goal_pos):
    # Must be built identically at zarr conversion and at eval.
    return np.concatenate([qpos, tcp_pose, goal_pos]).astype(np.float32)


# Adapter: batched-torch-gymnasium ManiSkill env -> single-env numpy gym-0.21
# obs_dict interface that RL-100's runner/policy expect.
class ManiSkillPointcloudEnv(gym.Env):
    def __init__(self, task_name="PickCubeRL100-v1", control_mode="pd_joint_pos",
                 num_points=512, max_episode_steps=100, seed=0, render_mode="rgb_array"):
        self._env = gymnasium.make(
            task_name, obs_mode="pointcloud", control_mode=control_mode,
            num_envs=1, max_episode_steps=max_episode_steps, sim_backend="physx_cpu",
            render_mode=render_mode)
        self.num_points = num_points
        self._seed = seed
        self._episode = 0

        action_dim = int(self._env.action_space.shape[-1])
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (action_dim,), dtype=np.float32)
        agent_pos_dim = self._reset_raw()["agent_pos"].shape[0]
        self.observation_space = gym.spaces.Dict({
            "point_cloud": gym.spaces.Box(-np.inf, np.inf, (num_points, 6), dtype=np.float32),
            "agent_pos": gym.spaces.Box(-np.inf, np.inf, (agent_pos_dim,), dtype=np.float32),
        })

    def _to_obs(self, obs):
        pc = obs["pointcloud"]
        point_cloud = process_point_cloud(
            pc["xyzw"][0].cpu().numpy(), pc["rgb"][0].cpu().numpy(), self.num_points)
        extra = obs["extra"]
        agent_pos = build_agent_pos(
            obs["agent"]["qpos"][0].cpu().numpy(),
            extra["tcp_pose"][0].cpu().numpy(),
            extra["goal_pos"][0].cpu().numpy())
        return {"point_cloud": point_cloud, "agent_pos": agent_pos}

    def _reset_raw(self):
        obs, _ = self._env.reset(seed=self._seed)
        return self._to_obs(obs)

    def reset(self):
        # Distinct seed per episode so eval rollouts see diverse initial states.
        self._episode += 1
        obs, _ = self._env.reset(seed=self._seed + self._episode)
        return self._to_obs(obs)

    def step(self, action):
        act = torch.as_tensor(np.asarray(action), dtype=torch.float32).unsqueeze(0)
        obs, reward, terminated, truncated, info = self._env.step(act)
        done = bool(terminated[0].item() or truncated[0].item())
        if "success" not in info:
            raise KeyError("ManiSkill step info has no 'success' key; evaluate() output is not being merged into info.")
        success = bool(info["success"][0].item())
        return self._to_obs(obs), float(reward[0].item()), done, {"success": success}

    def render(self):
        frame = self._env.render()
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:  # (num_envs, H, W, 3) -> (H, W, 3)
            frame = frame[0]
        return frame.astype(np.uint8)

    def close(self):
        self._env.close()
