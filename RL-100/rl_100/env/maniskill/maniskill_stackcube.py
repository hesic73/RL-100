import torch
import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from rl_100.env.maniskill.maniskill_pickcube import _gemini_camera


# Stack red cubeA on green cubeB and release. Reuses RL-100's gemini camera and
# exposes goal_pos (= cubeB position) so agent_pos = qpos + tcp_pose + goal_pos
# (19 dims) stays identical to PickCube -> the runner/dataset/pipeline are reused
# unchanged. Cube spawn is confined to the point-cloud crop (+-0.15) and is a
# configurable difficulty knob via cube_spawn_half_size.
@register_env("StackCubeRL100-v1", max_episode_steps=100, override=True)
class StackCubeRL100Env(StackCubeEnv):
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02,
                 cube_spawn_half_size=0.06, **kwargs):
        self.cube_spawn_half_size = cube_spawn_half_size
        super().__init__(*args, robot_uids=robot_uids,
                         robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    @property
    def _default_sensor_configs(self):
        return [_gemini_camera("base_camera")]

    @property
    def _default_human_render_camera_configs(self):
        return [_gemini_camera("render_camera")]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            h = self.cube_spawn_half_size
            sampler = randomization.UniformPlacementSampler(
                bounds=[[-h, -h], [h, h]], batch_size=b, device=self.device)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            cubeA_xy = sampler.sample(radius, 100)
            cubeB_xy = sampler.sample(radius, 100, verbose=False)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xyz[:, :2] = cubeA_xy
            qA = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qA))
            xyz[:, :2] = cubeB_xy
            qB = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qB))

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)   # provides tcp_pose
        obs["goal_pos"] = self.cubeB.pose.p   # place target; reuses build_agent_pos
        return obs
