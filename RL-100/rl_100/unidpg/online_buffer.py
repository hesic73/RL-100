import torch
import numpy as np
from rl_100.common.pytorch_util import dict_apply

class ReplayBuffer:
    def __init__(self, args, shape_info,  device, wo_visual=False):
        self.use_imagin_robot = False
        for key in shape_info['obs']:
            if 'imagin_robot' in key:
                self.use_imagin_robot = True
                break
        self.wo_visual = wo_visual
        self.use_image = (not wo_visual) and ('image' in shape_info['obs'])
        if self.use_image and shape_info['obs']['image'][-1] == 3:
            shape_info['obs']['image'] =  (
                shape_info['obs']['image'][0], 
                shape_info['obs']['image'][-1],
                shape_info['obs']['image'][1], 
                shape_info['obs']['image'][2], 
            )
        if not wo_visual:   
            self.point_cloud = np.zeros((args.batch_size, *shape_info['obs']['point_cloud']))
            if self.use_image:
                self.image = np.zeros((args.batch_size, *shape_info['obs']['image']))
            if self.use_imagin_robot:
                self.imagin_robot = np.zeros((args.batch_size, *shape_info['obs']['imagin_robot']))
        self.agent_pos = np.zeros((args.batch_size, *shape_info['obs']['agent_pos']))
        self.action = np.zeros((args.batch_size, args.num_inference_steps + 1, *shape_info['action']))
        self.a_logprob = np.zeros((args.batch_size, args.num_inference_steps, *shape_info['action']))

        if not wo_visual:
            self.next_point_cloud = np.zeros((args.batch_size, *shape_info['obs']['point_cloud']))
            if self.use_image:
                self.next_image = np.zeros((args.batch_size, *shape_info['obs']['image']))
            if self.use_imagin_robot:
                self.next_imagin_robot = np.zeros((args.batch_size, *shape_info['obs']['imagin_robot']))

        self.next_agent_pos = np.zeros((args.batch_size, *shape_info['obs']['agent_pos']))
        self.reward = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.dw = np.zeros((args.batch_size, 1))
        self.count = 0
        self.device = device

    def store(self, obs, action, a_logprob, reward, next_obs, done, dw):
        if not self.wo_visual:
            self.point_cloud[self.count] = obs['point_cloud']
            if self.use_image:
                self.image[self.count] = obs['image']
            if self.use_imagin_robot:
                self.imagin_robot[self.count] = obs['imagin_robot']
        self.agent_pos[self.count] = obs['agent_pos']
        self.action[self.count] = action
        self.a_logprob[self.count] = a_logprob
        self.reward[self.count] = reward
        if not self.wo_visual:
            self.next_point_cloud[self.count] = next_obs['point_cloud']
            if self.use_image:
                self.next_image[self.count] = next_obs['image']
            if self.use_imagin_robot:
                self.next_imagin_robot[self.count] = next_obs['imagin_robot']
        self.next_agent_pos[self.count] = next_obs['agent_pos']
        self.done[self.count] = done
        self.dw[self.count] = dw
        self.count += 1

    def sample(
        self, batch_size: int
    ) -> tuple:

        ind = np.random.randint(0, int(self.count), size=batch_size)
        action = torch.FloatTensor(self.action[ind]).to(self.device)
        obs = {'agent_pos': torch.FloatTensor(self.agent_pos[ind]).to(self.device)}
        if not self.wo_visual:
            obs['point_cloud'] = torch.FloatTensor(self.point_cloud[ind]).to(self.device)
            if self.use_image:
                obs['image'] = torch.FloatTensor(self.image[ind]).to(self.device)
            if self.use_imagin_robot:
                obs['imagin_robot'] = torch.FloatTensor(self.imagin_robot[ind]).to(self.device)
        return {'obs':obs, 'action': action}

    def numpy_to_dict(self):
        out = {
            'state': self.agent_pos,
            'action': self.action,
            'a_logprob': self.a_logprob,
            'reward': self.reward,
            'next_state': self.next_agent_pos,
            'done': self.done,
            'dw': self.dw,
        }
        if not self.wo_visual:
            out['point_cloud'] = self.point_cloud
            out['next_point_cloud'] = self.next_point_cloud
            if self.use_image:
                out['img'] = self.image
                out['next_img'] = self.next_image
            if self.use_imagin_robot:
                out['imagin_robot'] = self.imagin_robot
                out['next_imagin_robot'] = self.next_imagin_robot
        return out

    
    def numpy_to_tensor(self):
        action = torch.tensor(self.action, dtype=torch.float).to(self.device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        reward = torch.tensor(self.reward, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        obs = {'agent_pos': torch.tensor(self.agent_pos, dtype=torch.float).to(self.device)}
        next_obs = {'agent_pos': torch.tensor(self.next_agent_pos, dtype=torch.float).to(self.device)}
        if not self.wo_visual:
            obs['point_cloud'] = torch.tensor(self.point_cloud, dtype=torch.float).to(self.device)
            next_obs['point_cloud'] = torch.tensor(self.next_point_cloud, dtype=torch.float).to(self.device)
            if self.use_image:
                obs['image'] = torch.tensor(self.image, dtype=torch.float).to(self.device)
                next_obs['image'] = torch.tensor(self.next_image, dtype=torch.float).to(self.device)
            if self.use_imagin_robot:
                obs['imagin_robot'] = torch.tensor(self.imagin_robot, dtype=torch.float).to(self.device)
                next_obs['imagin_robot'] = torch.tensor(self.next_imagin_robot, dtype=torch.float).to(self.device)
        return obs, action, a_logprob, reward, next_obs, dw, done
class IqlBuffer:
    def __init__(self, offline_data, args, shape_info,  device, wo_visual=False):
        self.use_imagin_robot = False
        for key in shape_info['obs']:
            if 'imagin_robot' in key:
                self.use_imagin_robot = True
                break
        self.wo_visual = wo_visual
        self.use_image = (not wo_visual) and ('image' in shape_info['obs'])
        self.offline_data = offline_data
        if not wo_visual:   
            self.point_cloud = np.zeros((args.capacity, *shape_info['obs']['point_cloud']))
            if self.use_image:
                self.image =  np.zeros((args.capacity, *shape_info['obs']['image']))
            if self.use_imagin_robot:
                self.imagin_robot = np.zeros((args.capacity, *shape_info['obs']['imagin_robot']))

        self.agent_pos = np.zeros((args.capacity, *shape_info['obs']['agent_pos']))
        self.action = np.zeros((args.capacity,  *shape_info['action']))
        if not self.wo_visual:
            self.next_point_cloud = np.zeros((args.capacity, *shape_info['obs']['point_cloud']))
            if self.use_image:
                self.next_image = np.zeros((args.capacity, *shape_info['obs']['image']))
            if self.use_imagin_robot:
                self.next_imagin_robot = np.zeros((args.capacity, *shape_info['obs']['imagin_robot']))
        self.next_agent_pos = np.zeros((args.capacity, *shape_info['obs']['agent_pos']))
        self.reward = np.zeros((args.capacity, 1))
        self.not_done = np.zeros((args.capacity, 1))
        self.count = 0
        self.capacity = args.capacity
        self.device = device
        self.full = False
    def store(self, obs, action, reward, next_obs, done):
        if not self.wo_visual:
            self.point_cloud[self.count] = obs['point_cloud']
            if self.use_image:
                self.image[self.count] = obs['image']
            if self.use_imagin_robot:
                self.imagin_robot[self.count] = obs['imagin_robot']
        self.agent_pos[self.count] = obs['agent_pos']
        self.action[self.count] = action
        self.reward[self.count] = reward
        if not self.wo_visual:
            self.next_point_cloud[self.count] = next_obs['point_cloud']
            if self.use_image:
                self.next_image[self.count] = next_obs['image']
            if self.use_imagin_robot:
                self.next_imagin_robot[self.count] = next_obs['imagin_robot']
        self.next_agent_pos[self.count] = next_obs['agent_pos']
        self.not_done[self.count] = 1 - done
        self.count = (self.count + 1) % self.capacity
        self.full = self.full or self.count == 0

    def initial_with_dataset(self, dataset):
        dataset = dict_apply(dataset, lambda x: x.cpu().numpy())
        data_size = dataset['action'].shape[0]
        if not self.wo_visual:
            self.point_cloud[:data_size] = dataset['obs']['point_cloud']
            if self.use_image:
                self.image[:data_size] = dataset['obs']['image']
            if self.use_imagin_robot:
                self.imagin_robot[:data_size] = dataset['obs']['imagin_robot']
        self.agent_pos[:data_size] = dataset['obs']['agent_pos']
        self.action[:data_size] = dataset['action']
        self.reward[:data_size] = dataset['reward'].squeeze(1)
        if not self.wo_visual:
            self.next_point_cloud[:data_size] = dataset['next_obs']['point_cloud']
            if self.use_image:
                self.next_image[:data_size] = dataset['next_obs']['image']
            if self.use_imagin_robot:
                self.next_imagin_robot[:data_size] = dataset['next_obs']['imagin_robot']
        self.next_agent_pos[:data_size] = dataset['next_obs']['agent_pos']
        self.not_done[:data_size] = dataset['not_done'].squeeze(1)
        self.count = data_size

    def merge(self, online_batch, offline_batch):
        if self.use_image and offline_batch['obs']['image'].shape[-3] != 3:
            offline_batch['obs']['image'] = offline_batch['obs']['image'].permute(0, 1, 4, 3, 2)
            offline_batch['next_obs']['image'] = offline_batch['next_obs']['image'].permute(0, 1, 4, 3, 2)
        action = torch.cat([online_batch['action'], offline_batch['action']], dim=0)
        reward = torch.cat([online_batch['reward'], offline_batch['reward'].squeeze(1)], dim=0)
        not_done = torch.cat([online_batch['not_done'], offline_batch['not_done'].squeeze(1)], dim=0)
        obs = {'agent_pos': torch.cat([online_batch['obs']['agent_pos'], offline_batch['obs']['agent_pos']], dim=0)}
        next_obs = {'agent_pos': torch.cat([online_batch['next_obs']['agent_pos'], offline_batch['next_obs']['agent_pos']], dim=0)}
        if not self.wo_visual:
            obs['point_cloud'] = torch.cat([online_batch['obs']['point_cloud'], offline_batch['obs']['point_cloud']], dim=0)
            next_obs['point_cloud'] = torch.cat([online_batch['next_obs']['point_cloud'], offline_batch['next_obs']['point_cloud']], dim=0)
            if self.use_image:
                obs['image'] = torch.cat([online_batch['obs']['image'], offline_batch['obs']['image']], dim=0)
                next_obs['image'] = torch.cat([online_batch['next_obs']['image'], offline_batch['next_obs']['image']], dim=0)
            if self.use_imagin_robot:
                obs['imagin_robot'] = torch.cat([online_batch['obs']['imagin_robot'], offline_batch['obs']['imagin_robot']], dim=0)
                next_obs['imagin_robot'] = torch.cat([online_batch['next_obs']['imagin_robot'], offline_batch['next_obs']['imagin_robot']], dim=0)
        return {'obs':obs, 'action': action, 'reward': reward, 'next_obs': next_obs, 'not_done': not_done}



    def sample(
        self, batch_size: int
    ) -> tuple:

        ind = np.random.randint(0, int(self.count), size=batch_size)
        action = torch.FloatTensor(self.action[ind]).to(self.device)
        reward = torch.FloatTensor(self.reward[ind]).to(self.device)
        not_done = torch.FloatTensor(self.not_done[ind]).to(self.device)
        obs = {'agent_pos': torch.FloatTensor(self.agent_pos[ind]).to(self.device)}
        next_obs = {'agent_pos': torch.FloatTensor(self.next_agent_pos[ind]).to(self.device)}
        if not self.wo_visual:
            obs['point_cloud'] = torch.FloatTensor(self.point_cloud[ind]).to(self.device)
            next_obs['point_cloud'] = torch.FloatTensor(self.next_point_cloud[ind]).to(self.device)
            if self.use_image:
                obs['image'] = torch.FloatTensor(self.image[ind]).to(self.device)
                next_obs['image'] = torch.FloatTensor(self.next_image[ind]).to(self.device)
            if self.use_imagin_robot:
                obs['imagin_robot'] = torch.FloatTensor(self.imagin_robot[ind]).to(self.device)
                next_obs['imagin_robot'] = torch.FloatTensor(self.next_imagin_robot[ind]).to(self.device)
        return {'obs':obs, 'action': action, 'reward': reward, 'next_obs': next_obs, 'not_done': not_done}
