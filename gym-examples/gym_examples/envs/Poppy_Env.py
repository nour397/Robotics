import gym
import numpy as np
from gym import spaces
from tqdm import tqdm

import numpy as np
from utils.skeleton import *
from utils.quaternion import *
from utils.blazepose import blazepose_skeletons
import os
from pypot.creatures import PoppyTorso
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#from pypot.creatures.ik import IKChain
from pypot.primitive.move import Move
from pypot.primitive.move import MovePlayer

from pypot import vrep

class PoppyEnv(gym.Env):

    def __init__(self, goals=2, terminates=True):
       
        vrep.close_all_connections()
        self.poppy = PoppyTorso(simulator='vrep')
        
        self.current_step =0
        self.num_steps = 0
        self.target_loaded = False

        self.done = False
        self.infos=[]
      
        
        self.episodes = 0  # used for resetting the sim every so often
        self.restart_every_n_episodes = 1000
        
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)  # 6 joints
        
        self.action_space = spaces.Box(low=np.array([0,-180]), high=np.array([180,0]), shape=(2,), dtype=np.float32)  # 2 actions

        super().__init__()
        
    def seed(self, seed=None):
        return [np.random.seed(seed)]
    
    def calculate_reward(self, dis):
        # Dense reward component: exponential decay based on distance
        dense_reward = np.exp(-10 * dis)

        if dis <= 0.3:
            # Sparse reward increases as distance decreases from 0.3 to 0, max at 0
            sparse_reward = (0.3 - dis) / 0.3  # This ensures the reward is 1 when dis is 0 and 0 when dis is 0.3
        else:
            sparse_reward = 0.0


        # Combine both rewards
        total_reward = dense_reward + sparse_reward
        return total_reward
    
    def step(self, action):
        
        action_l = action[0]
        action_r = action[1]
        
        if self.current_step > 125 :  
                        
            for k,m in enumerate(self.poppy.r_arm_chain.motors):
                            
                            if (m.name != 'r_elbow_y'):   
                                   m.goto_position(0.0, 1, wait= True)
                            else:
                                   m.goto_position(90.0,1, wait=True)
                                
            for k,m in enumerate(self.poppy.l_arm_chain.motors):
                if (m.name == 'l_shoulder_x') :   
                            m.goto_position(action_l, 1, wait= True)
                            
                elif (m.name == 'l_elbow_y') :
                            m.goto_position(90.0, 1, wait= True)
                else: 
                            m.goto_position(0.0, 1, wait= True)                

        else:     
            for k,m in enumerate(self.poppy.r_arm_chain.motors):
                if (m.name == 'r_shoulder_x') :   
                            m.goto_position(action_r, 1, wait= True)
                            
                elif (m.name == 'r_elbow_y') :
                            m.goto_position(90.0, 1, wait= True)
                else: 
                            m.goto_position(0.0, 1, wait= True)     
                    
            for k,m in enumerate(self.poppy.l_arm_chain.motors):
                            
                            if (m.name != 'l_elbow_y'):   
                                   m.goto_position(0.0, 1, wait= True)
                            else:
                                   m.goto_position(90.0,1, wait=True)
            
        obs = self.get_obs() 
        
        if self.current_step <=125 :        
             dis = np.linalg.norm(obs[3:] - np.array(self.targets[self.current_step].flatten())[3:])
        else:
             dis = np.linalg.norm(obs[0:3] - np.array(self.targets[self.current_step].flatten())[0:3])       
        

        reward = self.calculate_reward(dis)
            
        self.current_step += 5
            
        print("reward : ", reward)
        print("current step : ", self.current_step)
               
        info={'episode':self.episodes, 'step':self.current_step, 'reward':reward}
        self.infos.append(info)
        
        self.done = (self.current_step >=self.num_steps)
        
        if self.done:
            self.episodes += 1
        
        print("episode : ", self.episodes)
            
        
        info={}

        return np.float32(obs), reward, self.done,info
    
    def reset(self):
        joint_pos = { 'l_elbow_y':90.0,
                     'head_y': 0.0,
                     'r_arm_z': 0.0, 
                     'head_z': 0.0,
                     'r_shoulder_x': 0.0, 
                     'r_shoulder_y': 0.0,
                     'r_elbow_y': 90.0, 
                     'l_arm_z': 0.0,
                     'abs_z': 0.0,
                     'bust_y': 0.0, 
                     'bust_x':0.0,
                     'l_shoulder_x': 0.0,
                     'l_shoulder_y': 0.0
                    }
        
        for m in self.poppy.motors:
               m.goto_position(joint_pos[m.name], 1, wait= True)
        
        
        self.current_step =0
        self.done = False
        
        if self.target_loaded == False :
            self.get_target()
            self.target_loaded =True
            
        self.num_steps = self.targets.shape[0]
        obs = np.r_[self.poppy.l_arm_chain.position, self.poppy.r_arm_chain.position]
        
        return np.float32(obs)

    def get_obs(self):        
        return np.r_[self.poppy.l_arm_chain.position, self.poppy.r_arm_chain.position]
       
    def moving_average(self,a, n=3) :
        repeat_shape = list(a.shape)
        repeat_shape[1:] = [1 for _ in range(len(repeat_shape)-1)]
        repeat_shape[0] = n//2
        a = torch.cat([a[:1].repeat(*repeat_shape), a, a[-2:].repeat(*repeat_shape)])
        ret = torch.cumsum(a, axis=0)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    def interpolate_targets(self,targets, factor=1):
    
        length, joints, _ = targets.shape
        new_targets = torch.zeros((length-1) * factor + 1, joints, 3)
        for i in range(new_targets.shape[0]):

            target_id = float(i/factor)
            before_id = int(np.floor(target_id))
            after_id = int(np.floor(target_id + 1))

            before_coef = 1 - (target_id - before_id)
            after_coef = 1 - (after_id - target_id)

            if after_id > length - 1:
                after_id = length - 1

            new_targets[i] = before_coef * targets[before_id] + after_coef * targets[after_id]

        return new_targets
    
    
    def get_target(self):
        
        self.skeletons = blazepose_skeletons('mai1.mov')        
        self.topology = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]        
        self.poppy_lengths = torch.Tensor([
                            0.0,
                            0.07,
                            0.18,
                            0.19,
                            0.07,
                            0.18,
                            0.19,
                            0.12,
                            0.08,
                            0.07,
                            0.05,
                            0.1, 
                            0.15,
                            0.13,
                            0.1,
                            0.15,
                            0.13
                            ])
        
        targets, _ = self.targets_from_skeleton(self.skeletons, self.topology,self.poppy_lengths)
        
        interpolated_targets = self.interpolate_targets(targets)
        
        smoothed_targets = self.moving_average(interpolated_targets, n=15)
        
        self.targets = smoothed_targets
        
        
        
    def targets_from_skeleton(self, source_positions, topology, poppy_lengths):
        # Works in batched
        batch_size, n_joints, _ = source_positions.shape

        # Measure skeleton bone lengths
        source_lengths = torch.Tensor(batch_size, n_joints)
        for child, parent in enumerate(topology):
            source_lengths[:, child] = torch.sqrt(
                torch.sum(
                    (source_positions[:, child] - source_positions[:, parent])**2,
                    axis=-1
                )
            )

        # Find the corresponding angles
        source_offsets = torch.zeros(batch_size, n_joints, 3)
        source_offsets[:, :, -1] = source_lengths
        quaternions = find_quaternions(topology, source_offsets, source_positions)

        # Re-orient according to the pelvis->chest orientation
        base_orientation = quaternions[:, 7:8].repeat(1, n_joints, 1).reshape(batch_size*n_joints, 4)
        base_orientation += 1e-3 * torch.randn_like(base_orientation)
        quaternions = quaternions.reshape(batch_size*n_joints, 4)
        quaternions = batch_quat_left_multiply(
            batch_quat_inverse(base_orientation),
            quaternions
        )
        quaternions = quaternions.reshape(batch_size, n_joints, 4)

        # Use these quaternions in the forward kinematics with the Poppy skeleton
        target_offsets = torch.zeros(batch_size, n_joints, 3)
        target_offsets[:, :, -1] = poppy_lengths.unsqueeze(0).repeat(batch_size, 1)
        target_positions = forward_kinematics(
            topology,
            torch.zeros(batch_size, 3),
            target_offsets,
            quaternions
        )[0]

        # Measure the hip orientation
        alpha = np.arctan2(
            target_positions[0, 1, 1] - target_positions[0, 0, 1],
            target_positions[0, 1, 0] - target_positions[0, 0, 0]
        )

        # Rotate by alpha around z
        alpha = alpha
        rotation = torch.Tensor([np.cos(alpha/2), 0, 0, np.sin(alpha/2)]).unsqueeze(0).repeat(batch_size*n_joints, 1)
        quaternions = quaternions.reshape(batch_size*n_joints, 4)
        quaternions = batch_quat_left_multiply(
            batch_quat_inverse(rotation),
            quaternions
        )
        quaternions = quaternions.reshape(batch_size, n_joints, 4)

        # Use these quaternions in the forward kinematics with the Poppy skeleton
        target_positions = forward_kinematics(
            topology,
            torch.zeros(batch_size, 3),
            target_offsets,
            quaternions
        )[0]

        end_effector_indices = [13, 16]

        return target_positions[:, end_effector_indices], target_positions