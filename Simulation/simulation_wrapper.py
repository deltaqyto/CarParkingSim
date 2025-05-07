import gymnasium as gym

import numpy as np
import torch
import random


class SimulationWrapper(gym.Env):
    def __init__(self, environment, rank=0, seed=None):
        super(SimulationWrapper, self).__init__()
        self.env = environment()
        seed = random.randint(1, 100000) if seed is None else seed
        self.seed(seed + rank)

        # 19 inputs, 2 outputs
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, **kwargs):
        seed = random.randint(1, 100000) if seed is None else seed
        self.seed(seed)
        self.env.reset_environment()

        # Take a no-op action to get initial observation
        _, observation, _, _ = self.env.step([0, 0])
        return np.array(observation, dtype=np.float32), {}

    def step(self, action):
        # Done: Boolean 'is the env done'
        # Observation: 19 float np.array (car direction (2 components), speed, wheel angle, 12x raycast, goal x, goal y, goal angle)
        # Reward: Reward signal computed from modules
        # State: Unified environment state. Contains all information about the simulation. Print it to see the structure
        done, observation, reward, state = self.env.step(action)
        return np.array(observation), reward, done, False, state

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        return [seed]

    def get_digest(self):
        return self.env.get_digest()
    
    def close(self):
        pass
