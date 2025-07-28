import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NChainEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(5)
        self.action_space = spaces.Discrete(2)  # 0: forward, 1: backward
        self.n = 5
        self.state = 0
        self.max_steps = 20
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        self.step_count += 1
        if action == 0:  # Forward
            self.state = min(self.state + 1, self.n - 1)
            reward = 2
        else:  # Backward
            self.state = max(self.state - 1, 0)
            reward = 10 if np.random.rand() < 0.1 else 0
        terminated = False
        truncated = self.step_count >= self.max_steps
        return self.state, reward, terminated, truncated, {}

# Register environment
gym.envs.registration.register(id='NChain-v0', entry_point='nchain:NChainEnv')