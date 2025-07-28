import gymnasium as gym
from gymnasium import spaces
import numpy as np

class KellyCoinflipEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(101)  # Wealth 0 to 100
        self.action_space = spaces.Discrete(10)  # Bet 0% to 90%
        self.max_wealth = 100
        self.bias = 0.6  # P(heads)
        self.wealth = 50
        self.max_steps = 20
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.wealth = 50
        self.step_count = 0
        return int(self.wealth), {}

    def step(self, action):
        self.step_count += 1
        bet_fraction = action * 0.1
        bet = self.wealth * bet_fraction
        if np.random.rand() < self.bias:
            self.wealth += bet
        else:
            self.wealth -= bet
        self.wealth = np.clip(self.wealth, 0, self.max_wealth)
        reward = self.wealth / self.max_wealth
        terminated = self.wealth <= 0 or self.wealth >= self.max_wealth
        truncated = self.step_count >= self.max_steps
        return int(self.wealth), reward, terminated, truncated, {}

# Register environment
gym.envs.registration.register(id='KellyCoinflip-v0', entry_point='kellycoinflip:KellyCoinflipEnv')