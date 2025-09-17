import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ComplexMazeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(25)  # 5x5 grid
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.grid_size = 5
        self.state = 0  # (0,0)
        self.goal = 24  # (4,4)
        self.max_steps = 20
        self.step_count = 0
        # Walls: block (1,1)-(1,2), (2,2)-(3,2), (3,1)-(4,1), (2,3)-(2,4), (3,3)-(4,3)
        self.walls = {(6, 7), (7, 6), (12, 17), (17, 12), (16, 21), (21, 16), (13, 14), (14, 13), (18, 23), (23, 18)}
        # Bonus states: (2,2), (3,3)
        self.bonus_states = {12, 18}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        self.step_count += 1
        x, y = divmod(self.state, self.grid_size)
        if action == 0:  # Up
            next_x, next_y = x - 1, y
        elif action == 1:  # Down
            next_x, next_y = x + 1, y
        elif action == 2:  # Left
            next_x, next_y = x, y - 1
        else:  # Right
            next_x, next_y = x, y + 1

        if (0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size):
            next_state = next_x * self.grid_size + next_y
            if (self.state, next_state) not in self.walls and (next_state, self.state) not in self.walls:
                self.state = next_state

        reward = 1.0 if self.state == self.goal else (0.5 if self.state in self.bonus_states and np.random.rand() < 0.2 else 0.0)
        terminated = self.state == self.goal
        truncated = self.step_count >= self.max_steps
        return self.state, reward, terminated, truncated, {}

gym.envs.registration.register(id='ComplexMaze-v0', entry_point='complexmaze:ComplexMazeEnv')