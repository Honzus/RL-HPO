import numpy as np
from collections import defaultdict

class TabularRL:
    def __init__(self, env, learning_rate, epsilon, discount_factor):
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.q_sa = defaultdict(lambda: np.zeros(env.action_space.n))

    def select_action(self, s_t):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_sa[s_t]))

    def update_qlearning(self, s_t, a_t, s_t1, r_t1, terminated):
        self.q_sa[s_t][a_t] = self.q_sa[s_t][a_t] + self.learning_rate * (
                    r_t1 + self.discount_factor * (not terminated) * np.max(self.q_sa[s_t1]) - self.q_sa[s_t][a_t])

    def update_sarsa(self, s_t, a_t, s_t1, a_t1, r_t1, terminated):
        self.q_sa[s_t][a_t] = self.q_sa[s_t][a_t] + self.learning_rate * (
                    r_t1 + self.discount_factor * (not terminated) * self.q_sa[s_t1][a_t1] - self.q_sa[s_t][a_t])