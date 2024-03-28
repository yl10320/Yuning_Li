import numpy as np


class Testbed:
    def __init__(self, n=10, stationary=True):
        self.n = n
        self.stationary = stationary
        self.q_star = None
        self.a_star = None
        self.reset()

    def reset(self):
        if self.stationary:
            self.q_star = np.random.normal(0, 1, self.n)
        else:
            self.q_star = np.zeros(self.n)
        self.a_star = np.argmax(self.q_star)

    def step(self, action):
        reward = np.random.normal(self.q_star[action], 1)
        if not self.stationary:
            self.q_star += np.random.normal(0, 0.01, self.n)
            self.a_star = np.argmax(self.q_star)
        regret = np.max(self.q_star) - self.q_star[action]
        return reward, regret


def simulate_bandit(bandit, method, episodes=2000, steps=1000):
    rewards = np.zeros(steps)
    optimal = np.zeros(steps)
    regrets = np.zeros(steps)

    for i in range(episodes):
        bandit.reset()
        method.reset()

        for step in range(steps):
            action = method.action()
            reward, regret = bandit.step(action)
            method.update(action, reward)

            rewards[step] += reward
            regrets[step] += regret
            if action == bandit.a_star:
                optimal[step] += 1

    AverageReward = rewards / episodes
    PercentOptimal = optimal / episodes
    TotalRegret = np.cumsum(regrets / episodes)

    return AverageReward, PercentOptimal, TotalRegret


