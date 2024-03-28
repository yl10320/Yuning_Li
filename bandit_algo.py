import numpy as np


class Method:
    def __init__(self):
        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)

    def action(self):
        raise NotImplementedError()

    def update(self, arm, reward):
        pass

    def reset(self):
        pass


class SampleAverageMethod(Method):

    def __init__(self, n=10, epsilon=0.1, opt_init=0.0):
        super().__init__()
        self.n, self.epsilon, self.opt_init = n, epsilon, opt_init
        self.N = None
        self.Q = None
        self.reset()

    def action(self):
        if self.np_random.rand() < self.epsilon:
            action = self.np_random.randint(0, self.n)
        else:
            maxi = np.max(self.Q)
            action = self.np_random.choice(np.flatnonzero(self.Q == maxi))
        return action

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def reset(self):
        self.N = np.zeros([self.n])
        self.Q = np.ones([self.n]) * self.opt_init


class ConstStepSizeMethod(Method):

    def __init__(self, n=10, epsilon=0.1, alpha=0.1):
        super().__init__()
        self.n, self.epsilon, self.alpha = n, epsilon, alpha
        self.Q = None
        self.reset()

    def action(self):
        if self.np_random.rand() < self.epsilon:
            action = self.np_random.randint(0, self.n)
        else:
            maxi = np.max(self.Q)
            action = self.np_random.choice(np.flatnonzero(self.Q == maxi))
        return action

    def update(self, action, reward):
        self.Q[action] += (reward - self.Q[action]) * self.alpha

    def reset(self):
        self.Q = np.zeros([self.n])


class UCBMethod(Method):

    def __init__(self, n=10, c=2.0, epsilon=0.1):
        super().__init__()
        self.n, self.c, self.epsilon = n, c, epsilon
        self.N = None
        self.Q = None
        self.reset()

    def ucb(self):
        t = np.sum(self.N) + 1
        ucb = self.Q + self.c * np.sqrt(np.log(t) / (self.N + 1e-5))
        return ucb

    def action(self):
        ucb = self.ucb()
        if self.np_random.rand() < self.epsilon:
            action = self.np_random.randint(0, self.n)
        else:
            maxi = np.max(ucb)
            action = self.np_random.choice(np.flatnonzero(ucb == maxi))
        return action

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def reset(self):
        self.N = np.zeros([self.n])
        self.Q = np.zeros([self.n])
