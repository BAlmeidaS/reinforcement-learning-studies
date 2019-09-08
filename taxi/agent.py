import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state, epsilon):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA,
                                p=self._epsilon_policy(self.Q[state], epsilon))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += reward

    def _epsilon_policy(self, q_s, e):
        if np.array_equal(q_s, np.zeros(self.nA)):
            return np.ones(self.nA) * 1/self.nA

        policy_s = np.ones(self.nA) * [e/self.nA]
        best_action = np.argmax(q_s)
        policy_s[best_action] = 1 - e + e/self.nA
        return policy_s
