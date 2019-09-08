import numpy as np
from collections import defaultdict


stack = []


def epsilon_policy(q_s, e, nA):
    if np.array_equal(q_s, np.zeros(nA)):
        return np.ones(nA) * 1/nA

    policy_s = np.ones(nA) * [e/nA]
    best_action = np.argmax(q_s)
    policy_s[best_action] = 1 - e + e/nA
    return policy_s


# def epsilon_policy_new(q_s, e, nA):
#     if np.array_equal(q_s, np.zeros(nA)):
#         return np.ones(nA) * 1/nA
#
#     policy_s = np.zeros(nA)
#     best_action = np.argmax(q_s)
#     policy_s[best_action] = 1
#     return policy_s


class Subtask:
    def __init__(self):
        self.C = defaultdict(lambda: np.zeros(len(self.nodes)))

    def select_action(self, state, epsilon):
        index = np.random.choice(len(self.nodes),
                                 p=epsilon_policy(self.C[state],
                                                  epsilon,
                                                  len(self.nodes)))

        stack.append((self, state, index))

        subtask = self.nodes[index]

        return subtask.select_action(state, epsilon)


class AgentMaxQ(Subtask):
    def __init__(self, alpha=0.9, gamma=1):
        self.nodes = [Get(),
                      Put()]

        self.alpha = alpha
        self.gamma = gamma

        super().__init__()

    def select_action(self, state, epsilon):
        stack.clear()

        return super().select_action(state, epsilon)

    def step(self, state, action, reward, next_state, done, epsilon):

        for node, state, action in stack:
            future_policy = epsilon_policy(node.C[next_state], epsilon, len(node.nodes))

            node.C[state][action] += self.alpha * (reward
                                                   + (self.gamma
                                                      * future_policy.dot(node.C[next_state]))
                                                   - node.C[state][action])

        stack.clear()


class Get(Subtask):
    def __init__(self):
        super().__init__()

        self.nodes = [Navigate(),
                      Pickup()]


class Put(Subtask):
    def __init__(self):
        super().__init__()

        self.nodes = [Navigate(),
                      Putdown()]


class Navigate(Subtask):
    def __init__(self):
        super().__init__()

        self.nodes = [South(),
                      North(),
                      East(),
                      West()]


class South:
    def select_action(self, state, *args):
        return 0


class North:
    def select_action(self, state, *args):
        return 1


class East:
    def select_action(self, state, *args):
        return 2


class West:
    def select_action(self, state, *args):
        return 3


class Pickup:
    def select_action(self, state, *args):
        return 4


class Putdown:
    def select_action(self, state, *args):
        return 5
