from agent import Agent
from monitor import interact
import gym
import numpy as np

import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)

# plot performance
plt.plot(np.linspace(0, 20000, len(avg_rewards), endpoint=False),
         np.asarray(avg_rewards))
plt.ylabel('Average Reward')
plt.xlabel('Episode Number')
plt.show()
