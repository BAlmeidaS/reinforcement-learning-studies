from agent import Agent
from monitor import interact
import gym
import numpy as np

import matplotlib.pyplot as plt


def main():
    env = gym.make('Taxi-v3')
    agent = Agent()
    avg_rewards, best_avg_reward, plot_rewards = interact(env, agent, 200000)

    # plot performance
    plt.plot(np.linspace(0, 200000, len(plot_rewards), endpoint=False),
             np.asarray(plot_rewards))
    plt.ylabel('Average Reward')
    plt.xlabel('Episode Number')
    plt.show()


if __name__ == '__main__':
    main()
