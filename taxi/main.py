from agent_expected_sarsa import AgentExpectedSarsa
from monitor import interact
import gym
import numpy as np

import matplotlib.pyplot as plt

from itertools import product

from tqdm import tqdm

import logging
from default_logger import setup_logging


def main():
    env = gym.make('Taxi-v3')

    agent = AgentExpectedSarsa()
    avg_rewards, best_avg_reward, plot_rewards = interact(env, agent)

    print(f"best_avg_reward: {best_avg_reward}")

    # plot performance
    plt.plot(np.linspace(0, 200000, len(plot_rewards), endpoint=False),
             np.asarray(plot_rewards))
    plt.ylabel('Average Reward')
    plt.xlabel('Episode Number')
    plt.show()


def greedy_search():
    setup_logging("expected_sarsa_gs")

    alphas = np.arange(.8, 1.03, 0.03)
    gammas = np.arange(.8, 1.03, 0.03)
    eps_decay = range(5, 110, 15)

    # linear product of each variable
    vrs = list(product(alphas, gammas, eps_decay))

    # Initializing result dictionary
    results = {}

    # looping for each
    for a, g, e_decay in tqdm(vrs, desc='greedy search'):
        results[(a, g, e_decay)] = run(a, g, e_decay)

    # get the top 10 results
    best_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]

    # log the best results
    for i in best_results:
        logging.info(f"alpha: {i[0][0]} gamma: {i[0][1]} decay: {i[0][2]} - avg_reward: {i[1]}")


def run(alpha, gamma, decay_ep):
    env = gym.make('Taxi-v3')
    agent = AgentExpectedSarsa(alpha=alpha, gamma=gamma)
    avg_rewards, best_avg_reward, plot_rewards = interact(env,
                                                          agent,
                                                          4000,
                                                          decay_epsilon=decay_ep)
    return best_avg_reward


if __name__ == '__main__':
    main()
