from agent_maxq import AgentMaxQ
from monitor import interact
import gym
import numpy as np

import matplotlib.pyplot as plt

from itertools import product

from tqdm import tqdm

import logging
from default_logger import setup_logging

import ray


def main():
    env = gym.make('Taxi-v2')

    agent = AgentMaxQ(alpha=.1, gamma=1)
    avg_rewards, best_avg_reward, plot_rewards = interact(env, agent, 100000, decay_epsilon=10)

    print(f"best_avg_reward: {best_avg_reward}")

    # plot performance
    plt.plot(np.linspace(0, 50000, len(plot_rewards), endpoint=False),
             np.asarray(plot_rewards))
    plt.title(f'Best Reward: {best_avg_reward}')
    plt.ylabel('Average Reward')
    plt.xlabel('Episode Number')
    plt.show()


@ray.remote
def greedy_search(x, total):
    setup_logging("maxq_gs_" + str(x))

    alphas = np.array([.1, .2, .5, .9, .95, 1])
    gammas = np.array([.9, .95, .99, 1])
    eps_decay = [10, 30, 70, 100, 150]

    # linear product of each variable
    all_objs = list(product(alphas, gammas, eps_decay))

    np.random.shuffle(all_objs)

    fator = int(len(all_objs)/total)

    if x == 0:
        vrs = all_objs[:fator]
    elif x == (total-1):
        vrs = all_objs[x*fator:]
    else:
        vrs = all_objs[x*fator: (x+1)*fator]

    # Initializing result dictionary
    results = {}

    # looping for each
    for a, g, e_decay in tqdm(vrs, desc='greedy search', leave=False):
        results[(a, g, e_decay)] = run(a, g, e_decay)

    # get the top 100 results
    best_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:50]

    # log the best results
    for i in best_results:
        logging.info(f"alpha: {i[0][0]} gamma: {i[0][1]} decay: {i[0][2]} - avg_reward: {i[1]}")

#    logging.info(best_results)


def parallel():
    ray.init()
    try:
        futures = [greedy_search.remote(i, 8) for i in range(8)]
        ray.get(futures)
    finally:
        ray.shutdown()


def run(alpha, gamma, decay_ep):
    env = gym.make('Taxi-v3')
    agent = AgentMaxQ(alpha=alpha, gamma=gamma)
    avg_rewards, best_avg_reward, plot_rewards = interact(env,
                                                          agent,
                                                          20000,
                                                          decay_epsilon=decay_ep)
    return best_avg_reward


if __name__ == '__main__':
    main()
