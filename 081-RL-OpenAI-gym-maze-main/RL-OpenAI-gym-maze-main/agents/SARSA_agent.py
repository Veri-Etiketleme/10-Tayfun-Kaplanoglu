#!/usr/bin/env python

from typing import Sequence, Tuple
from typing import NamedTuple, Iterable

import numpy as np
import random
import time

import gym
import gym_maze

class SARSAAgent(object):
    """
    Implementation of the SARSA RL technique
    """

    def __init__(self, env):
        self.env = env
        maze_width = env.observation_space.high[0] - env.observation_space.low[0] + 1
        maze_height = env.observation_space.high[1] - env.observation_space.low[1] + 1
        self.maze_size = (maze_width, maze_height)

    def get_state_index_from(self, *, observation):
        # mapping from observation (x, y) into an integer index
        return int(observation[0] + observation[1] * self.maze_size[0])

    def ucb_policy(self, state, q_table, n_table, episode):
        """
        Creates an Upper Confidence Bound policy
        """
        q_values = list(q_table[state])
        q_values_with_bonus = [x + np.sqrt((2 * np.log(episode + 1))/(n_table[state, i])) for i, x in enumerate(q_values)]
        return int(np.argmax(q_values_with_bonus))

    def on_policy_train(self, *, num_episodes=150, alpha=0.4, gamma=0.95):
        """
        agent training phase
        """
        print("\nTRAINING (with SARSA method)")
        observations_count = self.maze_size[0] * self.maze_size[1]
        actions_count = self.env.action_space.n

        # q-table stores the q values for each (state, action) pair
        q_table = np.zeros([observations_count, actions_count])
        # n-table keeps a counter for the number of time a given pair (state, action) 
        # was experienced by the agent
        n_table = np.full([observations_count, actions_count], 0.0001)
        rewards = []

        for episode in range(num_episodes):
            if episode % (num_episodes // 10) == 0:
                print(f"running episode: {episode} (out of {num_episodes})")

            steps = 1
            observation = self.env.reset()
            state = self.get_state_index_from(observation=observation)
            action = self.ucb_policy(state, q_table, n_table, steps)

            done = False
            episode_reward = 0

            while not done:
                new_observation, reward, done, _ = self.env.step(action)
                new_state = self.get_state_index_from(observation=new_observation)
                next_action = self.ucb_policy(new_state, q_table, n_table, episode)

                n_table[state, action] = int(n_table[state, action] + 1)
                q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * q_table[new_state, next_action] - q_table[state, action])

                episode_reward += reward
                steps += 1
                state = new_state
                action = next_action

            rewards.append(episode_reward) 

        print()
        print("-" * 22 + " N-table " + "-" * 22)
        print(n_table, end="\n\n")

        print("-" * 22 + " Q-table " + "-" * 22)
        print(q_table, end="\n\n")

        print("-" * 20 + " Training rewards " + "-" * 20)
        print(rewards)
    
        return q_table, rewards

    def run_optimal(self, *, with_q_table):
        """
        run the agent in the given environment following the policy being calculated
        """
        q_table = with_q_table

        observation = self.env.reset()
        state = self.get_state_index_from(observation=observation)

        done = False
        episode_reward = 0

        while not done:
            action = int(np.argmax(q_table[state]))
            new_observation, reward, done, _ = self.env.step(action)
            new_state = self.get_state_index_from(observation=new_observation)
            self.env.render()
            time.sleep(0.3)

            episode_reward += reward
            state = new_state

        print()
        print(f"Reward following the optimal policy: {episode_reward}")
        self.env.close()

