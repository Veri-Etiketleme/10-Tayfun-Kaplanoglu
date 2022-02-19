#!/usr/bin/env python

from typing import Sequence, Tuple
from typing import NamedTuple, Iterable

import numpy as np
import math
import random
import time

import gym
import gym_maze

class StepExperience(NamedTuple):
    """
    represents a single step experience perceived by the agent
    """
    state: int
    action: int
    reward: float

class MCFirstVisitAgent(object):
    """
    Implementation of the Monte Carlo first visit RL technique
    """

    def __init__(self, env):
        self.env = env
        maze_width = env.observation_space.high[0] - env.observation_space.low[0] + 1
        maze_height = env.observation_space.high[1] - env.observation_space.low[1] + 1
        self.maze_size = (maze_width, maze_height)

    def get_state_index_from(self, *, observation):
        # mapping from observation (x, y) into an integer index
        return int(observation[0] + observation[1] * self.maze_size[0])

    def get_epsilon(self, *, episode, total_episodes):
        """
        Stretched exponential decay
        https://medium.com/analytics-vidhya/stretched-exponential-decay-function-for-epsilon-greedy-algorithm-98da6224c22f
        """
        A = 0.5
        B = 0.1
        C = 0.1
        upper_value = 1.1
        
        standardized_time = (episode - A * total_episodes) / (B * total_episodes)
        cosh = np.cosh(math.exp(-standardized_time))
        epsilon = upper_value - (1 / cosh + (episode * C / total_episodes))
        return epsilon / upper_value

    def epsilon_greedy_policy(self, state, q_table, epsilon):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon
        """
        actions_count = q_table.shape[-1]
        greedy_action = int(np.argmax(q_table[state]))
        probs = np.full(actions_count, epsilon / actions_count)
        probs[greedy_action] += 1 - epsilon
        return int(np.random.choice(range(actions_count), p=probs))

    def on_policy_train(self, *, num_episodes=150, gamma=0.95):
        """
        agent training phase
        """
        print("\nTRAINING (with Monte Carlo method)")
        observations_count = self.maze_size[0] * self.maze_size[1]
        actions_count = self.env.action_space.n

        # q-table stores the q values for each (state, action) pair
        q_table = np.zeros([observations_count, actions_count])
        # n-table keeps a counter for the number of time a given pair (state, action) 
        # was experienced by the agent
        n_table = np.zeros([observations_count, actions_count])
        rewards = []

        for episode in range(num_episodes):
            if episode % (num_episodes // 10) == 0:
                print(f"running episode: {episode} (out of {num_episodes})")

            observation = self.env.reset()
            state = self.get_state_index_from(observation=observation)
            done = False
            # keeps track of all (state, action, reward) experiences of the agent in a single episode
            episode_experiences = []
            episode_reward = 0
            epsilon = self.get_epsilon(episode=episode, total_episodes=num_episodes)

            while not done:
                action = self.epsilon_greedy_policy(state, q_table, epsilon)
                new_observation, reward, done, _ = self.env.step(action)

                episode_experiences.append(StepExperience(state, action, reward))
                episode_reward += reward
                state = self.get_state_index_from(observation=new_observation)

            rewards.append(episode_reward)
            
            # set of all distinct (state, action) pairs experienced by the agent
            # used only to check if experiences have already been encountered in the list of experiences
            visited_state_action_pairs = set()
            # list of indices recording only the first time a distinct (state, action)
            # pair was experienced by the agent in a single episode
            first_occurences = set()
            for index, experience in enumerate(episode_experiences):
                if (experience.state, experience.action) not in visited_state_action_pairs:
                    visited_state_action_pairs.add((experience.state, experience.action))
                    first_occurences.add(index)

            # g(t) is the return (expected reward) for a given (state, action) pair
            g_t = 0
            for index, experience in enumerate(reversed(episode_experiences)):
                # recursive formula to calculate the expected reward for each experience 
                # in an episode going backward from the end
                g_t = g_t * gamma + experience.reward

                if ((len(episode_experiences) - 1) - index) in first_occurences:
                    n_table[experience.state, experience.action] = int(n_table[experience.state, experience.action] + 1)
                    # running average to efficiently keep track of the average rewards
                    q_table[experience.state, experience.action] = q_table[experience.state, experience.action] + 1/(n_table[experience.state, experience.action]) * (g_t - q_table[experience.state, experience.action])

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

