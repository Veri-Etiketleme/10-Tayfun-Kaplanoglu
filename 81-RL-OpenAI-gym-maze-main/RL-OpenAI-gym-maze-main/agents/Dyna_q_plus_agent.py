#!/usr/bin/env python

from typing import Sequence, Tuple
from typing import NamedTuple, Iterable

import numpy as np
import random
import time

import gym
import gym_maze

class StepExperience(NamedTuple):
    """
    represents a single step experience perceived from the agent
    """
    state: int
    action: int
    reward: float
    new_state: int

class EnvModel(object):
    """
    represents the model of the interactions of the agent in the environment
    """
    def __init__(self, observations_count, actions_count):
        self.model_dict = {}
        self.learnt_steps = set()

    def learn(self, *, experience):
        """
        stores an agent experience in the model  
        """
        state, action, reward, new_state = experience
        if state not in self.model_dict:
            self.model_dict[state] = {action: (reward, new_state)}
        else:
            self.model_dict[state][action] = (reward, new_state)

        self.learnt_steps.add((state, action))

    def get_random_experience(self):
        state, action = random.sample(self.learnt_steps, 1)[0]
        reward, new_state = self.model_dict[state][action]
        return StepExperience(state, action, reward, new_state)

class DynaQplusAgent(object):
    """
    Implementation of the Dyna-q plus RL technique
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

    def train_with_planning(self, *, num_episodes=10, planning_steps=5, alpha=0.4, gamma=0.95, kappa=0.00001):
        """
        agent training phase
        """
        print("\nTRAINING (with DynaQ-plus method)")
        observations_count = self.maze_size[0] * self.maze_size[1]
        actions_count = self.env.action_space.n

        # q-table stores the q values for each (state, action) pair
        q_table = np.zeros([observations_count, actions_count])
        # n-table keeps a counter for the number of time a given pair (state, action) 
        # was experienced by the agent
        n_table = np.full([observations_count, actions_count], 0.0001)
        # tau-table keeps a counter of how long each distinct (state, action) pair
        # was not experienced by the agent
        tau = np.zeros([observations_count, actions_count])
        rewards = []
        env_model = EnvModel(observations_count, actions_count)

        for episode in range(num_episodes):
            if episode % (num_episodes // 10) == 0:
                print(f"running episode: {episode} (out of {num_episodes})")

            steps = 0
            observation = self.env.reset()
            state = self.get_state_index_from(observation=observation)

            done = False
            episode_reward = 0

            while not done:
                action = self.ucb_policy(state, q_table, n_table, episode)
                new_observation, reward, done, _ = self.env.step(action)
                new_state = self.get_state_index_from(observation=new_observation)

                tau += 1
                tau[state, action] = 0
                n_table[state, action] = int(n_table[state, action] + 1)
                q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state, action])

                env_model.learn(experience=StepExperience(state, action, reward, new_state))
                
                # planning step
                if len(env_model.learnt_steps) > 5:
                    for _ in range(planning_steps):
                        exp = env_model.get_random_experience()
                        # exploration bonus
                        bonus = kappa * np.sqrt(tau[exp.state, exp.action])
                        expect = gamma * np.max(q_table[exp.new_state])
                        q_table[exp.state, exp.action] = q_table[exp.state, exp.action] + alpha * (exp.reward + bonus + expect - q_table[exp.state, exp.action])

                episode_reward += reward
                steps += 1
                state = new_state

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
        self.env.render()

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

