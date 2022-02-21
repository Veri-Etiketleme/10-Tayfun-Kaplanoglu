import os, sys
sys.path.insert(1, os.path.abspath("agents/"))

from agents.MC_first_visit_agent import MCFirstVisitAgent
from agents.SARSA_agent import SARSAAgent
from agents.Q_learning_agent import QLearningAgent
from agents.Dyna_q_plus_agent import DynaQplusAgent

import argparse
import gym
import gym_maze

parser = argparse.ArgumentParser()
parser.add_argument('--method', 
                    default='dynaqplus', 
                    help='specify the RL technique to be used')
arguments = parser.parse_args()

maze_env = gym.make("maze-random-10x10-plus-v0")

def run_monte_carlo_method(*, env):
    mc_agent = MCFirstVisitAgent(env=env)
    q_table, rewards = mc_agent.on_policy_train()
    mc_agent.run_optimal(with_q_table=q_table)

def run_sarsa_method(*, env):
    sarsa_agent = SARSAAgent(env=env)
    q_table, rewards = sarsa_agent.on_policy_train()
    sarsa_agent.run_optimal(with_q_table=q_table)

def run_q_learning_method(*, env):
    qlearning_agent = QLearningAgent(env=env)
    q_table, rewards = qlearning_agent.off_policy_train()
    qlearning_agent.run_optimal(with_q_table=q_table)

def run_dyna_q_plus_method(*, env):
    dyna_q_plus_agent = DynaQplusAgent(env=env)
    q_table, rewards = dyna_q_plus_agent.train_with_planning()
    dyna_q_plus_agent.run_optimal(with_q_table=q_table)

switcher = {
    "montecarlo": run_monte_carlo_method,
    "qlearning": run_q_learning_method,
    "sarsa": run_sarsa_method,
    "dynaqplus": run_dyna_q_plus_method
}
solver = switcher.get(arguments.method, run_dyna_q_plus_method)
solver(env=maze_env)







