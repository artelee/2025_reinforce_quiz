import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collections import defaultdict
import numpy as np
from homework.dynamic.common.gridworld import FiveGridWorld
from homework.monte.mc_control import McAgent

def run_experiment(epsilon, alpha, episodes=10000):
    env = FiveGridWorld()
    agent = McAgent()
    agent.epsilon = epsilon
    agent.alpha = alpha
    
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)
            if done:
                agent.update()
                break

            state = next_state
    
    return agent.Q, env

if __name__ == '__main__':
    # 실험 1: epsilon 변경 (alpha=0.1 고정)
    epsilons = [0.01, 0.1, 0.3]
    alpha = 0.1
    
    for eps in epsilons:
        Q, env = run_experiment(epsilon=eps, alpha=alpha)
        env.render_q(Q)
        
    # 실험 2: alpha 변경 (epsilon=0.1 고정)
    alphas = [0.01, 0.1, 0.3]
    epsilon = 0.1
    
    for alp in alphas:
        Q, env = run_experiment(epsilon=epsilon, alpha=alp)
        env.render_q(Q)
        