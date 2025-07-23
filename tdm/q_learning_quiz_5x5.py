import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collections import defaultdict
import numpy as np
from homework.dynamic.common.gridworld import FiveGridWorld
from homework.dynamic.common.utils import greedy_probs
from homework.tdm.q_learning import QLearningAgent

def get_v(Q, env):
    V = defaultdict(float)
    for state in env.states():
        qs = [Q[(state, action)] for action in range(4)]
        V[state] = max(qs)
    return V

def get_policy(Q, env):
    policy = {}
    for state in env.states():
        qs = [Q[(state, action)] for action in range(4)]
        max_action = np.argmax(qs)
        probs = {a: 0.0 for a in range(4)}
        probs[max_action] = 1.0
        policy[state] = probs
    return policy


if __name__ == '__main__':
    env = FiveGridWorld()
    agent = QLearningAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            if done:
                break
            state = next_state

    # 1. Q-table 기반 정책 시각화 (Q + 방향)
    env.render_q(agent.Q)

    # 2. Value Function (V) + Policy (화살표) 통합 시각화
    V = get_v(agent.Q, env)
    policy = get_policy(agent.Q, env)
    env.render_v(V, policy)