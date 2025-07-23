import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collections import defaultdict
import numpy as np
from homework.dynamic.common.gridworld import GridWorld
from homework.dynamic.common.utils import greedy_probs

class FourGridWorld(GridWorld):
    def __init__(self):
        super().__init__()
        self.action_space = [0, 1, 2, 3]
        
        self.reward_map = np.array([
            [0, 0, 0, 0],
            [0, None, 0, None],
            [0, 0, 0, None],
            [None, 0, 0, 1]
        ])
        
        self.goal_state = (3, 3)
        self.wall_states = [(1, 1), (1, 3), (2, 3), (3, 0)]
        self.wall_state = self.wall_states[0]  # render_q 메서드를 위해 추가
        self.start_state = (0, 0)
        self.agent_state = self.start_state

    def next_state(self, state, action):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state in self.wall_states:
            next_state = state

        return next_state
    
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                state = (h, w)
                if state not in self.wall_states:
                    yield state

    def render_q(self, q=None, print_value=True):
        import matplotlib.pyplot as plt
        from homework.dynamic.common import render_helper
        
        renderer = render_helper.Renderer(self.reward_map, self.goal_state, None)
        renderer.set_figure()
        
        ys, xs = renderer.ys, renderer.xs
        ax = renderer.ax
        action_space = [0, 1, 2, 3]

        if q is not None:
            qmax, qmin = max(q.values()), min(q.values())
            qmax = max(qmax, abs(qmin))
            qmin = -1 * qmax
            qmax = 1 if qmax < 1 else qmax
            qmin = -1 if qmin > -1 else qmin

            color_list = ['red', 'white', 'green']
            cmap = plt.cm.LinearSegmentedColormap.from_list(
                'colormap_name', color_list)

            for y in range(ys):
                for x in range(xs):
                    state = (y, x)
                    r = self.reward_map[y, x]
                    
                    if r != 0 and r is not None:
                        txt = 'R ' + str(r)
                        if state == self.goal_state:
                            txt = txt + ' (GOAL)'
                        ax.text(x+.05, ys-y-0.95, txt)

                    if state == self.goal_state:
                        continue

                    tx, ty = x, ys-y-1

                    action_map = {
                        0: ((0.5+tx, 0.5+ty), (tx+1, ty+1), (tx, ty+1)),
                        1: ((tx, ty), (tx+1, ty), (tx+0.5, ty+0.5)),
                        2: ((tx, ty), (tx+0.5, ty+0.5), (tx, ty+1)),
                        3: ((0.5+tx, 0.5+ty), (tx+1, ty), (tx+1, ty+1)),
                    }
                    offset_map = {
                        0: (0.1, 0.8),
                        1: (0.1, 0.1),
                        2: (-0.2, 0.4),
                        3: (0.4, 0.4),
                    }

                    if state in self.wall_states:
                        ax.add_patch(plt.Rectangle((tx, ty), 1, 1, fc=(0.4, 0.4, 0.4, 1.)))
                        continue

                    for action in action_space:
                        if state == self.goal_state:
                            continue

                        tq = q.get((state, action), 0)
                        color_scale = 0.5 + (tq / qmax) / 2  # normalize: 0.0-1.0

                        poly = plt.Polygon(action_map[action], fc=cmap(color_scale))
                        ax.add_patch(poly)

                        if print_value:
                            offset = offset_map[action]
                            ax.text(tx+offset[0], ty+offset[1], "{:12.2f}".format(tq))

        plt.show()

    def render_v(self, v=None, policy=None):
        import matplotlib.pyplot as plt
        from homework.dynamic.common import render_helper
        
        renderer = render_helper.Renderer(self.reward_map, self.goal_state, None)
        renderer.set_figure()
        
        ys, xs = renderer.ys, renderer.xs
        ax = renderer.ax

        if v is not None:
            vmax, vmin = max(v.values()), min(v.values())
            vmax = max(vmax, abs(vmin))
            vmin = -1 * vmax
            vmax = 1 if vmax < 1 else vmax
            vmin = -1 if vmin > -1 else vmin

            color_list = ['red', 'white', 'green']
            cmap = plt.cm.LinearSegmentedColormap.from_list(
                'colormap_name', color_list)

            for y in range(ys):
                for x in range(xs):
                    state = (y, x)
                    r = self.reward_map[y, x]
                    
                    if r != 0 and r is not None:
                        txt = 'R ' + str(r)
                        if state == self.goal_state:
                            txt = txt + ' (GOAL)'
                        ax.text(x+.05, ys-y-0.95, txt)

                    if state == self.goal_state:
                        continue

                    tx, ty = x, ys-y-1

                    if state in self.wall_states:
                        ax.add_patch(plt.Rectangle((tx, ty), 1, 1, fc=(0.4, 0.4, 0.4, 1.)))
                        continue

                    tv = v.get(state, 0)
                    color_scale = 0.5 + (tv / vmax) / 2  # normalize: 0.0-1.0

                    rect = plt.Rectangle((tx, ty), 1, 1, fc=cmap(color_scale))
                    ax.add_patch(rect)

                    if policy is not None:
                        actions = policy.get(state)
                        if actions is None:
                            continue
                        max_actions = [0] * len(actions)
                        for action, prob in actions.items():
                            max_actions[action] = prob
                        renderer.draw_action_policy(max_actions, state, ax)

class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)

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
    env = FourGridWorld()
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

    # Q-table 기반 정책 시각화 (Q + 방향)
    env.render_q(agent.Q)

    # Value Function (V) + Policy (화살표) 통합 시각화
    V = get_v(agent.Q, env)
    policy = get_policy(agent.Q, env)
    env.render_v(V, policy) 