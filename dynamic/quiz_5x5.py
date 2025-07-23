from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld
from policy_eval import policy_eval
import argparse
import matplotlib
import matplotlib.pyplot as plt
import common.gridworld_render as render_helper

class FiveGridWorld(GridWorld):
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        
        self.reward_map = np.array([
            [0, 0, 0, -1, 1],
            [0, 0, 0, 0, 0],
            [0, None, None, 0, 0],
            [0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0]
        ])
        
        self.goal_state = (0, 4)
        self.wall_state = (2, 1)
        self.wall_states = [(2, 1), (2, 2)]
        self.start_state = (4, 0)

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
    
    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state, self.wall_states)
        
        renderer.set_figure()
        
        ys, xs = renderer.ys, renderer.xs
        ax = renderer.ax
        
        if v is not None:
            color_list = ['red', 'white', 'green']
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'colormap_name', color_list)
            
            v_dict = v
            v_array = np.zeros(self.reward_map.shape)
            for state, value in v_dict.items():
                v_array[state] = value
            
            vmax, vmin = v_array.max(), v_array.min()
            vmax = max(vmax, abs(vmin))
            vmin = -1 * vmax
            vmax = 1 if vmax < 1 else vmax
            vmin = -1 if vmin > -1 else vmin
            
            ax.pcolormesh(np.flipud(v_array), cmap=cmap, vmin=vmin, vmax=vmax)
        
        for y in range(ys):
            for x in range(xs):
                state = (y, x)
                r = self.reward_map[y, x]
                
                if r != 0 and r is not None:
                    txt = 'R ' + str(r)
                    if state == self.goal_state:
                        txt = txt + ' (GOAL)'
                    ax.text(x+.1, ys-y-0.9, txt)
                
                if (v is not None) and state not in self.wall_states:
                    if print_value:
                        offsets = [(0.4, -0.15), (-0.15, -0.3)]
                        key = 0
                        if v_array.shape[0] > 7: key = 1
                        offset = offsets[key]
                        ax.text(x+offset[0], ys-y+offset[1], "{:12.2f}".format(v_array[y, x]))
                
                if policy is not None and state not in self.wall_states:
                    actions = policy[state]
                    max_actions = [kv[0] for kv in actions.items() if kv[1] == max(actions.values())]
                    
                    arrows = ["↑", "↓", "←", "→"]
                    offsets = [(0, 0.1), (0, -0.1), (-0.1, 0), (0.1, 0)]
                    for action in max_actions:
                        arrow = arrows[action]
                        offset = offsets[action]
                        if state == self.goal_state:
                            continue
                        ax.text(x+0.45+offset[0], ys-y-0.5+offset[1], arrow)
                
                if state in self.wall_states:
                    ax.add_patch(plt.Rectangle((x,ys-y-1), 1, 1, fc=(0.4, 0.4, 0.4, 1.)))
        
        plt.show()

def argmax(d):
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V, env, gamma):
    pi = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value

        max_action = argmax(action_values)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi


def policy_iter(env, gamma, threshold=0.001, is_render=True):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold) 
        new_pi = greedy_policy(V, env, gamma)  

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:
            break
        pi = new_pi

    return pi

def value_iter_onestep(V, env, gamma):
    for state in env.states():       
        if state == env.goal_state: 
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():  
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]  
            action_values.append(value)

        V[state] = max(action_values) 
    return V


def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy() 
        V = value_iter_onestep(V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', action='store_true')
    parser.add_argument('--value', action='store_true')
    args = parser.parse_args()
    

    if args.policy:
        env = FiveGridWorld()
        gamma = 0.9
        pi = policy_iter(env, gamma)
        
    if args.value:  
        env = FiveGridWorld()
        gamma = 0.9
        V = defaultdict(lambda: 0)
        V = value_iter(V, env, gamma)
        pi = greedy_policy(V, env, gamma)
        env.render_v(V, pi)