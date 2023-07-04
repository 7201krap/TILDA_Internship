import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from gym import spaces

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MazeEnv, self).__init__()

        self.action_space = spaces.Discrete(4)  # four actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 3, 1), dtype=np.float32)  # grid of 3x3

        self.maze = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1],
            [-1, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, -1],
            [-1, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1],
            [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, -1],
            [-1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, 0, 0, -1, -1, 0, -1],
            [-1, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, -1],
            [-1, 0, 0, 0, -1, 0, -1, -1, -1, -1, 0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])

        self.start = (1, 1)
        self.end = (10, 10)
        self.state = self.start
        self.visited_states = np.zeros(self.maze.shape)

        self.done = False
        self.total_steps = 0

    def step(self, action):
        if action == 0:  # up
            next_state = (self.state[0] - 1, self.state[1])
        elif action == 1:  # right
            next_state = (self.state[0], self.state[1] + 1)
        elif action == 2:  # down
            next_state = (self.state[0] + 1, self.state[1])
        elif action == 3:  # left
            next_state = (self.state[0], self.state[1] - 1)
        else:
            raise ValueError("Invalid action")

        reward = 0  # default reward for moving to any cell

        if (next_state[0] < 0 or next_state[0] >= self.maze.shape[0] or
                next_state[1] < 0 or next_state[1] >= self.maze.shape[1] or
                self.maze[next_state] == -1):
            next_state = self.state
            reward = -1  # penalize for hitting the wall
        elif self.visited_states[next_state] == 0:  # unvisited state
            self.visited_states[next_state] = 1
            reward = 1  # reward for exploring a new state

        is_done = (next_state == self.end)

        if is_done:
            num_visited_states = np.count_nonzero(self.visited_states)
            reward = 100 * (num_visited_states / self.total_steps)  # reward function taking into account efficiency
            is_done = True
        elif self.total_steps >= 1000:  # if agent exceeds max steps without reaching the goal
            reward = -20
            is_done = True

        self.state = next_state
        self.total_steps += 1

        return np.expand_dims(self._get_obs(), axis=-1), reward, is_done, {}

    def reset(self):
        self.start = (1, 1)
        self.end = (10, 10)
        self.state = self.start
        self.visited_states = np.zeros(self.maze.shape)

        self.done = False
        self.total_steps = 0
        return np.expand_dims(self._get_obs(), axis=-1)

    def render(self, mode='human'):
        out = self.maze.copy()
        color_dict = {-1: 'black', 0: 'white', 1: 'red', 'E': 'green'}
        cmap = mcolors.ListedColormap(list(color_dict.values()))
        out[self.state] = 1
        out[self.end] = 2
        plt.matshow(out, cmap=cmap)
        plt.grid(visible=False)
        plt.pause(0.05)
        plt.close()

    def _get_obs(self):
        obs = np.zeros((3, 3)).astype(np.float32)
        for i in range(-1, 2):
            for j in range(-1, 2):
                x = self.state[0] + i
                y = self.state[1] + j
                if (x >= 0 and x < self.maze.shape[0] and
                    y >= 0 and y < self.maze.shape[1]):
                    obs[i+1, j+1] = self.maze[x, y]
                else:
                    obs[i+1, j+1] = -1  # treat out of bounds as walls
        obs[1, 1] = 1  # mark the current position with 1
        return obs
