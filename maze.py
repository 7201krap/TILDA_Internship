"""
Created by Jin Hyun Park @ Texas A&M University (Tilda Corp.)
Date: 2023-05-31
Experimental setup for a maze solving problem in Genetic Algorithm.
    Objective:
        - Find a robot that navigates the maze
    Fitness function (f):
        1. Based on Novelty:
            f = Euclidean distance between the ending positions of two individuals.
            We are able to get a value of sparseness(=sp) of an individual by using k-nearest neighbours.
            sparseness(=sp) is calculated by the following:
                sp(x) = 1/k * \sum_{i=0}^{k} dist(x, u_i) where u_i is the i-th nearest neighbor of x with respect to the distance metric 'dist'.
            We can calculate a novelty score based on the sparseness.
            The bigger the sparseness value, the higher the novelty score.
        2. Based on Fitness:
            f = b_f - d_g
                b_f is the longest path from the starting point to the end goal
                d_g is the distance from the robot to the end goal
    Population Size:
        - 1000 individuals
    When to terminate the program?:
        - 1000 generations
    Map size?
        - 20 x 20
    How many time steps are allowed for a robot(=agent)?
        1. medium map: 100 time steps
        2. hard map: 400 time steps
    There are two maps:
        1. medium map
        2. hard map
    Possible actions of a robot(=agent)
        1. move forward
        2. turn right without moving
        3. turn left without moving
    Constraints:
        1. act conditionally when a wall is encountered. Do a random action when the wall is encountered (25% prob. for each action)
    Three experiments should be conducted:
        1. Based on Novelty
        2. Based on Fitness
        3. Random
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

# Create a 20x20 grid to be all open space (1)
maze = np.ones((20, 20))

# Set the starting point to be the agent (2)
agent_position = [0, 0]
maze[agent_position[0]][agent_position[1]] = 2

# Set the center point to be the goal (3)
maze[10][10] = 3
goal_position = [10, 10]

# Add a 'ㄷ' shaped wall around the goal (0)
maze[9:12, 8] = 0
maze[9:12, 12] = 0
maze[9, 8:12] = 0

# Define the color mapping
cmap = ListedColormap(['black', 'white', 'red', 'green'])

# Create the figure and display the maze
fig, ax = plt.subplots()
im = ax.imshow(maze, cmap=cmap)

# Hide the gridlines
ax.grid(False)

# Hide the x and y axis
ax.set_xticks([])
ax.set_yticks([])

# Direction definitions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Initially, the agent is facing right
agent_direction = RIGHT

# Function to update the agent's position
def update_agent_position():
    old_position = agent_position.copy()
    new_position = agent_position.copy()

    # Adjust the new position based on the agent's direction
    if agent_direction == UP:
        new_position[0] = max(agent_position[0] - 1, 0)
    elif agent_direction == DOWN:
        new_position[0] = min(agent_position[0] + 1, maze.shape[0] - 1)
    elif agent_direction == LEFT:
        new_position[1] = max(agent_position[1] - 1, 0)
    elif agent_direction == RIGHT:
        new_position[1] = min(agent_position[1] + 1, maze.shape[1] - 1)

    # If the new position is a wall, don't move
    if maze[new_position[0]][new_position[1]] != 0:
        agent_position[:] = new_position

        maze[old_position[0]][old_position[1]] = 1  # Set old position to be open space
        maze[new_position[0]][new_position[1]] = 2  # Set new position to be the agent

        # Update the display
        im.set_data(maze)
        fig.canvas.draw()

    # If the agent reaches the goal, set the goal cell to be the agent
    if agent_position == goal_position:
        print("Goal reached!")
        plt.close()

# Function to change the agent's direction
def change_agent_direction(command):
    global agent_direction

    if command == 'L':
        agent_direction = (agent_direction - 1) % 4
    elif command == 'R':
        agent_direction = (agent_direction + 1) % 4

# Define a sequence of commands
commands = ['R', 'M', 'M', 'M', 'M', 'L', 'M', 'M']

for command in commands:
    if command == 'M':
        update_agent_position()
        plt.pause(0.5)  # Pause for half a second
    else:  # The command is either 'L' or 'R'
        change_agent_direction(command)

# Show the figure
plt.show()

