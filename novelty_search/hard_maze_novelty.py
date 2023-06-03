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
import copy
import numpy as np
import matplotlib.pyplot as plt
import random

from tqdm import tqdm
from matplotlib.colors import ListedColormap
from scipy.spatial import distance
from matplotlib.animation import FuncAnimation

# The following block sets up the maze
# ----------------------------------------------------------------------------------------------------------------------
maze = np.ones((20, 20))
agent_position = [0, 0]
maze[agent_position[0]][agent_position[1]] = 2

# Move the goal to the bottom right corner
goal_position = [14, 14]
maze[goal_position[0]][goal_position[1]] = 3

# Add more complicated walls around the maze
maze[1:19, 2] = 0  # long vertical wall
maze[1, 2:17] = 0  # long horizontal wall near top
maze[1:10, 16] = 0  # vertical wall, middle
maze[10:19, 12] = 0  # vertical wall, middle to bottom
maze[10, 2:16] = 0  # horizontal wall, middle
maze[18, 12:19] = 0  # horizontal wall, near bottom
maze[14, 6:12] = 0  # horizontal wall, bottom part
maze[14:18, 6] = 0  # vertical wall, bottom part
maze[5, 6:10] = 0  # horizontal wall, top part
maze[5:9, 10] = 0  # vertical wall, top part

# Adding some dead ends
maze[4, 13:15] = 0
maze[7:9, 14] = 0
maze[15, 7:9] = 0
maze[13:15, 8] = 0

# Openings in the walls to create paths
maze[10, 2] = 1
maze[1, 10] = 1
maze[10, 16] = 1
maze[18, 16] = 1
maze[14, 12] = 1
maze[5, 6] = 1

# Define the color mapping
cmap = ListedColormap(['black', 'white', 'red', 'green'])
# ----------------------------------------------------------------------------------------------------------------------

# Direction definitions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Initially, the agent is facing right
# Global Variables
agent_direction = RIGHT
first_run = True


def plot_agent_positions(positions):
    """
    :param positions: the current population's coordinates with shape (num_individuals, 2)
    :return: a plot depicts final positions
    """
    maze = np.ones((20, 20))

    # Add more complicated walls around the maze
    maze[1:19, 2] = 0  # long vertical wall
    maze[1, 2:17] = 0  # long horizontal wall near top
    maze[1:10, 16] = 0  # vertical wall, middle
    maze[10:19, 12] = 0  # vertical wall, middle to bottom
    maze[10, 2:16] = 0  # horizontal wall, middle
    maze[18, 12:19] = 0  # horizontal wall, near bottom
    maze[14, 6:12] = 0  # horizontal wall, bottom part
    maze[14:18, 6] = 0  # vertical wall, bottom part
    maze[5, 6:10] = 0  # horizontal wall, top part
    maze[5:9, 10] = 0  # vertical wall, top part

    # Adding some dead ends
    maze[4, 13:15] = 0
    maze[7:9, 14] = 0
    maze[15, 7:9] = 0
    maze[13:15, 8] = 0

    # Openings in the walls to create paths
    maze[10, 2] = 1
    maze[1, 10] = 1
    maze[10, 16] = 1
    maze[18, 16] = 1
    maze[14, 12] = 1
    maze[5, 6] = 1

    # Set the starting point and goal position
    agent_position = [0, 0]
    goal_position = [14, 14]

    # Create a copy of the maze to modify
    maze_copy = np.copy(maze)

    # Define the color mapping
    cmap = ListedColormap(['black', 'white', 'red', 'green'])

    # Count the agent's positions
    counts = np.zeros_like(maze, dtype=float)
    for position in positions:
        counts[position[0]][position[1]] += 1

    # Normalize the counts to range between 0 and 1
    max_count = np.max(counts)
    if max_count > 0:
        counts /= max_count

    # Set the starting point to be the agent
    maze_copy[agent_position[0]][agent_position[1]] = 2

    # Set the center point to be the goal
    maze_copy[goal_position[0]][goal_position[1]] = 3

    # Create the figure and display the maze
    fig, ax1 = plt.subplots(figsize=(10, 10))
    _ = ax1.imshow(maze_copy, cmap=cmap)

    # Overlay the agent positions with varying opacities
    for i in range(maze_copy.shape[0]):
        for j in range(maze_copy.shape[1]):
            if counts[i, j] > 0:
                ax1.scatter(j, i, color='red', alpha=counts[i, j])

    # Hide the gridlines
    ax1.grid(False)
    plt.savefig('results/hard maze novelty final agent pos')

    plt.show(block=False)


# Function to update the agent's position
def update_agent_position():
    """
    Update agent position
    """
    global goal_position, agent_position, agent_direction

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

    # If the agent reaches the goal, set the goal cell to be the agent
    if agent_position == goal_position:
        print("* Goal reached!")
        plt.close()


def coordinates(individual):
    """
    Given a sequence, return the final agent position
    :param individual e.g. ['R', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'L']
    :return: agent_position
    """
    global agent_position, agent_direction

    # reset the agent's position and direction
    agent_position = [0, 0]
    agent_direction = RIGHT

    for command in individual:
        if command == 'M':
            update_agent_position()  # Update the agent's position without visualization
        elif command == 'L':
            agent_direction = (agent_direction - 1) % 4
        elif command == 'R':
            agent_direction = (agent_direction + 1) % 4

    # Return the final position of the agent
    agent_position = np.array(agent_position)
    return agent_position


def find_coordinates(population):
    """
    Calculates each individual's coordinate
    :param population: individuals. (num_individuals, num_sequence)
    :return: all individual's coordinates. (num_individuals, 2)
    """
    result = np.array([coordinates(individual) for individual in tqdm(population)])
    return result


def sparseness(population_coordinates, k=25):
    """
    Calculates the sparseness(=fitness) for each individual in a population.
    The paper 'Efficiently evolving programs through the search for Novelty' defines that sparseness determines fitness.
    The higher the sparseness, the higher the fitness
    :param population_coordinates: numpy array of shape (n_individuals, 2) representing final positions
    :param k: number of nearest neighbors to consider
    :return: numpy array of shape (n_individuals,) representing the sparseness of each individual
    """

    # Compute the distance between each pair of individuals
    dists = distance.cdist(population_coordinates, population_coordinates, 'euclidean')

    # Sort each row (each individual's distances to all others)
    dists.sort(axis=1)

    # Get the sum of distances to the k nearest neighbors (excluding the individual itself: dists[i, 0] = 0)
    nearest_k_dists_sum = dists[:, 1:k + 1].sum(axis=1)

    # Divide by k to get the average
    sparseness_values = np.array(nearest_k_dists_sum / k)

    print(sparseness_values.shape)

    return sparseness_values


def pick_mate_index(scores):
    """
    returns an index based on roulette wheel selection.
    The higher the sparseness, the higher the fitness
    scores == fitnesses
    :param sparseness: numpy array of shape (num_individuals,) representing the sparseness of each individual
    :return: an index chosen by roulette wheel selection
    """
    scores = np.array(scores)
    ranks = np.argsort(-scores)

    fitnesses = [len(ranks) - x for x in ranks]  # fitnesses for routes. One fitness value for one route

    fitnesses_sum = np.sum(fitnesses)

    select_prob = [fitness / fitnesses_sum for fitness in fitnesses]

    index = random.choices(range(len(select_prob)), select_prob)[0]

    return index


def preprocess_individuals(individuals):
    """
    Truncates each individual's sequence as soon as the goal is reached
    :param individuals: numpy array of shape (n_individuals, sequence_length)
    :param goal_position: 2D position of the goal in the maze
    :return
        new_individuals: list of lists where each individual's sequence is truncated at the goal
    """
    global agent_position, agent_direction, goal_position
    new_individuals = []

    for individual in tqdm(individuals):
        # reset the agent's position and direction
        new_sequence = []
        agent_position = [0, 0]
        agent_direction = RIGHT
        for command in individual:
            new_sequence.append(command)
            if command == 'M':
                update_agent_position()  # Update the agent's position without visualization
            elif command == 'L':
                agent_direction = (agent_direction - 1) % 4
            elif command == 'R':
                agent_direction = (agent_direction + 1) % 4

            # Check if the agent has reached the goal
            if agent_position == goal_position:
                # print("* Truncation activated")
                break

        new_sequence = np.array(new_sequence)
        new_individuals.append(new_sequence)

    new_individuals = np.array(new_individuals, dtype=list)
    return new_individuals


def crossover(parent1, parent2):
    """
    Performs one point crossover between parent1 and parent2
    :param parent1: list of genetic material (e.g. list of commands)
    :param parent2: list of genetic material
    :return: offspring1, offspring2: new offspring generated by the crossover
    """

    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    # Find the minimum length of the two parents
    min_length = min(len(parent1), len(parent2))

    if min_length >= 3:
        # Randomly choose an index excluding the first and the last index
        crossover_point = random.randint(1, min_length - 2)

        # Create offspring by swapping the segments before the crossover point
        offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return offspring1, offspring2

    else:
        return parent1, parent2


def mutate(individuals, mutation_rate=0.005):
    """
    Apply mutation to a population of individuals
    :param individuals: a list of individuals, where an individual is a list of moves (R, L, M)
    :param mutation_rate: the chance of a command being mutated
    :return: a new list of individuals after mutation
    """
    # List of possible moves
    moves = ['R', 'L', 'M']

    for individual in individuals:
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                # Replace the move at the current position with a random new move
                individual[i] = random.choice(moves)

    return individuals


def shortest_distance_to_goal(curr_pos):
    """
    Calculate the shortest Euclidean distance from the individuals to the goal
    :param curr_pos: a list of final positions of the individuals, each position is a list [x, y]
    :return np.min(distances): the shortest Euclidean distance from the individuals to the goal
    """
    global goal_position

    # Convert the list to a numpy array
    curr_pos = np.array(curr_pos)

    # Calculate the Euclidean distances
    distances = np.sqrt(np.sum((curr_pos - goal_position)**2, axis=1))

    # Return the shortest distance
    return np.min(distances)


def plot_history(generations, best_paths):
    fig, ax = plt.subplots()
    ax.plot(np.arange(generations), best_paths)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Best distance')
    ax.set_title('Hard maze with novelty')
    plt.savefig('results/hard maze novelty history')
    plt.show()


if __name__ == '__main__':

    # * Generate initial population
    possible_moves = ['R', 'L', 'M']
    num_individuals = 500   # population size
    sequence_length = 100
    num_generations = 1000
    num_of_crossovers = int(num_individuals * 0.1)
    num_of_remainings = int(num_individuals * 0.8)
    coordinates_accumulator = list()
    best_path_accumulator = list()
    file_saved = True

    individuals = [[random.choice(possible_moves) for _ in range(sequence_length)] for _ in range(num_individuals)]
    population = np.array(individuals)
    print("Initial population generated:", population.shape)

    if file_saved:
        accumulated_coordinates = np.load('hard_coordinates_accumulator_novelty.npy')
        accumulated_best_path = np.load('hard_best_path_novelty.npy')
        last_gen_coordinates = accumulated_coordinates[-1]
        plot_agent_positions(last_gen_coordinates)
        plot_history(num_generations, accumulated_best_path)
    else:
        for gen in range(num_generations):
            new_population = list()

            print(f"\n ---------------- Generation: {gen} ----------------")
            print(f"\n* Preprocessing population ...")
            population = preprocess_individuals(population)

            print(f"\n* Finding coordinates for each individual ...")
            population_coordinates = find_coordinates(population)
            coordinates_accumulator.append(population_coordinates)

            # print(f"\n* Plot of final positions ...")
            # plot_agent_positions(population_coordinates)

            best_path = shortest_distance_to_goal(population_coordinates)
            print("\nThe best route so far (shortest distance to the goal):", best_path)
            best_path_accumulator.append(best_path)

            # Calculate fitness
            individuals_fitnesses = sparseness(population_coordinates)

            # 20% crossover
            for _ in range(num_of_crossovers):
                indi1, indi2 = crossover(population[pick_mate_index(individuals_fitnesses)], population[pick_mate_index(individuals_fitnesses)])
                new_population.append(indi1)
                new_population.append(indi2)

            # 80% remainings
            top80_remainings_indices = np.argpartition(individuals_fitnesses, -num_of_remainings)[-num_of_remainings:]
            for idx in top80_remainings_indices:
                new_population.append(population[idx])

            # mutation
            mutated_new_population = mutate(new_population)

            # move to next generation
            population = copy.deepcopy(mutated_new_population)

        coordinates_accumulator = np.array(coordinates_accumulator)
        np.save('hard_coordinates_accumulator_novelty.npy', coordinates_accumulator)
        np.save('hard_best_path_novelty.npy', best_path_accumulator)









