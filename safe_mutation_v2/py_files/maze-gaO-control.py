#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import statistics
import time
import torch.nn.functional as F

from maze_env import MazeEnv
from torch import nn
from copy import deepcopy
from tqdm import tqdm

# In[2]:


# Use the following gym version.
# pip install gym==0.25.0
# pip install pygame

import os


def seed_setter(seed_value=0):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `pytorch` pseudo-random generator at a fixed value
    torch.manual_seed(seed_value)

    # 6. If you have cuDNN library installed, you should also set its random generator at a fixed value:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# define global variable
env = MazeEnv()
pop_size = 100
gens = 50
elit = int(pop_size * 0.4)
tot_size = 5
MAX_EP = 1


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=1, output_dim=4):
        super(PolicyNetwork, self).__init__()

        self.hidden_dim_lstm = 128
        self.hidden_dim_fffn = 256
        self.hidden_dim_fffn2 = 64


        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, self.hidden_dim_lstm, num_layers=1, batch_first=True)

        # Define the FFFN layers
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim_lstm, self.hidden_dim_fffn),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_fffn, self.hidden_dim_fffn2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_fffn2, output_dim)
        )

        # Initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        # Initial hidden state for LSTM
        h0 = torch.zeros(1, 1, self.hidden_dim_lstm)
        c0 = torch.zeros(1, 1, self.hidden_dim_lstm)

        # LSTM layer
        out, _ = self.lstm(x.view(1, -1, 1), (h0, c0))

        # Taking the last output of the LSTM
        out = out[:, -1, :]

        # FFFN layer
        out = self.ffn(out)

        return out

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.1)
            nn.init.constant_(m.bias, 0)

    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = self.forward(state_t)
        action_probs = nn.functional.softmax(q_values, dim=1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action


def calculate_fitness(network, env, num_episodes=MAX_EP):
    total_rewards = 0
    for _ in range(num_episodes):
        reward, _ = run_episode(network, env)
        total_rewards += reward
    return total_rewards


def run_episode(network, env):
    state = env.reset()
    total_reward = 0.0
    log_probs = []  # Store log probabilities of actions
    done = False
    while not done:
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = network(state_t)
        action_probs = nn.functional.softmax(q_values, dim=1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        log_probs.append(log_prob)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward, log_probs


def mutate_and_tournament(population, tournament_size, mutation_rate, mutation_strength):
    # Select individuals for the tournament
    individuals = random.sample(population, tournament_size)
    # Calculate fitness for each individual
    fitnesses = [calculate_fitness(individual, env) for individual in individuals]
    # Select the best individual
    parent = individuals[np.argmax(fitnesses)]
    # Create offspring by deep copying the parent
    offspring = deepcopy(parent)

    # Apply mutation
    with torch.no_grad():
        for param in offspring.parameters():
            if random.random() < mutation_rate:
                delta = torch.randn_like(param)
                param.add_(mutation_strength * delta)

    # Return the mutated offspring
    return offspring


def main(POPULATION_SIZE, GENERATIONS, ELITISM, TOURNAMENT_SIZE, MUTATION_STRENGTH, MUTATION_RATE):
    start_time = time.time()

    FITNESS_HISTORY = list()
    FITNESS_STDERROR_HISTORY = list()

    # Create initial population
    population = [PolicyNetwork(1, 4) for _ in range(POPULATION_SIZE)]

    for generation in range(1, GENERATIONS + 1):

        # Calculate fitness for each network
        fitnesses = [calculate_fitness(network, env) for network in tqdm(population, desc="Calculating fitnesses")]

        # average fitness
        avg_fitness = np.average(fitnesses)
        max_fitness = np.max(fitnesses)
        min_fitness = np.min(fitnesses)
        FITNESS_HISTORY.append([avg_fitness, max_fitness, min_fitness])

        # std error
        standard_deviation = statistics.stdev(fitnesses)
        standard_error = standard_deviation / (POPULATION_SIZE ** 0.5)
        FITNESS_STDERROR_HISTORY.append(standard_error)

        print(f"[Generation: {generation}] \n Average Fitness: {avg_fitness} \n Best Fitness: {max_fitness} \n Worst Fitness: {min_fitness} \n Standard Error: {standard_error}")

        # Sort population by fitness
        population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]

        # Select the best networks to pass their genes to the next generation
        survivors = population[:ELITISM]

        # Create the next generation
        next_population = survivors  # Start with the survivors

        num_individuals_to_add = POPULATION_SIZE - len(next_population)
        # Add offspring by tournament selection and mutation
        for _ in tqdm(range(num_individuals_to_add), desc="Generating Offspring"):
            offspring = mutate_and_tournament(population, TOURNAMENT_SIZE, MUTATION_RATE, MUTATION_STRENGTH)
            next_population.append(offspring)

        # The next generation becomes the current population
        population = next_population

    print("--- %s seconds ---" % (time.time() - start_time))

    return population, FITNESS_HISTORY, FITNESS_STDERROR_HISTORY


import sys
import pygame

def visualize_best_individual(network, env):
    state = env.reset()
    done = False
    total_reward = 0
    count = 0
    while not done:
        print(f"Alive for {count} actions with total reward of {total_reward}")
        env.render()  # Render the environment to the screen
        action = network.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        count = count + 1
    print(f"Total reward: {total_reward}")
    env.close()

first_run = True

if first_run == True:

    seed_setter(seed_value=0)

    env = MazeEnv()
    # Run the genetic algorithm
    population, history, history_std = main(POPULATION_SIZE=pop_size,
                                            GENERATIONS=gens,
                                            ELITISM=elit,
                                            TOURNAMENT_SIZE=tot_size,
                                            MUTATION_STRENGTH=1,
                                            MUTATION_RATE=0.5)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(gens), np.array(history)[:, 0], marker='o', linestyle='-', label='Average Fitness')
    plt.plot(np.arange(gens), np.array(history)[:, 1], marker='^', linestyle='-', label='Max Fitness')
    plt.plot(np.arange(gens), np.array(history)[:, 2], marker='s', linestyle='-', label='Min Fitness')
    plt.fill_between(np.arange(gens), np.array(history)[:, 0] - np.array(history_std),
                     np.array(history)[:, 0] + np.array(history_std),
                     alpha=0.2, color='blue', label='Standard Error')
    plt.axhline(y=100, color='r', linestyle='-')

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness History of Maze Classic Mutation')
    plt.grid()
    plt.legend()
    plt.savefig('../results/maze ga control')
    plt.show()

    # save the best model
    fitnesses = [calculate_fitness(network, env, MAX_EP) for network in tqdm(population, desc="Calculating fitnesses")]
    torch.save(population, '../results/maze_gaO_control.pth')

else:
    seed_setter(seed_value=0)

    loaded_population = torch.load('../results/maze_gaO_control.pth')

    fitnesses = [calculate_fitness(network, env, MAX_EP) for network in tqdm(loaded_population, desc="Calculating fitnesses")]
    population = [x for _, x in sorted(zip(fitnesses, loaded_population), key=lambda pair: pair[0], reverse=True)]
    best_network = population[0]

    visualize_best_individual(best_network, env)
    pygame.quit()
    sys.exit()
