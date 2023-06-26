#!/usr/bin/env python
# coding: utf-8
import sys

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import gym
import pygame as pygame
import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
import statistics
import time 

from maze_env import MazeEnv
from torch import nn
from tqdm import tqdm
from scipy.optimize import minimize_scalar

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


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, 36)
        self.fc2 = nn.Linear(36, 72)
        self.fc3 = nn.Linear(72, 36)
        self.fc4 = nn.Linear(36, output_dim)

        self.dropout = nn.Dropout(p=0.2)

        # Apply the weights initialization
        self.apply(self.init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Add dropout layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Add dropout layer
        x = F.relu(self.fc3(x))
        x = self.dropout(x)  # Add dropout layer
        x = self.fc4(x)
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.1)
            nn.init.constant_(m.bias, 0)

    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = self.forward(state_t.reshape(1, int(np.prod(env.observation_space.shape))))
        action_probs = nn.functional.softmax(q_values, dim=1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action


    def inject_parameters(self, pvec):
        new_state_dict = {}
        count = 0
        for name, param in self.named_parameters():
            sz = param.data.numel()
            raw = pvec[count:count + sz]
            reshaped = raw.reshape(param.data.shape)
            new_state_dict[name] = torch.from_numpy(reshaped).float()
            count += sz
        self.load_state_dict(new_state_dict)


# In[3]:


def calculate_fitness(network, env, num_episodes):
    total_rewards = 0
    for _ in range(num_episodes):
        reward, _ = run_episode(network, env)
        total_rewards += reward
    avg_reward = total_rewards / num_episodes
    return avg_reward

def run_episode(network, env):
    state = env.reset()
    total_reward = 0.0
    log_probs = []  # Store log probabilities of actions
    done = False
    while not done:
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = network(state_t.reshape(1, int(np.prod(env.observation_space.shape))))
        action_probs = nn.functional.softmax(q_values, dim=1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        log_probs.append(log_prob)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward, log_probs

def select_survivors(population, fitnesses, ELITISM):
    sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_population[:ELITISM]

def tournament_selection(population, fitnesses, tournament_size):
    selected_indices = np.random.randint(len(population), size=tournament_size)
    selected_fitnesses = [fitnesses[i] for i in selected_indices]
    winner_index = selected_indices[np.argmax(selected_fitnesses)]
    return population[winner_index]

def perturb_parameters(network, weight_clip, n_episodes):

    for episode in range(n_episodes):
        # Reset the environment
        state = env.reset()
        done = False
        current_output = None

        while not done:

            if episode == 0 and current_output is None:
                prev_output = torch.Tensor([0.25, 0.25, 0.25, 0.25])
            else:
                prev_output = current_output.detach()

            # Get the original parameters
            current_param = torch.cat([param.view(-1) for param in network.parameters()])

            # Perturb the model parameters
            delta = torch.randn_like(current_param)

            # Forward pass to calculate the output
            current_output = network(torch.from_numpy(state).reshape(1, int(np.prod(env.observation_space.shape))))

            # Calculate the error
            error = ((current_output - prev_output)**2).mean()

            # Clear the gradients from the last backward pass
            network.zero_grad()

            # Backward pass to calculate the gradients
            error.backward()

            # Extract the gradients
            gradient = torch.cat([param.grad.view(-1) for param in network.parameters()])

            # Generate a random mask with 10% True values
            mask = torch.rand(gradient.shape) < 0.5

            # Apply the operation only to the masked elements
            gradient[mask] = gradient[mask] / torch.sqrt((gradient[mask] ** 2).sum() + 1e-10)

            # Calculate the new parameters
            perturbation = np.clip(delta * gradient, -weight_clip, weight_clip)
            new_param = current_param + perturbation

            # Inject the new parameters into the model
            network.inject_parameters(new_param.detach().numpy())

            state_t = torch.as_tensor(state, dtype=torch.float32)
            q_values = network.forward(state_t.reshape(1, int(np.prod(env.observation_space.shape))))
            action_probs = nn.functional.softmax(q_values, dim=1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            state, reward, done, _ = env.step(action)


def visualize_best_individual(network, env):
    state = env.reset()
    done = False
    total_reward = 0
    count = 0
    while not done:
        print(f"Alive for {count} actions")
        env.render()  # Render the environment to the screen
        action = network.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        count = count + 1
    print(f"Total reward: {total_reward}")
    env.close()


# Constants
env = MazeEnv()

POPULATION_SIZE = 100
GENERATIONS = 50
ELITISM = int(POPULATION_SIZE * 0.4)
TOURNAMENT_SIZE = 5
WEIGHT_CLIP = 0.2
INPUT_DIM = int(np.prod(env.observation_space.shape))
OUTPUT_DIM = env.action_space.n
MAX_EP = 1


FITNESS_HISTORY = list()
FITNESS_STDERROR_HISTORY = list()

# Creating initial population of networks
population = [PolicyNetwork(INPUT_DIM, OUTPUT_DIM) for _ in range(POPULATION_SIZE)]

first_run = False

if first_run == True:

    start_time = time.time()

    # Generations loop
    for generation in range(GENERATIONS):
        print(f"[Generation {generation}]")

        # Calculate fitness for each network
        print("Calculating Fitnesses For Population ...")
        fitnesses = [calculate_fitness(network, env, MAX_EP) for network in tqdm(population)]

        avg_fitness = np.average(fitnesses)
        max_fitness = np.max(fitnesses)
        min_fitness = np.min(fitnesses)
        FITNESS_HISTORY.append([avg_fitness, max_fitness, min_fitness])

        # std error
        standard_deviation = statistics.stdev(fitnesses)
        standard_error = standard_deviation / (POPULATION_SIZE ** 0.5)
        FITNESS_STDERROR_HISTORY.append(standard_error)

        print(f"Average Fitness: {avg_fitness} \n Best Fitness: {max_fitness} \n Worst Fitness: {min_fitness} \n Standard Error: {standard_error}")

        # Select the best networks to pass their genes to the next generation
        survivors = select_survivors(population, fitnesses, ELITISM)

        # Create the next generation
        next_population = survivors  # Start with the survivors

        print("Generating Offsprings ...")
        # Add offspring by tournament selection and mutation
        for _ in tqdm(range(POPULATION_SIZE - len(survivors))):
            parent = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            offspring = copy.deepcopy(parent)

            # Perturb the offspring using your approach
            perturb_parameters(offspring, WEIGHT_CLIP, MAX_EP)

            next_population.append(offspring)

        # The next generation becomes the current population
        population = next_population

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(GENERATIONS), np.array(FITNESS_HISTORY)[:,0], marker='o', linestyle='-', label='Average Fitness')
    plt.plot(np.arange(GENERATIONS), np.array(FITNESS_HISTORY)[:,1], marker='^', linestyle='-', label='Max Fitness')
    plt.plot(np.arange(GENERATIONS), np.array(FITNESS_HISTORY)[:,2], marker='s', linestyle='-', label='Min Fitness')
    plt.fill_between(np.arange(GENERATIONS), np.array(FITNESS_HISTORY)[:,0] - np.array(FITNESS_STDERROR_HISTORY), np.array(FITNESS_HISTORY)[:,0] + np.array(FITNESS_STDERROR_HISTORY),
                     alpha=0.2, color='blue', label='Standard Error')

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness History of Maze SM-G-FO Mutation')
    plt.grid()
    plt.legend()
    plt.savefig('../results/Maze ga sm-g-fo')
    plt.show()

    # save the best model
    fitnesses = [calculate_fitness(network, env, MAX_EP) for network in tqdm(population, desc="Calculating fitnesses")]
    torch.save(population, '../results/maze_gaO_smg-fo.pth')

else:
    seed_setter(seed_value=0)

    loaded_population = torch.load('../results/maze_gaO_smg-fo.pth')

    fitnesses = [calculate_fitness(network, env, MAX_EP) for network in
                 tqdm(loaded_population, desc="Calculating fitnesses")]
    population = [x for _, x in sorted(zip(fitnesses, loaded_population), key=lambda pair: pair[0], reverse=True)]
    best_network = population[0]

    visualize_best_individual(best_network, env)
    pygame.quit()
    sys.exit()

