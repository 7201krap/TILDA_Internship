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

from torch import nn
from copy import deepcopy
from torch.nn import functional as F
from tqdm import tqdm


# In[2]:


# Use the following gym version.
# pip install gym==0.25.0
# pip install pygame

import os

# Set a seed value:
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

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
env = gym.make('LunarLander-v2')
pop_size = 200
gens = 150
elit = int(pop_size * 0.4)
tot_size = 5
MAX_EP = 1


# In[3]:


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def act(self, state):
        print(f"Performing an optimal action for state: {state}")
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = self.forward(state_t.unsqueeze(0))  
        max_q_index = torch.argmax(q_values, dim=1)[0]   
        action = max_q_index.detach().item()   
        return action  


# In[4]:


def calculate_fitness(network, env, num_episodes=MAX_EP):
    total_rewards = 0
    for _ in range(num_episodes):
        reward, _ = run_episode(network, env)
        total_rewards += reward
    avg_reward = total_rewards / num_episodes
    return avg_reward


# In[5]:


def run_episode(network, env):
    state = env.reset(seed=seed_value)
    total_reward = 0.0
    log_probs = []  # Store log probabilities of actions
    done = False
    while not done:
        action = torch.argmax(network(torch.from_numpy(state).float().unsqueeze(0))).item()
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward, _


# In[6]:


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


# In[7]:


# Define genetic algorithm
def main(POPULATION_SIZE, GENERATIONS, ELITISM, TOURNAMENT_SIZE, MUTATION_STRENGTH, MUTATION_RATE):
    
    start_time = time.time()
    
    FITNESS_HISTORY = list()
    FITNESS_STDERROR_HISTORY = list()
    
    # Create initial population
    population = [PolicyNetwork(8, 4) for _ in range(POPULATION_SIZE)]

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


def visualize_best_individual(network, env):
    state = env.reset(seed=seed_value)
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


import sys
import pygame

# ### Version Control

first_run = False

if first_run == True:
    population, history, history_std = main(POPULATION_SIZE=pop_size,
                                GENERATIONS=gens,
                                ELITISM=elit,
                                TOURNAMENT_SIZE=tot_size,
                                MUTATION_STRENGTH=1,
                                MUTATION_RATE=0.01)


    # In[9]:


    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(gens), np.array(history)[:,0], marker='o', linestyle='-', label='Average Fitness')
    plt.plot(np.arange(gens), np.array(history)[:,1], marker='^', linestyle='-', label='Max Fitness')
    plt.plot(np.arange(gens), np.array(history)[:,2], marker='s', linestyle='-', label='Min Fitness')
    plt.fill_between(np.arange(gens), np.array(history)[:,0] - np.array(history_std), np.array(history)[:,0] + np.array(history_std),
                     alpha=0.2, color='blue', label='Standard Error')

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness History of LunarLander Classic Mutation')
    plt.grid()
    plt.legend()
    plt.savefig('../results/lundarlander ga control')
    plt.show()

    # save the best model 
    fitnesses = [calculate_fitness(network, env, MAX_EP) for network in tqdm(population, desc="Calculating fitnesses")]
    population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
    best_network = population[0]
    torch.save(best_network.state_dict(), '../results/lunarlander_gaO_control.pth')

else:
    # load the best model
    # First, you have to create an instance of the same model architecture
    new_network = PolicyNetwork(8, 4)

    # Then you can load the weights
    new_network.load_state_dict(torch.load('../results/lunarlander_gaO_control.pth'))

    visualize_best_individual(new_network, env)

    pygame.quit()
    sys.exit()


# In[ ]:




