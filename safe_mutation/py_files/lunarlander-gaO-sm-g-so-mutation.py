#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import gym
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

from torch import nn
from tqdm import tqdm

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

env = gym.make("LunarLander-v2")

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)  

        # Apply the weights initialization
        self.apply(self.init_weights)

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

    def init_weights(self, m):
        if type(m) == nn.Linear:
            init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

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
    state = env.reset(seed=seed_value)
    total_reward = 0.0
    done = False
    while not done:
        action = torch.argmax(network(torch.from_numpy(state).float().unsqueeze(0))).item()
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward, _

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
        state = env.reset(seed=seed_value)
        done = False
        total_reward = 0
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
            direction = delta / torch.sqrt((delta ** 2).sum())
            direction_t = direction.detach()

            # Forward pass to calculate the output
            current_output = network(torch.from_numpy(state).float().unsqueeze(0))

            # Calculate the error
            error = ((current_output - prev_output) ** 2).mean()

            # Clear the gradients from the last backward pass
            network.zero_grad()

            # Backward pass to calculate the gradients
            # Set create_graph=True to allow higher order derivative
            error.backward(create_graph=True)

            # Extract the gradients
            gradient = torch.cat([param.grad.view(-1) for param in network.parameters()])

            # Calculate the gradient vector product
            grad_v_prod = (gradient * direction_t).sum()

            second_order_grad = torch.autograd.grad(grad_v_prod, network.parameters())
            second_order_grad = torch.cat([grad.view(-1) for grad in second_order_grad])

            sensitivity = second_order_grad
            scaling = torch.sqrt(torch.abs(sensitivity))

            # Normalize the gradients
            so_gradient = (gradient / (scaling + 1e-10)).detach()

            # Calculate the new parameters
            perturbation = np.clip(delta * so_gradient, -weight_clip, weight_clip)
            new_param = current_param + perturbation

            # Inject the new parameters into the model
            network.inject_parameters(new_param.detach().numpy())

            action = torch.argmax(network(torch.from_numpy(state).float().unsqueeze(0))).item()
            state, reward, done, _ = env.step(action)


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


# Constants
POPULATION_SIZE = 200 
GENERATIONS = 300
ELITISM = int(POPULATION_SIZE * 0.4)
TOURNAMENT_SIZE = 5
WEIGHT_CLIP = 0.2
INPUT_DIM = 8  # For LunarLander environment
OUTPUT_DIM = 4  # For LunarLander environment
MAX_EP = 1

FITNESS_HISTORY = list()
FITNESS_STDERROR_HISTORY = list()

population = [PolicyNetwork(INPUT_DIM, OUTPUT_DIM) for _ in range(POPULATION_SIZE)]

first_run = True

if first_run:
    start_time = time.time()

    for generation in range(GENERATIONS):
        print(f"[Generation {generation}]")
        fitnesses = [calculate_fitness(network, env, MAX_EP) for network in tqdm(population)]
        avg_fitness = np.average(fitnesses)
        max_fitness = np.max(fitnesses)
        min_fitness = np.min(fitnesses)
        FITNESS_HISTORY.append([avg_fitness, max_fitness, min_fitness])
        standard_deviation = statistics.stdev(fitnesses)
        standard_error = standard_deviation / (POPULATION_SIZE ** 0.5)
        FITNESS_STDERROR_HISTORY.append(standard_error)
        print(f"Average Fitness: {avg_fitness} \n Best Fitness: {max_fitness} \n Worst Fitness: {min_fitness} \n Standard Error: {standard_error}")

        survivors = select_survivors(population, fitnesses, ELITISM)

        next_population = survivors

        for _ in tqdm(range(POPULATION_SIZE - len(survivors))):
            parent = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            offspring = copy.deepcopy(parent)
            perturb_parameters(offspring, WEIGHT_CLIP, MAX_EP)
            next_population.append(offspring)

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
    plt.title('Fitness History of LunarLander SM-G-SO Mutation')
    plt.grid()
    plt.legend()
    plt.ylim(top=500)
    plt.savefig('../results/lunarlander ga sm-g-so')
    plt.show()

    # save the best model
    fitnesses = [calculate_fitness(network, env, MAX_EP) for network in tqdm(population, desc="Calculating fitnesses")]
    population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
    best_network = population[0]
    torch.save(best_network.state_dict(), '../results/lunarlander_gaO_smg-so.pth')

else:
    # load the best model
    # First, you have to create an instance of the same model architecture
    new_network = PolicyNetwork(INPUT_DIM, OUTPUT_DIM)

    # Then you can load the weights
    new_network.load_state_dict(torch.load('../results/lunarlander_gaO_smg-so.pth'))

    visualize_best_individual(new_network, env)

