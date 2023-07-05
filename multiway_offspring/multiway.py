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
import torch.optim as optim
from torch.optim import Adam

# Seed value
seed_value = 0

# Set seed
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Constants
POPULATION_SIZE = 50
GENERATIONS = 300
ELITISM = int(POPULATION_SIZE * 0.2)  # Top 20%
WEIGHT_CLIP = 0.2
INPUT_DIM = 8  # For LunarLander
OUTPUT_DIM = 4  # For LunarLander
MAX_EP = 1
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.01

# Fitness history
FITNESS_HISTORY = list()
FITNESS_STDERROR_HISTORY = list()

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

def calculate_fitness(network, env, num_episodes):
    total_rewards = 0
    for _ in range(num_episodes):
        reward = run_episode(network, env)
        total_rewards += reward
    avg_reward = total_rewards / num_episodes
    return avg_reward

def run_episode(network, env):
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = torch.argmax(network(torch.from_numpy(state).float().unsqueeze(0))).item()
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward

def select_survivors(population, fitnesses, elitism):
    sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_population[:elitism]

# SM-G-SO Mutation
def sm_g_so(network):
    network = copy.deepcopy(network)
    env = gym.make("LunarLander-v2")
    for episode in range(1):
        state = env.reset()
        done = False
        current_output = None
        while not done:
            if episode == 0 and current_output is None:
                prev_output = torch.Tensor([0.25, 0.25, 0.25, 0.25])
            else:
                prev_output = current_output.detach()
            current_param = torch.cat([param.view(-1) for param in network.parameters()])
            delta = torch.randn_like(current_param)
            direction = delta / torch.sqrt((delta ** 2).sum())
            direction_t = direction.detach()
            current_output = network(torch.from_numpy(state).float().unsqueeze(0))
            error = ((current_output - prev_output) ** 2).mean()
            network.zero_grad()
            error.backward(create_graph=True)
            gradient = torch.cat([param.grad.view(-1) for param in network.parameters()])
            grad_v_prod = (gradient * direction_t).sum()
            second_order_grad = torch.autograd.grad(grad_v_prod, network.parameters())
            second_order_grad = torch.cat([grad.view(-1) for grad in second_order_grad])
            sensitivity = second_order_grad
            scaling = torch.sqrt(torch.abs(sensitivity))
            mask = torch.rand(delta.shape) > 0
            perturbation = torch.clamp(delta / scaling, -0.2, 0.2)
            perturbation[mask] = torch.clamp(delta[mask] / scaling[mask], -0.2, 0.2)
            new_param = current_param + perturbation
            network.inject_parameters(new_param.detach().numpy())
            action = torch.argmax(network(torch.from_numpy(state).float().unsqueeze(0))).item()
            state, reward, done, _ = env.step(action)
    return network

# SM-G-SUM Mutation
def sm_g_sum(network):
    network = copy.deepcopy(network)
    tot_size = sum(p.numel() for p in network.parameters())
    env = gym.make("LunarLander-v2")
    for episode in range(1):
        state = env.reset()
        done = False
        while not done:
            current_param = torch.cat([param.view(-1) for param in network.parameters()])
            delta = torch.randn_like(current_param)
            current_output = network(torch.from_numpy(state).float().unsqueeze(0))
            num_outputs = current_output.shape[-1]
            jacobian = torch.zeros(num_outputs, tot_size)
            grad_output = torch.zeros(*current_output.shape)
            for i in range(num_outputs):
                network.zero_grad()
                grad_output.zero_()
                grad_output[:, i] = 1.0
                current_output.backward(grad_output, retain_graph=True)
                jacobian[i] = torch.cat([param.grad.view(-1) for param in network.parameters()])
            mask = torch.rand(delta.shape) > 0
            perturbation = torch.clamp(delta / torch.sqrt(((jacobian**2).sum() + 1e-10)), -0.2, 0.2)
            perturbation[mask] = torch.clamp(delta[mask] / torch.sqrt(((jacobian ** 2).sum() + 1e-10)), -0.2, 0.2)
            new_param = current_param + perturbation
            network.inject_parameters(new_param.detach().numpy())
            action = torch.argmax(network(torch.from_numpy(state).float().unsqueeze(0))).item()
            state, reward, done, _ = env.step(action)
    return network

# Random mutation
def random_mutation(network):
    network = copy.deepcopy(network)
    with torch.no_grad():
        for param in network.parameters():
            if random.random() < 0.01:
                delta = torch.randn_like(param)
                param.add_(1 * delta)
    return network


def adam_mutation(network):
    network = copy.deepcopy(network)
    env = gym.make("LunarLander-v2")
    optimizer = optim.Adam(network.parameters())

    for episode in range(1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        done = False

        while not done:
            current_output = network(torch.from_numpy(state).float().unsqueeze(0))
            action_probabilities = F.softmax(current_output, dim=1) + 1e-10
            action_distribution = torch.distributions.Categorical(action_probabilities)
            action = action_distribution.sample()
            log_prob = action_distribution.log_prob(action)
            saved_log_probs.append(log_prob)

            state, reward, done, _ = env.step(action.item())
            rewards.append(reward)

        # Calculate cumulative rewards
        cumulative_rewards = []
        cum_reward = 0
        for reward in reversed(rewards):
            cum_reward = reward + 0.01 * cum_reward
            cumulative_rewards.insert(0, cum_reward)

        cumulative_rewards = torch.tensor(cumulative_rewards)
        cumulative_rewards = (cumulative_rewards - cumulative_rewards.mean()) / (
                    cumulative_rewards.std() + 1e-10)

        # Policy gradient update
        loss = []
        for log_prob, reward in zip(saved_log_probs, cumulative_rewards):
            loss.append(-log_prob * reward)
        loss = torch.cat(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1) # Add this line for gradient clipping
        optimizer.step()
    return network

def genetic_algorithm():
    env = gym.make("LunarLander-v2")
    population = [PolicyNetwork(INPUT_DIM, OUTPUT_DIM) for _ in range(POPULATION_SIZE)]
    mutations = [sm_g_so, sm_g_sum, random_mutation, adam_mutation]
    for generation in range(GENERATIONS):
        fitnesses = [calculate_fitness(net, env, MAX_EP) for net in population]
        survivors = select_survivors(population, fitnesses, ELITISM)
        offspring = [copy.deepcopy(random.choice(survivors)) for _ in range(POPULATION_SIZE - ELITISM)]
        mutated_offspring = [mutations[random.randint(0, len(mutations)-1)](net) for net in offspring]
        population = survivors + mutated_offspring
        FITNESS_HISTORY.append(statistics.mean(fitnesses))
        FITNESS_STDERROR_HISTORY.append(statistics.stdev(fitnesses))
        print(f"Generation: {generation}, Avg Fitness: {statistics.mean(fitnesses)}, Max Fitness: {np.max(fitnesses)}")
    return population

if __name__ == "__main__":
    start_time = time.time()
    population = genetic_algorithm()
    end_time = time.time()
    print("Training time: ", end_time - start_time)
    # Plot the fitness history
    plt.figure()
    plt.plot(FITNESS_HISTORY)
    plt.fill_between(list(range(len(FITNESS_HISTORY))), [x - y for x, y in zip(FITNESS_HISTORY, FITNESS_STDERROR_HISTORY)], [x + y for x, y in zip(FITNESS_HISTORY, FITNESS_STDERROR_HISTORY)], color='b', alpha=.1)
    plt.title("Fitness History")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()
