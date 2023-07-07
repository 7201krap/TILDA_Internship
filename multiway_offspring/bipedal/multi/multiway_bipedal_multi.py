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
import warnings

from torch.optim import Adam
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool


warnings.filterwarnings("ignore")
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Define global variables here
POPULATION_SIZE = 300
GENERATIONS = 200
ELITISM = int(POPULATION_SIZE * 0.2)
INPUT_DIM = 24
OUTPUT_DIM = 4
MAX_EP = 1
ENVIRONMENT = gym.make("BipedalWalker-v3")
WEIGHT_CLIP = 0.5
MUTATION_RATE = 0.5

# Seed value
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fitness history
FITNESS_HISTORY_AVG = list()
FITNESS_HISTORY_MAX = list()
FITNESS_HISTORY_MIN = list()
FITNESS_STDERROR_HISTORY = list()



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.tanh(self.l3(a))
        return a

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

    def act(self, state):
        print(f"Performing an optimal action for state: {state}")
        q_values = self.forward(torch.from_numpy(state)).detach().numpy()
        return q_values


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        return self.l3(q)


class DDPG(object):
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer):
        self.actor = actor.to(device)
        self.actor_target = actor.to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = actor_optimizer

        self.critic = critic.to(device)
        self.critic_target = critic.to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = critic_optimizer

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.1):
        for it in range(iterations):
            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.Tensor(x).to(device)
            action = torch.Tensor(u).to(device)
            next_state = torch.Tensor(y).to(device)
            done = torch.Tensor(1 - d).to(device)
            reward = torch.Tensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind:
            state, action, next_state, reward, done = self.storage[i]
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            next_states.append(np.array(next_state, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)


def calculate_fitness(network, env, num_episodes):
    total_rewards = 0
    for _ in range(num_episodes):
        reward = run_episode(network, env)
        total_rewards += reward
    avg_reward = total_rewards / num_episodes
    return avg_reward

def run_episode(network, env):
    state = env.reset(seed=seed_value)
    total_reward = 0.0
    done = False
    while not done:
        action = network(torch.from_numpy(state)).detach().numpy()
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward

def select_survivors(population, fitnesses, elitism):
    sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_population[:elitism]


def tournament_selection(population, fitnesses):
    selected_indices = np.random.randint(len(population), size=5)
    selected_fitnesses = [fitnesses[i] for i in selected_indices]
    winner_index = selected_indices[np.argmax(selected_fitnesses)]
    return population[winner_index]


# SM-G-SO Mutation
def sm_g_so(network, weight_clip=0.2, mr=0):
    network = copy.deepcopy(network)
    env = ENVIRONMENT
    for episode in range(1):
        state = env.reset(seed=seed_value)
        done = False
        current_output = None
        while not done:
            if episode == 0 and current_output is None:
                prev_output = torch.Tensor([0, 0, 0, 0])
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
            scaling = torch.sqrt(torch.abs(sensitivity)).detach()
            mask = torch.rand(delta.shape) > (1 - mr)
            perturbation = torch.clamp(delta / (scaling + 1e-10), -weight_clip, weight_clip)
            perturbation[mask] = torch.clamp(delta[mask] / (scaling[mask] + 1e-10), -weight_clip, weight_clip)
            new_param = current_param + perturbation
            network.inject_parameters(new_param.detach().numpy())
            action = network(torch.from_numpy(state)).detach().numpy()
            state, reward, done, _ = env.step(action)
    return network

# SM-G-SUM Mutation
def sm_g_sum(network, weight_clip=0.2, mr=0):
    network = copy.deepcopy(network)
    tot_size = sum(p.numel() for p in network.parameters())
    env = ENVIRONMENT
    for episode in range(1):
        state = env.reset(seed=seed_value)
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
            mask = torch.rand(delta.shape) > (1 - mr)
            perturbation = torch.clamp(delta / torch.sqrt(((jacobian**2).sum() + 1e-10)), -weight_clip, weight_clip)
            perturbation[mask] = torch.clamp(delta[mask] / torch.sqrt(((jacobian ** 2).sum() + 1e-10)), -weight_clip, weight_clip)
            new_param = current_param + perturbation
            network.inject_parameters(new_param.detach().numpy())
            action = network(torch.from_numpy(state)).detach().numpy()
            state, reward, done, _ = env.step(action)
    return network

# Random mutation
def random_mutation(network, weight_clip, mr=1):
    network = copy.deepcopy(network)
    with torch.no_grad():
        for param in network.parameters():
            if random.random() < mr:
                delta = torch.randn_like(param)
                delta = torch.clamp(delta, -weight_clip, weight_clip)
                param.add_(1 * delta)
    return network


def ddpg_mutation(network, weight_clip=None, mr=None):
    network = copy.deepcopy(network)
    env = ENVIRONMENT
    replay_buffer = ReplayBuffer(100000)
    ddpg.actor = network  # copy your actor network to DDPG's actor

    for episode in range(10):
        state = env.reset(seed=0)
        done = False
        while not done:
            action = ddpg.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, next_state, reward, float(done)))
            state = next_state
            ddpg.train(replay_buffer, iterations=10)

    return ddpg.actor  # return updated actor network


def visualize_best_individual(network):
    env = ENVIRONMENT
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


def calculate_fitness_wrapper(net):
    return calculate_fitness(net, ENVIRONMENT, MAX_EP)


def mutate_offspring_wrapper(net):
    mutations = [sm_g_so, sm_g_sum, random_mutation]
    mutation_choice = mutations[random.randint(0, len(mutations) - 1)]
    return mutation_choice(net, WEIGHT_CLIP, MUTATION_RATE)


if __name__ == "__main__":

    TRAINING = True

    if TRAINING:

        start_time = time.time()

        actor = Actor(8, 4)
        critic = Critic(8, 4)

        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.003)

        ddpg = DDPG(actor, critic, actor_optimizer, critic_optimizer)

        best_fitness = -float('inf')
        population = [Actor(INPUT_DIM, OUTPUT_DIM) for _ in range(POPULATION_SIZE)]

        for generation in range(GENERATIONS):
            fitnesses = process_map(calculate_fitness_wrapper, population, max_workers=8, chunksize=1, desc="Fitness Calculation")

            survivors = select_survivors(population, fitnesses, ELITISM)
            offspring = [copy.deepcopy(tournament_selection(population, fitnesses)) for _ in
                         range(POPULATION_SIZE - ELITISM)]

            mutated_offspring = process_map(mutate_offspring_wrapper, offspring, max_workers=8, chunksize=1, desc="Offspring Generation (Mutation)")

            population = survivors + mutated_offspring

            mean_fit = statistics.mean(fitnesses)
            std_fit = statistics.stdev(fitnesses)
            min_fit = np.min(fitnesses)
            max_fit = np.max(fitnesses)

            if max_fit > best_fitness:
                best_fitness = max_fit
                best_model = population[np.argmax(fitnesses)]  # Get the model with the best fitness
                torch.save(best_model.state_dict(), 'best_model_weights_bipedal.pth')
                print(f"[INFO] Saving weights in generation {generation}")

            FITNESS_HISTORY_AVG.append(mean_fit)
            FITNESS_STDERROR_HISTORY.append(std_fit)
            FITNESS_HISTORY_MAX.append(max_fit)
            FITNESS_HISTORY_MIN.append(min_fit)

            print(f"[Generation: {generation}]\n"
                  f"Avg Fitness: {mean_fit}\n"
                  f"Max Fitness: {max_fit}\n"
                  f"Min Fitness: {min_fit}\n"
                  f"Standard Deviation: {std_fit}\n"
                  f"Current Best Fitness: {best_fitness}\n"
                  f"-----------------------------------\n")


        end_time = time.time()
        print(f"Training time: {end_time - start_time} seconds")

        # Plot the fitness history
        plt.figure(figsize=(10, 6))
        plt.plot(FITNESS_HISTORY_AVG, label='Average Fitness', marker='o', linestyle='-')
        plt.plot(FITNESS_HISTORY_MIN, label='Minimum Fitness', color='green', marker='^', linestyle='-')
        plt.plot(FITNESS_HISTORY_MAX, label='Maximum Fitness', color='orange', marker='s', linestyle='-')
        plt.fill_between(list(range(len(FITNESS_HISTORY_AVG))), [x - y for x, y in zip(FITNESS_HISTORY_AVG, FITNESS_STDERROR_HISTORY)], [x + y for x, y in zip(FITNESS_HISTORY_AVG, FITNESS_STDERROR_HISTORY)], color='blue', alpha=.2, label='Standard Deviation')
        plt.title("Fitness History")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid()
        plt.savefig('multiway_bipedal')

    else:
        # model loading
        loaded_model = Actor(INPUT_DIM, OUTPUT_DIM)
        loaded_model.load_state_dict(torch.load('best_model_weights_bipedal.pth'))
        loaded_model.eval()

        visualize_best_individual(loaded_model)


