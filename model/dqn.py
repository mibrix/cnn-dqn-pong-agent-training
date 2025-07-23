import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward, done)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward, done)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.convert_action = {0:2,1:3} # Converts actions to fit Pong

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Initialize steps counter
        self.steps_done = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully-connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self, observation, exploit=False):
        # Calculate current epsilon
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                  max(0, (self.anneal_length - self.steps_done) / self.anneal_length)
        if exploit:
            epsilon = self.eps_end

        self.steps_done += 1

        if random.random() > epsilon:
            with torch.no_grad():

                q_values = self.forward(observation)
                action = torch.argmax(q_values, dim=-1).view(-1, 1)
        else:
            action = torch.randint(0, self.n_actions, (observation.size(0), 1), device=observation.device)

        return action


def optimize(dqn, target_dqn, memory, optimizer):
    if len(memory) < dqn.batch_size:
        return  # Ensure sufficient samples are available

    batch = memory.sample(dqn.batch_size)
    observations, actions, next_observations, rewards, done_batch = batch

    # Check and adjust observations shape
    observations = torch.stack(observations).to(device).squeeze(1)  # Removing any unwanted middle dimension
    actions = torch.stack(actions).to(device).long().view(-1, 1)
    next_observations = torch.stack(next_observations).to(device).squeeze(1)  # Same adjustment for next_observations
    rewards = torch.stack(rewards).to(device)
    done_batch = torch.stack(done_batch).to(device)


     # Convert done_batch from a boolean tensor to a float tensor for arithmetic operations
    done_batch = done_batch.float()

    q_values = dqn(observations).gather(1, actions)
    next_q_values = target_dqn(next_observations).max(1)[0].detach().unsqueeze(1)

    # Compute the Q-value targets, handling terminal states
    q_value_targets = rewards + (1 - done_batch) * dqn.gamma * next_q_values

    loss = F.mse_loss(q_values, q_value_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()




