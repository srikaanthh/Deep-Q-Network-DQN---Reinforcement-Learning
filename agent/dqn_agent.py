import numpy as np
import random
from collections import namedtuple, deque
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_layer_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_layer_size)
        self.fc2 = nn.Linear(fc_layer_size, fc_layer_size)
        self.fc3 = nn.Linear(fc_layer_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size,
                 action_size, seed, fc_layer_size,
                 tau, gamma, update_every, lr, batch_size, buffer_size,
                 type_dqn):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # Agent hypeparameters

        self.TAU = tau
        self.GAMMA = gamma
        self.UPDATE_EVERY = update_every
        self.LR = lr
        self.type_dqn = type_dqn

        # Memory params
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        fc_layer_size = fc_layer_size
        self.qnetwork_online = QNetwork(state_size, action_size, seed, fc_layer_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc_layer_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_online.parameters(), lr=self.LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_online.eval()
        with torch.no_grad():
            action_values = self.qnetwork_online(state)

        self.qnetwork_online.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    @staticmethod
    def greedy_action_selection(states, online_network):
        _, actions = online_network(states).max(dim=1, keepdim=True)
        return actions

    @staticmethod
    def action_evaluation(states, actions, rewards, done, gamma, q_network):
        next_q_values = q_network(states).gather(dim=1, index=actions)
        q_values = rewards + (gamma * next_q_values * (1 - done))
        return q_values

    def learn_dnn(self, states, **kwarks):
        """

        :param states: States
        :param kwarks: Actions, rewards, done, gamma and qnetwork
        :return:
        """

        if self.type_dqn == 'double_dqn':

            actions = self.greedy_action_selection(states=states, online_network=self.qnetwork_online)
            q_values_target = self.action_evaluation(states, actions, kwarks['rewards'],
                                                     kwarks['done'],
                                                     kwarks['gamma'],
                                                     q_network=self.qnetwork_target)
        elif self.type_dqn == 'vanilla_dqn':
            actions = self.greedy_action_selection(states=states, online_network=self.qnetwork_target)
            q_values_target = self.action_evaluation(states, actions, kwarks['rewards'],
                                                     kwarks['done'],
                                                     kwarks['gamma'],
                                                     q_network=self.qnetwork_target)

        else:
            raise Exception("type_dqn not implemented")
        return q_values_target

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Compute Q targets for current states
        Q_targets = self.learn_dnn(next_states,
                                   actions=actions,
                                   rewards=rewards,
                                   done=dones,
                                   gamma=gamma)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_online(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_online, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        tau = self.TAU
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

##
