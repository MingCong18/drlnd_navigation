import numpy as np
import random

from model import QNetwork, DuelingDQN
from replay_buffer import ReplayBuffer, PrioReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, args, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            args (object): parameters
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.ddqn = args.ddqn
        self.dueling = args.dueling
        self.prio = args.prio
        self.seed = random.seed(seed)

        if self.ddqn:
            print("Double DQN Enabled!")
        else:
            print("Double DQN Not Enabled!")

        if self.dueling: # when dueling is enable we initialize dueling DQN
            print("Dueling DQN Enabled!")
            self.qnetwork_local = DuelingDQN(state_size, action_size, seed).to(device)
            self.qnetwork_target = DuelingDQN(state_size, action_size, seed).to(device)
        else: # else we initialize the basic DQN
            print("Dueling DQN Not Enabled!")
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if self.prio:
            print("Prioritized Replay Buffer Enabled!")
            self.memory = PrioReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        else:
            print("Prioritized Replay Buffer Not Enabled!")
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                self.learn(GAMMA)

    def learn(self, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.prio:
            (states, actions, rewards, next_states, dones), batch_indices, batch_weights = self.memory.sample()
            batch_weights_v = torch.from_numpy(np.vstack(batch_weights)).float().to(device)
        else:
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # When double DQN is enabled, we calculate the best action to take in the next state
        # using our local trained network, but values corresponding to this action come
        # from the target network
        if self.ddqn:
            next_state_actions = self.qnetwork_local(next_states).max(1)[1]
            Q_targets_next = self.qnetwork_target(next_states).detach().\
                gather(1, next_state_actions.unsqueeze(-1))
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        # When prioritized replay buffer is enabled, we calculate the MSE manually.
        # This allows us to take into account weights of sampesla dn keep individual
        # loss values for every sample
        if self.prio:
            losses_v = batch_weights_v * (Q_expected - Q_targets) ** 2
            sample_prios_v = losses_v + 1e-5
            loss = losses_v.mean()
            self.memory.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy().squeeze(-1))
        else:
            loss = F.mse_loss(Q_expected, Q_targets)
            
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


