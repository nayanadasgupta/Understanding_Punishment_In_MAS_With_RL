from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from DQN.ReplayBuffer import ReplayBuffer
from DQN.Network import Network


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, max_buffer_size: int, batch_size: int, target_update: int,
                 min_eps: float, max_eps: float, eps_decay: float, gamma: float, lr: float):

        self.replay_buffer = ReplayBuffer(state_dim=state_dim, max_size=max_buffer_size, batch_size=batch_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size

        self.target_update = target_update

        self.min_eps = min_eps
        self.max_eps = max_eps
        self.epsilon = max_eps
        self.eps_decay = eps_decay

        self.gamma = gamma
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_network = Network(state_dim, action_dim).to(self.device)
        self.target_network = Network(state_dim, action_dim).to(self.device)

        self.update_count = 0

        # Set target model parameters equal to online model parameters
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.lr)

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action based on current state using an epsilon-greedy strategy to balance exploitation and exploration
        tradeoff.
        :param state: np.ndarray
        :return: selected_action: int
        """
        if self.epsilon > np.random.random():
            return np.random.randint(self.action_dim)
        else:
            cur_state = torch.FloatTensor(state).to(self.device)
            q_values = self.online_network(cur_state)
            selected_action = q_values.argmax().detach().cpu().item()
            return selected_action

    def batch_loss(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        states = torch.FloatTensor(batch["states"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).reshape(-1, 1).to(self.device).long()
        rewards = torch.FloatTensor(batch["rewards"]).reshape(-1, 1).to(self.device)
        terminals = torch.FloatTensor(batch["terminals"]).reshape(-1, 1).to(self.device)

        curr_q_values = self.online_network(states).gather(dim=1, index=actions)
        next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0].detach()
        # With terminal state management.
        q_targets = (rewards + self.gamma * next_q_values * torch.sub(torch.ones_like(terminals), terminals)).to(self.device)
        losses = F.smooth_l1_loss(curr_q_values, q_targets)
        return losses

    def update(self) -> int:
        self.optimizer.zero_grad()
        batch = self.replay_buffer.sample_batch()
        loss = self.batch_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def optimise_model(self):
        """ Optimises DQN Model. Returns has_updated, loss, where loss is None if has_updated is False. """
        has_updated = False
        loss = None
        if len(self.replay_buffer) >= self.batch_size:
            has_updated = True
            loss = self.update()
            # Linearly decrease epsilon
            self.epsilon = max(self.min_eps, self.epsilon -
                                (self.max_eps - self.min_eps) * self.eps_decay)
            self.update_count += 1
            if self.update_count % self.target_update == 0:
                self.update_target_network()
        return has_updated, loss


