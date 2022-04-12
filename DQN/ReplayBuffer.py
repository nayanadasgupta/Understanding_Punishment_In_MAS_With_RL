from typing import Dict
import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim: int, max_size: int, batch_size: int):
        self.max_size = max_size
        self.batch_size = batch_size
        self.states = np.zeros([max_size, state_dim], dtype=np.float32)
        self.next_states = np.zeros([max_size, state_dim], dtype=np.float32)
        self.actions = np.zeros([max_size], dtype=np.float32)
        self.rewards = np.zeros([max_size], dtype=np.float32)
        self.terminals = np.zeros([max_size], dtype=np.float32)
        self.buffer_ptr = 0
        self.cur_buffer_size = 0

    def store(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray,
              is_terminal: bool):
        self.states[self.buffer_ptr] = state
        self.next_states[self.buffer_ptr] = next_state
        self.actions[self.buffer_ptr] = action
        self.rewards[self.buffer_ptr] = reward
        self.terminals[self.buffer_ptr] = is_terminal
        self.buffer_ptr = (self.buffer_ptr + 1) % self.max_size
        self.cur_buffer_size = min(self.cur_buffer_size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        sample_indexes = np.random.choice(self.cur_buffer_size, self.batch_size, replace=False)
        return dict(states=self.states[sample_indexes],
                    next_states=self.next_states[sample_indexes],
                    actions=self.actions[sample_indexes],
                    rewards=self.rewards[sample_indexes],
                    terminals=self.terminals[sample_indexes])

    def __len__(self) -> int:
        return self.cur_buffer_size
