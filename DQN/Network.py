import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Network, self).__init__()
        self.fc_1 = nn.Linear(input_dim, 64)
        self.fc_3 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc_1(x))
        x = self.fc_3(x)
        return x
