from dataclasses import dataclass

@dataclass
class ModelParameters:
    input_state_dim: int
    output_action_dim: int
    max_buffer_size: int
    batch_size: int
    target_update: int
    min_eps: float
    max_eps: float
    eps_decay: float
    gamma: float
    lr: float