from Agents.ModelParameters import ModelParameters
from DQN.DQNAgent import DQNAgent


class ReputationOnlyAgent:
    def __init__(self, play_stage_model_parameters: ModelParameters):

        self.play_model = DQNAgent(
            state_dim=play_stage_model_parameters.input_state_dim,
            action_dim=play_stage_model_parameters.output_action_dim,
            max_buffer_size=play_stage_model_parameters.max_buffer_size,
            batch_size=play_stage_model_parameters.batch_size,
            target_update=play_stage_model_parameters.target_update,
            max_eps=play_stage_model_parameters.max_eps,
            min_eps=play_stage_model_parameters.min_eps,
            eps_decay=play_stage_model_parameters.eps_decay,
            gamma=play_stage_model_parameters.gamma,
            lr=play_stage_model_parameters.lr
        )
