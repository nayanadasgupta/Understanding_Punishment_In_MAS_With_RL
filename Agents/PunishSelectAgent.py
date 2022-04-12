from Agents.ModelParameters import ModelParameters
from DQN.DQNAgent import DQNAgent


class PunishSelectAgent:
    def __init__(self, select_stage_model_parameters: ModelParameters, play_stage_model_parameters: ModelParameters,
                 punish_stage_model_parameters: ModelParameters):

        self.select_model = DQNAgent(
            state_dim=select_stage_model_parameters.input_state_dim,
            action_dim=select_stage_model_parameters.output_action_dim,
            max_buffer_size=select_stage_model_parameters.max_buffer_size,
            batch_size=select_stage_model_parameters.batch_size,
            target_update=select_stage_model_parameters.target_update,
            max_eps=select_stage_model_parameters.max_eps,
            min_eps=select_stage_model_parameters.min_eps,
            eps_decay=select_stage_model_parameters.eps_decay,
            gamma=select_stage_model_parameters.gamma,
            lr=select_stage_model_parameters.lr
        )

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

        self.punish_model = DQNAgent(
            state_dim=punish_stage_model_parameters.input_state_dim,
            action_dim=punish_stage_model_parameters.output_action_dim,
            max_buffer_size=punish_stage_model_parameters.max_buffer_size,
            batch_size=punish_stage_model_parameters.batch_size,
            target_update=punish_stage_model_parameters.target_update,
            max_eps=punish_stage_model_parameters.max_eps,
            min_eps=punish_stage_model_parameters.min_eps,
            eps_decay=punish_stage_model_parameters.eps_decay,
            gamma=punish_stage_model_parameters.gamma,
            lr=punish_stage_model_parameters.lr
        )
