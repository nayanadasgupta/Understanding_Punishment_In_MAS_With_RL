from tqdm import tqdm
import json

from Agents.ModelParameters import ModelParameters
from Agents.PunishSelectAgent import PunishSelectAgent
from play_TPPSelect_playPunishRep_repInPlayState_pd import play_ttp_s_play_punish_rep_rep_in_play_state
from MatrixGame.IPDGame import cooperation_game

NUM_AGENTS = 5
NUM_RUNS = 20
NUM_EPISODES = 2000
NUM_ROUNDS = 10

select_stage_model_parameters = ModelParameters(
    input_state_dim=NUM_AGENTS,
    output_action_dim=NUM_AGENTS - 1,  # Which agent to choose as a partner (not including yourself).
    max_buffer_size=131072,
    batch_size=100,
    target_update=200,
    min_eps=0.0001,
    max_eps=0.8889,
    eps_decay=0.30006666666666665,
    gamma=0.9,
    lr=0.01
)

play_stage_model_parameters = ModelParameters(
    input_state_dim=4,  # Opponent's previous game action and previous punishment action.
    output_action_dim=2,  # Cooperate or defect.
    max_buffer_size=131072,
    batch_size=100,
    target_update=200,
    min_eps=0.01,
    max_eps=0.8889,
    eps_decay=0.30006666666666665,
    gamma=0.9,
    lr=0.1
)

punish_stage_model_parameters = ModelParameters(
    input_state_dim=2,
    output_action_dim=2,  # Punish or not punish.
    max_buffer_size=524288,
    batch_size=100,
    target_update=200,
    min_eps=0.2,
    max_eps=0.8889,
    eps_decay=0.5000444444444445,
    gamma=0.9,
    lr=0.001
)

if __name__ == "__main__":
    game_name = "play_ttp_s_play_punish_rep_rep_in_play_state"
    run_stats = []
    run_mean_combined_reward = []
    for run in tqdm(range(NUM_RUNS)):
        population = [PunishSelectAgent(select_stage_model_parameters, play_stage_model_parameters,
                                             punish_stage_model_parameters) for _ in range(NUM_AGENTS)]
        mean_combined_sum_reward, agent_stats = play_ttp_s_play_punish_rep_rep_in_play_state(cooperation_game,
                                                                                             NUM_EPISODES,
                                                                                             NUM_ROUNDS, population)
        print()
        print(f"Run {run}: Combined Mean Sum Reward", mean_combined_sum_reward)
        print()
        run_stats.append(agent_stats)
        run_mean_combined_reward.append(mean_combined_sum_reward)
    with open(f"./Results/run_stats_{game_name}_{NUM_RUNS}_{NUM_EPISODES}_{NUM_ROUNDS}_{NUM_AGENTS}_agents.txt", 'w') as f:
        json.dump(run_stats, f)
