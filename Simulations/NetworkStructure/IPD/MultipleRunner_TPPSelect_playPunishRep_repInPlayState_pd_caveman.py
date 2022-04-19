import os

from tqdm import tqdm
import json
import networkx as nx
from Agents.ModelParameters import ModelParameters
from Agents.PunishSelectAgent import PunishSelectAgent
from MatrixGame.IPDGame import cooperation_game
from play_TPPSelect_playPunishRep_repInPlayState_pd_caveman import \
    play_tpp_select_play_punish_rep_rep_in_play_state_pd_caveman

if __name__ == "__main__":
    NUM_RUNS = 20
    NUM_EPISODES = 2000
    NUM_ROUNDS = 10
    NUM_AGENTS = 25
    COMMUNITY_SIZE = 5
    game_name = "play_tpp_select_play_punish_rep_rep_in_play_state_pd_caveman"

    run_stats = []
    run_mean_combined_reward = []
    for run in tqdm(range(NUM_RUNS)):
        G = nx.connected_caveman_graph(int(NUM_AGENTS / COMMUNITY_SIZE), COMMUNITY_SIZE)
        population = []
        for agent_index in range(NUM_AGENTS):
            select_stage_model_parameters = ModelParameters(
                input_state_dim=len(list(G.neighbors(agent_index))),
                output_action_dim=len(list(G.neighbors(agent_index))),  # Can select any of your neighbours
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
                input_state_dim=4,
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

            agent = PunishSelectAgent(select_stage_model_parameters, play_stage_model_parameters,
                                      punish_stage_model_parameters)
            population.append(agent)

        mean_combined_sum_reward, agent_stats = play_tpp_select_play_punish_rep_rep_in_play_state_pd_caveman(
            cooperation_game, NUM_EPISODES,
            NUM_ROUNDS, G, population)
        print()
        print(f"Run {run}: Combined Mean Sum Reward", mean_combined_sum_reward)
        print()
        run_stats.append(agent_stats)
        run_mean_combined_reward.append(mean_combined_sum_reward)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Results', f'run_stats_{game_name}_{NUM_RUNS}_{NUM_EPISODES}_{NUM_ROUNDS}_{NUM_AGENTS}_agents.txt'), 'w') as f:
        json.dump(run_stats, f)
