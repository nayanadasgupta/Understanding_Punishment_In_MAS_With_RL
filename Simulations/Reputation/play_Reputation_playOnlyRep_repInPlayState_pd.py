# Reputation impacted by playing alone. Play actions also depend on reputation (play state).
import json
import os
import time
from typing import List, Any

import numpy as np
from tqdm import tqdm

from Agents.ModelParameters import ModelParameters
from Agents.ReputationOnlyAgent import ReputationOnlyAgent
from MatrixGame.IPDGame import cooperation_game
from MatrixGame.MatrixGame import MatrixGame
from Mechanisms.GamePlay import choose_play_actions
from Mechanisms.PartnerSelection import rand_select_partners
from Mechanisms.Reputation import update_reputation_using_play_action


def play_rep_play_only_rep_rep_in_play_state(cooperation_dilemma: MatrixGame, num_eps: int, num_rounds: int,
                                             population: List[ReputationOnlyAgent], logging: bool = True):
    agent_reputations: List[int] = [0 for _ in population]
    agent_prev_play_actions: List[int] = [2 for _ in population]  # 2 is the UNKNOWN (starting) action.
    combined_sum_rewards_per_ep = []

    agent_stats: dict[str: Any] = [{"select_history": [],  # Selections fixed per episode.
                                    "prev_play_select": [],
                                    "prev_reputation_select": [],
                                    "prev_punish_select": [],
                                    "prev_punish_justly_select": [],
                                    "prev_punish_unjustly_select": [],
                                    "play_history": [[] for _ in range(num_eps)],
                                    "punish_history": [[] for _ in range(num_eps)],
                                    "punish_justly": [0 for _ in range(num_eps)],
                                    "punish_unjustly": [0 for _ in range(num_eps)],
                                    "reputation": [[] for _ in range(num_eps)],
                                    "rewards": [[] for _ in range(num_eps)]} for _ in population]
    for episode in tqdm(range(num_eps)):
        combined_sum_reward = 0
        for agent_1_idx, agent_1 in enumerate(population):
            agent_2_idx, agent_2 = rand_select_partners(population, agent_1_idx)
            is_terminal_round = 0
            for round_idx in range(num_rounds):
                play_state_1 = [agent_prev_play_actions[agent_2_idx], agent_prev_play_actions[agent_1_idx],
                                agent_reputations[agent_2_idx], agent_reputations[agent_1_idx]]
                play_state_2 = [agent_prev_play_actions[agent_1_idx], agent_prev_play_actions[agent_2_idx],
                                agent_reputations[agent_1_idx], agent_reputations[agent_2_idx]]

                play_action_idx_1, play_action_idx_2, play_reward_1, play_reward_2 = choose_play_actions(agent_1,
                                                                                                         agent_2,
                                                                                                         play_state_1,
                                                                                                         play_state_2,
                                                                                                         cooperation_dilemma)

                agent_prev_play_actions[agent_1_idx] = play_action_idx_1
                agent_prev_play_actions[agent_2_idx] = play_action_idx_2

                update_reputation_using_play_action(play_action_idx_1, agent_1_idx, agent_reputations)
                update_reputation_using_play_action(play_action_idx_2, agent_2_idx, agent_reputations)

                new_play_state_1 = [play_action_idx_2, play_action_idx_1, agent_reputations[agent_2_idx],
                                    agent_reputations[agent_1_idx]]
                play_exp_1 = (play_state_1, play_action_idx_1, play_reward_1, new_play_state_1, is_terminal_round)
                agent_1.play_model.replay_buffer.store(*play_exp_1)
                agent_1.play_model.optimise_model()

                new_play_state_2 = [play_action_idx_1, play_action_idx_2, agent_reputations[agent_1_idx],
                                    agent_reputations[agent_2_idx]]
                play_exp_2 = (play_state_2, play_action_idx_2, play_reward_2, new_play_state_2, is_terminal_round)
                agent_2.play_model.replay_buffer.store(*play_exp_2)
                agent_2.play_model.optimise_model()

                combined_sum_reward += play_reward_1 + play_reward_2

                if logging:
                    agent_stats[agent_1_idx]["play_history"][episode].append(play_action_idx_1)
                    agent_stats[agent_2_idx]["play_history"][episode].append(play_action_idx_2)
                    agent_stats[agent_1_idx]["reputation"][episode].append(agent_reputations[agent_1_idx])
                    agent_stats[agent_2_idx]["reputation"][episode].append(agent_reputations[agent_2_idx])
                    agent_stats[agent_1_idx]["rewards"][episode].append(play_reward_1)
                    agent_stats[agent_2_idx]["rewards"][episode].append(play_reward_2)

        combined_sum_rewards_per_ep.append(combined_sum_reward)
    mean_combined_sum_reward = np.mean(combined_sum_rewards_per_ep)
    if logging:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Logs',
                               'play_rep_play_only_rep_rep_in_play_state.json'), 'w+', encoding='utf-8') as f:
            json.dump(agent_stats, f, ensure_ascii=False, indent=4)
    return mean_combined_sum_reward, agent_stats


if __name__ == "__main__":
    NUM_AGENTS = 5
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

    population = [ReputationOnlyAgent(play_stage_model_parameters) for _ in
                  range(NUM_AGENTS)]

    start = time.perf_counter()
    mean_combined_sum_reward, agent_stats = play_rep_play_only_rep_rep_in_play_state(cooperation_game, 1000,
                                                                                       10, population)
    print("Combined Mean Sum Reward", mean_combined_sum_reward)
    end = time.perf_counter()

    print("DONE")
    print("Duration", end - start)
