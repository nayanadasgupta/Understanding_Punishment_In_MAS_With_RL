# Reputation impacted by both playing and punishing.
# No state holds reputation information.

import os

from Agents.ModelParameters import ModelParameters
from typing import List, Any
from tqdm import tqdm
import json
import time
import numpy as np

from Agents.PunishSelectAgent import PunishSelectAgent
from MatrixGame.MatrixGame import MatrixGame
from MatrixGame.IPDGame import cooperation_game
from Mechanisms.PartnerSelection import select_partners
from Mechanisms.Punishment import perform_punishment
from Mechanisms.Reputation import update_reputation_using_play_action, update_reputation_using_punish_action
from Mechanisms.GamePlay import choose_play_actions


def play_dp_s_play_punish_rep_rep_in_punish_state(cooperation_dilemma: MatrixGame, num_eps: int, num_rounds: int,
                                                population: List[PunishSelectAgent], logging: bool = True):
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
        select_state = agent_reputations.copy()
        play_pairings = select_partners(population, select_state)
        for agent_1_idx, agent_1 in enumerate(population):
            agent_2_idx, possible_partner_idx = play_pairings[agent_1_idx]
            agent_2 = population[agent_2_idx]
            if logging:
                agent_stats[agent_1_idx]["select_history"].append(agent_2_idx)
                if len(agent_stats[agent_2_idx]["select_history"]) > 0:
                    agent_stats[agent_1_idx]["prev_play_select"].append(
                        (agent_2_idx, agent_stats[agent_2_idx]["select_history"][-1]))
                agent_stats[agent_1_idx]["prev_reputation_select"].append((agent_2_idx, select_state[agent_2_idx]))
                flattened_agent_2_punish_history = [punish for episode in agent_stats[agent_2_idx]["punish_history"] for
                                                    punish in episode]
                if len(flattened_agent_2_punish_history) > 0:
                    agent_stats[agent_1_idx]["prev_punish_select"].append(
                        (agent_2_idx, flattened_agent_2_punish_history[-1]))
                if len(agent_stats[agent_2_idx]["punish_justly"]) > 0:
                    agent_stats[agent_1_idx]["prev_punish_justly_select"].append(
                        (agent_2_idx, agent_stats[agent_2_idx]["punish_justly"][-1]))
                if len(agent_stats[agent_2_idx]["punish_unjustly"]) > 0:
                    agent_stats[agent_1_idx]["prev_punish_unjustly_select"].append(
                        (agent_2_idx, agent_stats[agent_2_idx]["punish_unjustly"][-1]))
            is_terminal_round = 0
            for round_idx in range(num_rounds):
                play_state_1 = [agent_prev_play_actions[agent_2_idx], agent_prev_play_actions[agent_1_idx]]
                play_state_2 = [agent_prev_play_actions[agent_1_idx], agent_prev_play_actions[agent_2_idx]]

                play_action_idx_1, play_action_idx_2, play_reward_1, play_reward_2 = choose_play_actions(agent_1,
                                                                                                         agent_2,
                                                                                                         play_state_1,
                                                                                                         play_state_2,
                                                                                                         cooperation_dilemma)

                agent_prev_play_actions[agent_1_idx] = play_action_idx_1
                agent_prev_play_actions[agent_2_idx] = play_action_idx_2

                update_reputation_using_play_action(play_action_idx_1, agent_1_idx, agent_reputations)
                update_reputation_using_play_action(play_action_idx_2, agent_2_idx, agent_reputations)

                punisher_1 = agent_2
                punisher_1_idx = agent_2_idx
                punisher_2 = agent_1
                punisher_2_idx = agent_1_idx

                punish_state_1 = [play_action_idx_1, play_action_idx_2,
                                  agent_reputations[agent_2_idx], agent_reputations[agent_1_idx]]
                punish_state_2 = [play_action_idx_2, play_action_idx_1,
                                  agent_reputations[agent_1_idx], agent_reputations[agent_2_idx]]

                punish_action_idx_1 = punisher_1.punish_model.select_action(punish_state_1)
                punish_action_idx_2 = punisher_2.punish_model.select_action(punish_state_2)

                punish_reward_1, play_reward_1, agent_stats = perform_punishment(punish_action_idx_1, play_action_idx_1,
                                                                                 punisher_1_idx, play_reward_1,
                                                                                 agent_reputations, agent_stats,
                                                                                 logging, episode)

                update_reputation_using_punish_action(punish_action_idx_1, play_action_idx_1, punisher_1_idx,
                                                      agent_reputations)

                punish_reward_2, play_reward_2, agent_stats = perform_punishment(punish_action_idx_2, play_action_idx_2,
                                                                                 punisher_2_idx, play_reward_2,
                                                                                 agent_reputations, agent_stats,
                                                                                 logging, episode)

                update_reputation_using_punish_action(punish_action_idx_2, play_action_idx_2, punisher_2_idx,
                                                      agent_reputations)


                # Only agent 1 performs partner selection.
                new_select_state_1 = agent_reputations.copy()
                select_exp_1 = (
                    select_state, possible_partner_idx, play_reward_1, new_select_state_1, is_terminal_round)
                agent_1.select_model.replay_buffer.store(*select_exp_1)
                agent_1.select_model.optimise_model()

                new_play_state_1 = [play_action_idx_2, play_action_idx_1]
                play_exp_1 = (play_state_1, play_action_idx_1, play_reward_1, new_play_state_1, is_terminal_round)
                agent_1.play_model.replay_buffer.store(*play_exp_1)
                agent_1.play_model.optimise_model()

                new_play_state_2 = [play_action_idx_1, play_action_idx_2]
                play_exp_2 = (play_state_2, play_action_idx_2, play_reward_2, new_play_state_2, is_terminal_round)
                agent_2.play_model.replay_buffer.store(*play_exp_2)
                agent_2.play_model.optimise_model()

                new_punish_state_1 = [play_action_idx_1, play_action_idx_2,
                                      agent_reputations[agent_2_idx], agent_reputations[agent_1_idx]]
                punish_exp_1 = (punish_state_1, punish_action_idx_1, punish_reward_1, new_punish_state_1,
                                is_terminal_round)
                punisher_1.punish_model.replay_buffer.store(*punish_exp_1)
                punisher_1.punish_model.optimise_model()

                new_punish_state_2 = [play_action_idx_2, play_action_idx_1,
                                      agent_reputations[agent_1_idx], agent_reputations[agent_2_idx]]
                punish_exp_2 = (punish_state_2, punish_action_idx_2, punish_reward_2, new_punish_state_2,
                                is_terminal_round)
                punisher_2.punish_model.replay_buffer.store(*punish_exp_2)
                punisher_2.punish_model.optimise_model()

                combined_sum_reward += play_reward_1 + play_reward_2 + punish_reward_1 + punish_reward_2

                if logging:
                    agent_stats[agent_1_idx]["play_history"][episode].append(play_action_idx_1)
                    agent_stats[agent_2_idx]["play_history"][episode].append(play_action_idx_2)
                    agent_stats[punisher_1_idx]["punish_history"][episode].append(punish_action_idx_1)
                    agent_stats[punisher_2_idx]["punish_history"][episode].append(punish_action_idx_2)
                    agent_stats[agent_1_idx]["reputation"][episode].append(agent_reputations[agent_1_idx])
                    agent_stats[agent_2_idx]["reputation"][episode].append(agent_reputations[agent_2_idx])
                    agent_stats[agent_1_idx]["rewards"][episode].append(play_reward_1 + punish_reward_1)
                    agent_stats[agent_2_idx]["rewards"][episode].append(play_reward_2 + punish_reward_2)

        combined_sum_rewards_per_ep.append(combined_sum_reward)
    mean_combined_sum_reward = np.mean(combined_sum_rewards_per_ep)
    if logging:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Logs', 'play_dp_s_play_punish_rep_rep_in_punish_state.json'), 'w+', encoding='utf-8') as f:
            json.dump(agent_stats, f, ensure_ascii=False, indent=4)
    return mean_combined_sum_reward, agent_stats


if __name__ == "__main__":
    NUM_AGENTS = 5
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
        input_state_dim=2,
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
        input_state_dim=4,
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

    population = [PunishSelectAgent(select_stage_model_parameters, play_stage_model_parameters,
                                    punish_stage_model_parameters) for _ in range(NUM_AGENTS)]

    start = time.perf_counter()
    mean_combined_sum_reward, agent_stats = play_dp_s_play_punish_rep_rep_in_punish_state(cooperation_game, 1000,
                                                                                        10, population)
    print("Combined Mean Sum Reward", mean_combined_sum_reward)
    end = time.perf_counter()

    print("DONE")
    print("Duration", end - start)
