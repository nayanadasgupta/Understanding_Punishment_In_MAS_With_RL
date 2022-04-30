# Reputation impacted by both playing behaviour only. Play actions also depend on reputation (play state).

import os

from Agents.ModelParameters import ModelParameters
from typing import List, Any
from tqdm import tqdm
import json
import time
import numpy as np

from Agents.PunishSelectAgent import PunishSelectAgent
from Mechanisms.PartnerSelection import select_partners
from Mechanisms.Punishment import select_punishers, perform_punishment
from Mechanisms.Reputation import update_reputation_using_play_action
from CentipedeGame.Centipede import choose_centipede_play_actions

centipede_length = 6
giver_cooperate_reward = -1
recipient_cooperate_reward = 5
defect_reward = 0


def play_ttp_s_play_play_only_rep_rep_in_play_state(num_eps: int, max_num_rounds: int,
                                                      population: List[PunishSelectAgent],
                                                      logging: bool = True):
    agent_reputations: List[int] = [0 for _ in population]
    agent_prev_play_actions: List[int] = [2 for _ in population]  # 2 is the UNKNOWN (starting) action.
    combined_sum_rewards_per_ep = []

    agent_stats: dict[str: Any] = [{
        "agent_turn": [[] for _ in range(num_eps)],
        "select_history": [],  # Selections fixed per episode.
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
            running_reward_1 = 0
            running_reward_2 = 0
            combined_punishers_reward = 0
            cur_agent_turn = 0
            stop_activated = False
            for round_index in range(max_num_rounds):
                punisher_1_idx, punisher_1, punisher_2_idx, punisher_2 = select_punishers(population,
                                                                                          {agent_1_idx, agent_2_idx})
                if round_index == max_num_rounds - 1:
                    is_terminal_round = 1
                if cur_agent_turn == 0:
                    cur_agent_turn = 1
                    play_state_1 = [agent_prev_play_actions[agent_2_idx], agent_prev_play_actions[agent_1_idx],
                                    agent_reputations[agent_2_idx], agent_reputations[agent_1_idx]]
                    play_action_idx_1, running_reward_1, running_reward_2, is_terminal_round, stop_activated = choose_centipede_play_actions(
                        agent_1, play_state_1, running_reward_1, running_reward_2, is_terminal_round, stop_activated)
                    agent_prev_play_actions[agent_1_idx] = play_action_idx_1

                    punish_state_1 = [play_action_idx_1, agent_prev_play_actions[agent_2_idx]]

                    punish_action_idx_1 = punisher_1.punish_model.select_action(punish_state_1)
                    punish_reward_1, running_reward_1, agent_stats = perform_punishment(punish_action_idx_1,
                                                                                        play_action_idx_1,
                                                                                        punisher_1_idx,
                                                                                        running_reward_1,
                                                                                        agent_reputations, agent_stats,
                                                                                        logging, episode)

                    update_reputation_using_play_action(play_action_idx_1, agent_1_idx, agent_reputations)

                    combined_punishers_reward += punish_reward_1

                    new_play_state_1 = [agent_prev_play_actions[agent_2_idx], play_action_idx_1,
                                        agent_reputations[agent_2_idx], agent_reputations[agent_1_idx]]
                    play_exp_1 = (
                        play_state_1, play_action_idx_1, running_reward_1, new_play_state_1, is_terminal_round)
                    agent_1.play_model.replay_buffer.store(*play_exp_1)
                    agent_1.play_model.optimise_model()

                    new_punish_state_1 = punish_state_1
                    punish_exp_1 = (punish_state_1, punish_action_idx_1, punish_reward_1, new_punish_state_1,
                                    is_terminal_round)
                    punisher_1.punish_model.replay_buffer.store(*punish_exp_1)
                    punisher_1.punish_model.optimise_model()

                    if logging:
                        agent_stats[agent_1_idx]["agent_turn"][episode].append(cur_agent_turn)
                        agent_stats[agent_1_idx]["play_history"][episode].append(play_action_idx_1)
                        agent_stats[punisher_1_idx]["punish_history"][episode].append(punish_action_idx_1)
                        agent_stats[agent_1_idx]["reputation"][episode].append(agent_reputations[agent_1_idx])
                        agent_stats[punisher_1_idx]["reputation"][episode].append(agent_reputations[punisher_1_idx])
                        agent_stats[agent_1_idx]["rewards"][episode].append(running_reward_1)
                        agent_stats[punisher_1_idx]["rewards"][episode].append(punish_reward_1)

                if stop_activated:
                    break

                else:
                    cur_agent_turn = 0
                    play_state_2 = [agent_prev_play_actions[agent_1_idx], agent_prev_play_actions[agent_2_idx],
                                    agent_reputations[agent_1_idx], agent_reputations[agent_2_idx]]

                    play_action_idx_2, running_reward_2, running_reward_1, is_terminal_round, stop_activated = choose_centipede_play_actions(
                        agent_2, play_state_2, running_reward_2, running_reward_1, is_terminal_round, stop_activated)
                    agent_prev_play_actions[agent_2_idx] = play_action_idx_2

                    punish_state_2 = [play_action_idx_2, agent_prev_play_actions[agent_1_idx]]
                    punish_action_idx_2 = punisher_2.punish_model.select_action(punish_state_2)

                    punish_reward_2, running_reward_2, agent_stats = perform_punishment(punish_action_idx_2,
                                                                                        play_action_idx_2,
                                                                                        punisher_2_idx,
                                                                                        running_reward_2,
                                                                                        agent_reputations, agent_stats,
                                                                                        logging, episode)

                    update_reputation_using_play_action(play_action_idx_2, agent_2_idx, agent_reputations)

                    combined_punishers_reward += punish_reward_2

                    new_play_state_2 = [agent_prev_play_actions[agent_1_idx], play_action_idx_2,
                                        agent_reputations[agent_1_idx], agent_reputations[agent_2_idx]]
                    play_exp_2 = (
                        play_state_2, play_action_idx_2, running_reward_2, new_play_state_2, is_terminal_round)
                    agent_2.play_model.replay_buffer.store(*play_exp_2)
                    agent_2.play_model.optimise_model()

                    new_punish_state_2 = punish_state_2
                    punish_exp_2 = (punish_state_2, punish_action_idx_2, punish_reward_2, new_punish_state_2,
                                    is_terminal_round)
                    punisher_2.punish_model.replay_buffer.store(*punish_exp_2)
                    punisher_2.punish_model.optimise_model()

                    if logging:
                        agent_stats[agent_2_idx]["agent_turn"][episode].append(cur_agent_turn)
                        agent_stats[agent_2_idx]["play_history"][episode].append(play_action_idx_2)
                        agent_stats[punisher_2_idx]["punish_history"][episode].append(punish_action_idx_2)
                        agent_stats[agent_2_idx]["reputation"][episode].append(agent_reputations[agent_2_idx])
                        agent_stats[punisher_2_idx]["reputation"][episode].append(agent_reputations[punisher_2_idx])
                        agent_stats[agent_2_idx]["rewards"][episode].append(running_reward_2)
                        agent_stats[punisher_2_idx]["rewards"][episode].append(punish_reward_2)

                    if stop_activated:
                        break

            # Only agent 1 makes a choice of partner.
            new_select_state_1 = agent_reputations.copy()
            select_exp_1 = (select_state, possible_partner_idx, running_reward_1, new_select_state_1, is_terminal_round)
            agent_1.select_model.replay_buffer.store(*select_exp_1)
            agent_1.select_model.optimise_model()

            combined_sum_reward = running_reward_1 + running_reward_2 + combined_punishers_reward

        combined_sum_rewards_per_ep.append(combined_sum_reward)
    mean_combined_sum_reward = np.mean(combined_sum_rewards_per_ep)

    if logging:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Logs',
                               'play_ttp_s_play_play_only_rep_rep_in_play_state.json'), 'w+', encoding='utf-8') as f:
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

    population = [PunishSelectAgent(select_stage_model_parameters, play_stage_model_parameters,
                                    punish_stage_model_parameters) for _ in range(NUM_AGENTS)]

    start = time.perf_counter()
    mean_combined_sum_reward, agent_stats = play_ttp_s_play_play_only_rep_rep_in_play_state(10, 2, population)
    print("Combined Mean Sum Reward", mean_combined_sum_reward)
    end = time.perf_counter()

    print("DONE")
    print("Duration", end - start)
