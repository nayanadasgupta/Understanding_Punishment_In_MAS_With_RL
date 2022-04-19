from typing import List, Set
import random
from Agents.PunishSelectAgent import PunishSelectAgent
import networkx as nx

punishment_victim_penalty = -3

just_punish_reward = 2
no_punish_reward = 0
unjust_punish_reward = -10

def select_punishers(population: List[PunishSelectAgent], play_agent_idxs: Set[int]):
    possible_punishers = [(punisher_idx, punisher) for punisher_idx, punisher in enumerate(population)
                          if punisher_idx not in play_agent_idxs]
    # The punisher for the two agents can be the same or different.
    punisher_1_idx, punisher_1 = random.choice(possible_punishers)
    punisher_2_idx, punisher_2 = random.choice(possible_punishers)
    return punisher_1_idx, punisher_1, punisher_2_idx, punisher_2


def select_punishers_caveman(G: nx.Graph, population: List[PunishSelectAgent], play_agent_idxs: List[int]):
    neighbour_union = set(G.neighbors(play_agent_idxs[0])).union(set(G.neighbors(play_agent_idxs[1])))
    # Punishers must be a neighbour of either of the playing agents and cannot be the playing agents themselves.
    possible_punishers = [(punisher_idx, population[punisher_idx]) for punisher_idx in neighbour_union
                          if punisher_idx not in set(play_agent_idxs)]
    # The punisher for the two agents can be the same or different.
    punisher_1_idx, punisher_1 = random.choice(possible_punishers)
    punisher_2_idx, punisher_2 = random.choice(possible_punishers)
    return punisher_1_idx, punisher_1, punisher_2_idx, punisher_2


def perform_punishment(punish_action_idx, victim_action_idx, punisher_idx, current_victim_reward, agent_reputations,
                       agent_stats, logging, episode):
    punishment_dict = {0: "NP", 1: "P"}
    play_dict = {0: "C", 1: "D"}
    punish_reward = 0
    if punishment_dict[punish_action_idx] == "P":
        current_victim_reward += punishment_victim_penalty
        if play_dict[victim_action_idx] == "C":
            punish_reward += unjust_punish_reward
            if logging:
                agent_stats[punisher_idx]["punish_unjustly"][episode] += 1
        else:
            punish_reward += just_punish_reward
            if logging:
                agent_stats[punisher_idx]["punish_justly"][episode] += 1
    return punish_reward, current_victim_reward, agent_stats