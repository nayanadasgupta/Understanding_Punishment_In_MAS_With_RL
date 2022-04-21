from typing import List

cooperation_reputation = 1
defection_reputation = -1

just_punish_reputation = 2
unjust_punish_reputation = -3
no_punish_reputation = 0
avenger_reputation = 5
persecutor_reputation = -5


def update_reputation_using_play_action(action_idx, agent_idx, agent_reputations):
    play_dict = {0: "C", 1: "D"}
    if play_dict[action_idx] == "C":
        agent_reputations[agent_idx] += cooperation_reputation
    else:
        agent_reputations[agent_idx] += defection_reputation


def update_reputation_using_punish_action(punish_action_idx, victim_action_idx, punisher_idx, agent_reputations):
    punishment_dict = {0: "NP", 1: "P"}
    play_dict = {0: "C", 1: "D"}
    if punishment_dict[punish_action_idx] == "P":
        if play_dict[victim_action_idx] == "C":
            agent_reputations[punisher_idx] += unjust_punish_reputation
        else:
            agent_reputations[punisher_idx] += just_punish_reputation
    else:
        agent_reputations[punisher_idx] += no_punish_reputation


def societal_reputation_update(agent_reputations: List[List[int]], current_agent_idx: int, reputation_update: int):
    for i, reputation_view in enumerate(agent_reputations):
        if i == current_agent_idx:
            continue
        else:
            reputation_view[current_agent_idx] += reputation_update
    return agent_reputations