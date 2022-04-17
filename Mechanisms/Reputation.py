cooperation_reputation = 1
defection_reputation = -1

just_punish_reputation = 2
unjust_punish_reputation = -3
no_punish_reputation = 0


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
