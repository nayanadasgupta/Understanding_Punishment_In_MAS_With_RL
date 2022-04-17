def choose_play_actions(agent_1, agent_2, play_state_1, play_state_2, dilemma):
    play_action_idx_1 = agent_1.play_model.select_action(play_state_1)
    play_action_idx_2 = agent_2.play_model.select_action(play_state_2)
    play_reward_1 = dilemma.payoffs["row"][play_action_idx_1, play_action_idx_2]
    play_reward_2 = dilemma.payoffs["col"][play_action_idx_1, play_action_idx_2]
    return play_action_idx_1, play_action_idx_2, play_reward_1, play_reward_2