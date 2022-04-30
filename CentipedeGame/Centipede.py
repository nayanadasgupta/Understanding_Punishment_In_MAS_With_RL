from Agents.PunishSelectAgent import PunishSelectAgent
from typing import List


def choose_centipede_play_actions(agent: PunishSelectAgent, play_state: List, own_running_reward: int,
                                  opponent_running_reward: int, is_terminal_round: int, stop_activated: bool):
    play_dict = {0: "C", 1: "D"}
    play_action = agent.play_model.select_action(play_state)
    if play_dict[play_action] == "C":
        own_running_reward -= 1
        opponent_running_reward += 5
    else:
        is_terminal_round = 1
        stop_activated = True
    return play_action, own_running_reward, opponent_running_reward, is_terminal_round, stop_activated
