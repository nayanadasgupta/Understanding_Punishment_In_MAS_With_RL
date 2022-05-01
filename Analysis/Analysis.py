import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

def find_reward_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_reward_per_ep = []
    for run in range(NUM_RUNS):
        reward_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES):
            total_reward = 0
            for individual_agent_stats in run_stats:
                total_reward += np.sum(individual_agent_stats["rewards"][episode])
            reward_per_episode.append(total_reward)
        all_runs_reward_per_ep.append(reward_per_episode)
    mean_reward_per_episode = [np.mean(k) for k in
                               zip(*all_runs_reward_per_ep)]
    return mean_reward_per_episode


def find_cooperation_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_cooperation_per_episode = []
    for run in range(NUM_RUNS):
        percentage_cooperation_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES):
            num_cooperation = 0
            num_defection = 0
            for individual_agent_stats in run_stats:
                num_cooperation += individual_agent_stats["play_history"][episode].count(0)
                num_defection += individual_agent_stats["play_history"][episode].count(1)
            percentage_cooperation_per_episode.append(num_cooperation / (num_cooperation + num_defection))
        all_runs_percentage_cooperation_per_episode.append(percentage_cooperation_per_episode)
    mean_percentage_cooperation_per_episode = [np.mean(k) for k in
                                               zip(*all_runs_percentage_cooperation_per_episode)]
    return mean_percentage_cooperation_per_episode


def find_punishment_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_punishment_per_episode = []
    for run in range(NUM_RUNS):
        percentage_punishment_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES):
            num_punishment = 0
            num_not_punishment = 0
            for individual_agent_stats in run_stats:
                num_not_punishment += individual_agent_stats["punish_history"][episode].count(0)
                num_punishment += individual_agent_stats["punish_history"][episode].count(1)
            percentage_punishment_per_episode.append(num_punishment / (num_not_punishment + num_punishment))
        all_runs_percentage_punishment_per_episode.append(percentage_punishment_per_episode)
    mean_percentage_punishment_per_episode = [np.mean(k) for k in
                                              zip(*all_runs_percentage_punishment_per_episode)]
    return mean_percentage_punishment_per_episode


def find_combined_direct_tpp_punishment_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_punishment_per_episode = []
    for run in range(NUM_RUNS):
        percentage_punishment_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES):
            num_punishment = 0
            num_not_punishment = 0
            for individual_agent_stats in run_stats:
                num_not_punishment += individual_agent_stats["direct_punish_history"][episode].count(0)
                num_punishment += individual_agent_stats["direct_punish_history"][episode].count(1)
                num_not_punishment += individual_agent_stats["tpp_punish_history"][episode].count(0)
                num_punishment += individual_agent_stats["tpp_punish_history"][episode].count(1)
            percentage_punishment_per_episode.append(num_punishment / (num_not_punishment + num_punishment))
        all_runs_percentage_punishment_per_episode.append(percentage_punishment_per_episode)
    mean_percentage_punishment_per_episode = [np.mean(k) for k in
                                              zip(*all_runs_percentage_punishment_per_episode)]
    return mean_percentage_punishment_per_episode


def find_direct_punishment_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_punishment_per_episode = []
    for run in range(NUM_RUNS):
        percentage_punishment_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES):
            num_punishment = 0
            num_not_punishment = 0
            for individual_agent_stats in run_stats:
                num_not_punishment += individual_agent_stats["direct_punish_history"][episode].count(0)
                num_punishment += individual_agent_stats["direct_punish_history"][episode].count(1)
            percentage_punishment_per_episode.append(num_punishment / (num_not_punishment + num_punishment))
        all_runs_percentage_punishment_per_episode.append(percentage_punishment_per_episode)
    mean_percentage_punishment_per_episode = [np.mean(k) for k in
                                              zip(*all_runs_percentage_punishment_per_episode)]
    return mean_percentage_punishment_per_episode


def find_tpp_punishment_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_punishment_per_episode = []
    for run in range(NUM_RUNS):
        percentage_punishment_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES):
            num_punishment = 0
            num_not_punishment = 0
            for individual_agent_stats in run_stats:
                num_not_punishment += individual_agent_stats["tpp_punish_history"][episode].count(0)
                num_punishment += individual_agent_stats["tpp_punish_history"][episode].count(1)
            percentage_punishment_per_episode.append(num_punishment / (num_not_punishment + num_punishment))
        all_runs_percentage_punishment_per_episode.append(percentage_punishment_per_episode)
    mean_percentage_punishment_per_episode = [np.mean(k) for k in
                                              zip(*all_runs_percentage_punishment_per_episode)]
    return mean_percentage_punishment_per_episode


def find_punish_justly_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_punish_justly_per_episode = []
    for run in range(NUM_RUNS):
        percentage_just_punishment_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES):
            num_just_punishment = 0
            num_unjust_punishment = 0
            for individual_agent_stats in run_stats:
                num_just_punishment += individual_agent_stats["punish_justly"][episode]
                num_unjust_punishment += individual_agent_stats["punish_unjustly"][episode]
            if num_just_punishment == 0 and num_unjust_punishment == 0:
                percentage_just_punishment_per_episode.append(0)
            else:
                percentage_just_punishment_per_episode.append(
                    num_just_punishment / (num_just_punishment + num_unjust_punishment))
        all_runs_percentage_punish_justly_per_episode.append(percentage_just_punishment_per_episode)
    mean_percentage_punish_justly_per_episode = [np.mean(k) for k in
                                                 zip(*all_runs_percentage_punish_justly_per_episode)]
    return mean_percentage_punish_justly_per_episode

def find_combined_tpp_direct_punish_justly_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_punish_justly_per_episode = []
    for run in range(NUM_RUNS):
        percentage_just_punishment_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES):
            num_just_punishment = 0
            num_unjust_punishment = 0
            for individual_agent_stats in run_stats:
                num_just_punishment += individual_agent_stats["direct_punish_justly"][episode]
                num_unjust_punishment += individual_agent_stats["direct_punish_unjustly"][episode]
                num_just_punishment += individual_agent_stats["tpp_punish_justly"][episode]
                num_unjust_punishment += individual_agent_stats["tpp_punish_unjustly"][episode]
            if num_just_punishment == 0 and num_unjust_punishment == 0:
                percentage_just_punishment_per_episode.append(0)
            else:
                percentage_just_punishment_per_episode.append(
                    num_just_punishment / (num_just_punishment + num_unjust_punishment))
        all_runs_percentage_punish_justly_per_episode.append(percentage_just_punishment_per_episode)
    mean_percentage_punish_justly_per_episode = [np.mean(k) for k in
                                                 zip(*all_runs_percentage_punish_justly_per_episode)]
    return mean_percentage_punish_justly_per_episode

def find_direct_punish_justly_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_punish_justly_per_episode = []
    for run in range(NUM_RUNS):
        percentage_just_punishment_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES):
            num_just_punishment = 0
            num_unjust_punishment = 0
            for individual_agent_stats in run_stats:
                num_just_punishment += individual_agent_stats["direct_punish_justly"][episode]
                num_unjust_punishment += individual_agent_stats["direct_punish_unjustly"][episode]
            if num_just_punishment == 0 and num_unjust_punishment == 0:
                percentage_just_punishment_per_episode.append(0)
            else:
                percentage_just_punishment_per_episode.append(
                    num_just_punishment / (num_just_punishment + num_unjust_punishment))
        all_runs_percentage_punish_justly_per_episode.append(percentage_just_punishment_per_episode)
    mean_percentage_punish_justly_per_episode = [np.mean(k) for k in
                                                 zip(*all_runs_percentage_punish_justly_per_episode)]
    return mean_percentage_punish_justly_per_episode


def find_tpp_punish_justly_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_punish_justly_per_episode = []
    for run in range(NUM_RUNS):
        percentage_just_punishment_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES):
            num_just_punishment = 0
            num_unjust_punishment = 0
            for individual_agent_stats in run_stats:
                num_just_punishment += individual_agent_stats["tpp_punish_justly"][episode]
                num_unjust_punishment += individual_agent_stats["tpp_punish_unjustly"][episode]
            if num_just_punishment == 0 and num_unjust_punishment == 0:
                percentage_just_punishment_per_episode.append(0)
            else:
                percentage_just_punishment_per_episode.append(
                    num_just_punishment / (num_just_punishment + num_unjust_punishment))
        all_runs_percentage_punish_justly_per_episode.append(percentage_just_punishment_per_episode)
    mean_percentage_punish_justly_per_episode = [np.mean(k) for k in
                                                 zip(*all_runs_percentage_punish_justly_per_episode)]
    return mean_percentage_punish_justly_per_episode


def find_punish_select_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_punisher_selections_per_episode = []
    for run in range(NUM_RUNS):
        percentage_punisher_selections_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES - 3):
            num_punisher_selections = 0
            num_non_punisher_selections = 0
            for individual_agent_stats in run_stats:
                past_punish_action = individual_agent_stats["prev_punish_select"][episode][1]
                if past_punish_action == 0:  # Not Punish
                    num_non_punisher_selections += 1
                elif past_punish_action == 1:  # Punish
                    num_punisher_selections += 1
            percentage_punisher_selections_per_episode.append(
                num_punisher_selections / (num_punisher_selections + num_non_punisher_selections))
        all_runs_percentage_punisher_selections_per_episode.append(percentage_punisher_selections_per_episode)
    mean_percentage_punisher_selections_per_episode = [np.mean(k) for k in
                                                       zip(*all_runs_percentage_punisher_selections_per_episode)]
    return mean_percentage_punisher_selections_per_episode


def find_combined_direct_tpp_punish_select_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_punisher_selections_per_episode = []
    for run in range(NUM_RUNS):
        percentage_punisher_selections_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES - 3):
            num_punisher_selections = 0
            num_non_punisher_selections = 0
            for individual_agent_stats in run_stats:
                past_direct_punish_action = individual_agent_stats["prev_direct_punish_select"][episode][1]
                past_tpp_punish_action = individual_agent_stats["prev_tpp_punish_select"][episode][1]
                if past_direct_punish_action == 0 and past_tpp_punish_action == 0:  # Not Punish
                    num_non_punisher_selections += 1
                elif past_direct_punish_action == 1 or past_tpp_punish_action == 1:  # Punish
                    num_punisher_selections += 1
            percentage_punisher_selections_per_episode.append(
                num_punisher_selections / (num_punisher_selections + num_non_punisher_selections))
        all_runs_percentage_punisher_selections_per_episode.append(percentage_punisher_selections_per_episode)
    mean_percentage_punisher_selections_per_episode = [np.mean(k) for k in
                                                       zip(*all_runs_percentage_punisher_selections_per_episode)]
    return mean_percentage_punisher_selections_per_episode


def find_direct_punish_select_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_punisher_selections_per_episode = []
    for run in range(NUM_RUNS):
        percentage_punisher_selections_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES - 3):
            num_punisher_selections = 0
            num_non_punisher_selections = 0
            for individual_agent_stats in run_stats:
                past_punish_action = individual_agent_stats["prev_direct_punish_select"][episode][1]
                if past_punish_action == 0:  # Not Punish
                    num_non_punisher_selections += 1
                elif past_punish_action == 1:  # Punish
                    num_punisher_selections += 1
            percentage_punisher_selections_per_episode.append(
                num_punisher_selections / (num_punisher_selections + num_non_punisher_selections))
        all_runs_percentage_punisher_selections_per_episode.append(percentage_punisher_selections_per_episode)
    mean_percentage_punisher_selections_per_episode = [np.mean(k) for k in
                                                       zip(*all_runs_percentage_punisher_selections_per_episode)]
    return mean_percentage_punisher_selections_per_episode


def find_tpp_punish_select_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_punisher_selections_per_episode = []
    for run in range(NUM_RUNS):
        percentage_punisher_selections_per_episode = []
        run_stats = stats[run]
        for episode in range(NUM_EPISODES - 3):
            num_punisher_selections = 0
            num_non_punisher_selections = 0
            for individual_agent_stats in run_stats:
                past_punish_action = individual_agent_stats["prev_tpp_punish_select"][episode][1]
                if past_punish_action == 0:  # Not Punish
                    num_non_punisher_selections += 1
                elif past_punish_action == 1:  # Punish
                    num_punisher_selections += 1
            percentage_punisher_selections_per_episode.append(
                num_punisher_selections / (num_punisher_selections + num_non_punisher_selections))
        all_runs_percentage_punisher_selections_per_episode.append(percentage_punisher_selections_per_episode)
    mean_percentage_punisher_selections_per_episode = [np.mean(k) for k in
                                                       zip(*all_runs_percentage_punisher_selections_per_episode)]
    return mean_percentage_punisher_selections_per_episode


def find_just_punish_select_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_just_punisher_selections_per_episode = []
    for run in range(NUM_RUNS):
        percentage_just_punisher_selections_per_episode = []
        run_stats = stats[run]
        for episode in range(1, NUM_EPISODES):
            num_just_punisher_selections = 0
            num_unjust_punisher_selections = 0
            num_non_punisher_selections = 0
            for individuals_agent_stats in run_stats:
                selected_agent = individuals_agent_stats["select_history"][episode]
                # Was the selected agent majority just or majority unjust in the previous episode?
                selected_agent_punish_justly_prev_ep = run_stats[selected_agent]["punish_justly"][episode - 1]
                selected_agent_punish_unjustly_prev_ep = run_stats[selected_agent]["punish_unjustly"][episode - 1]
                if selected_agent_punish_justly_prev_ep == 0 and selected_agent_punish_unjustly_prev_ep == 0:
                    num_non_punisher_selections += 1
                elif selected_agent_punish_justly_prev_ep >= selected_agent_punish_unjustly_prev_ep:
                    num_just_punisher_selections += 1
                else:
                    num_unjust_punisher_selections += 1
            percentage_just_punisher_selections_per_episode.append(num_just_punisher_selections / (
                    num_just_punisher_selections + num_unjust_punisher_selections + num_non_punisher_selections))
        all_runs_percentage_just_punisher_selections_per_episode.append(percentage_just_punisher_selections_per_episode)
    mean_percentage_just_punisher_selections_per_episode = [np.mean(k) for k in
                                                            zip(*all_runs_percentage_just_punisher_selections_per_episode)]
    return mean_percentage_just_punisher_selections_per_episode


def find_just_combined_direct_tpp_punish_select_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_just_punisher_selections_per_episode = []
    for run in range(NUM_RUNS):
        percentage_just_punisher_selections_per_episode = []
        run_stats = stats[run]
        for episode in range(1, NUM_EPISODES):
            num_just_punisher_selections = 0
            num_unjust_punisher_selections = 0
            num_non_punisher_selections = 0
            for individuals_agent_stats in run_stats:
                selected_agent = individuals_agent_stats["select_history"][episode]
                # Was the selected agent majority just or majority unjust in the previous episode?
                selected_agent_direct_punish_justly_prev_ep = run_stats[selected_agent]["direct_punish_justly"][
                    episode - 1]
                selected_agent_direct_punish_unjustly_prev_ep = run_stats[selected_agent]["direct_punish_unjustly"][
                    episode - 1]
                selected_agent_tpp_punish_justly_prev_ep = run_stats[selected_agent]["tpp_punish_justly"][episode - 1]
                selected_agent_tpp_punish_unjustly_prev_ep = run_stats[selected_agent]["tpp_punish_unjustly"][
                    episode - 1]
                if selected_agent_direct_punish_justly_prev_ep == 0 and selected_agent_direct_punish_unjustly_prev_ep == 0 \
                        and selected_agent_tpp_punish_justly_prev_ep == 0 and selected_agent_tpp_punish_unjustly_prev_ep == 0:
                    num_non_punisher_selections += 1
                elif selected_agent_direct_punish_justly_prev_ep >= selected_agent_direct_punish_unjustly_prev_ep \
                        or selected_agent_tpp_punish_justly_prev_ep >= selected_agent_tpp_punish_unjustly_prev_ep:
                    num_just_punisher_selections += 1
                else:
                    num_unjust_punisher_selections += 1
            percentage_just_punisher_selections_per_episode.append(num_just_punisher_selections / (
                    num_just_punisher_selections + num_unjust_punisher_selections + num_non_punisher_selections))
        all_runs_percentage_just_punisher_selections_per_episode.append(percentage_just_punisher_selections_per_episode)
    mean_percentage_just_punisher_selections_per_episode = [np.mean(k) for k in
                                                            zip(*all_runs_percentage_just_punisher_selections_per_episode)]
    return mean_percentage_just_punisher_selections_per_episode


def find_just_direct_punish_select_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_just_punisher_selections_per_episode = []
    for run in range(NUM_RUNS):
        percentage_just_punisher_selections_per_episode = []
        run_stats = stats[run]
        for episode in range(1, NUM_EPISODES):
            num_just_punisher_selections = 0
            num_unjust_punisher_selections = 0
            num_non_punisher_selections = 0
            for individuals_agent_stats in run_stats:
                selected_agent = individuals_agent_stats["select_history"][episode]
                # Was the selected agent majority just or majority unjust in the previous episode?
                selected_agent_punish_justly_prev_ep = run_stats[selected_agent]["direct_punish_justly"][episode - 1]
                selected_agent_punish_unjustly_prev_ep = run_stats[selected_agent]["direct_punish_unjustly"][
                    episode - 1]
                if selected_agent_punish_justly_prev_ep == 0 and selected_agent_punish_unjustly_prev_ep == 0:
                    num_non_punisher_selections += 1
                elif selected_agent_punish_justly_prev_ep >= selected_agent_punish_unjustly_prev_ep:
                    num_just_punisher_selections += 1
                else:
                    num_unjust_punisher_selections += 1
            percentage_just_punisher_selections_per_episode.append(num_just_punisher_selections / (
                    num_just_punisher_selections + num_unjust_punisher_selections + num_non_punisher_selections))
        all_runs_percentage_just_punisher_selections_per_episode.append(percentage_just_punisher_selections_per_episode)
    mean_percentage_just_punisher_selections_per_episode = [np.mean(k) for k in
                                                            zip(*all_runs_percentage_just_punisher_selections_per_episode)]
    return mean_percentage_just_punisher_selections_per_episode


def find_just_tpp_punish_select_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_just_punisher_selections_per_episode = []
    for run in range(NUM_RUNS):
        percentage_just_punisher_selections_per_episode = []
        run_stats = stats[run]
        for episode in range(1, NUM_EPISODES):
            num_just_punisher_selections = 0
            num_unjust_punisher_selections = 0
            num_non_punisher_selections = 0
            for individuals_agent_stats in run_stats:
                selected_agent = individuals_agent_stats["select_history"][episode]
                # Was the selected agent majority just or majority unjust in the previous episode?
                selected_agent_punish_justly_prev_ep = run_stats[selected_agent]["tpp_punish_justly"][episode - 1]
                selected_agent_punish_unjustly_prev_ep = run_stats[selected_agent]["tpp_punish_unjustly"][episode - 1]
                if selected_agent_punish_justly_prev_ep == 0 and selected_agent_punish_unjustly_prev_ep == 0:
                    num_non_punisher_selections += 1
                elif selected_agent_punish_justly_prev_ep >= selected_agent_punish_unjustly_prev_ep:
                    num_just_punisher_selections += 1
                else:
                    num_unjust_punisher_selections += 1
            percentage_just_punisher_selections_per_episode.append(num_just_punisher_selections / (
                    num_just_punisher_selections + num_unjust_punisher_selections + num_non_punisher_selections))
        all_runs_percentage_just_punisher_selections_per_episode.append(percentage_just_punisher_selections_per_episode)
    mean_percentage_just_punisher_selections_per_episode = [np.mean(k) for k in
                                                            zip(*all_runs_percentage_just_punisher_selections_per_episode)]
    return mean_percentage_just_punisher_selections_per_episode


def find_cooperator_select_per_ep(stats, NUM_RUNS, NUM_EPISODES):
    all_runs_percentage_cooperator_selections_per_episode = []
    for run in range(NUM_RUNS):
        percentage_cooperator_selections_per_episode = []
        run_stats = stats[run]
        for episode in range(1, NUM_EPISODES):
            num_cooperator_selections = 0
            num_defector_selections = 0
            for individuals_agent_stats in run_stats:
                selected_agent = individuals_agent_stats["select_history"][episode]
                # Was the selected agent majority cooperator or majority defector in the previous episode?
                selected_agent_coop_prev_ep = run_stats[selected_agent]["play_history"][episode - 1].count(0)
                selected_agent_defect_prev_ep = run_stats[selected_agent]["play_history"][episode - 1].count(1)
                if selected_agent_coop_prev_ep >= selected_agent_defect_prev_ep:
                    num_cooperator_selections += 1
                else:
                    num_defector_selections += 1
            percentage_cooperator_selections_per_episode.append(
                num_cooperator_selections / (num_defector_selections + num_cooperator_selections))
        all_runs_percentage_cooperator_selections_per_episode.append(percentage_cooperator_selections_per_episode)
    mean_percentage_cooperator_selections_per_episode = [np.mean(k) for k in
                                                         zip(*all_runs_percentage_cooperator_selections_per_episode)]
    return mean_percentage_cooperator_selections_per_episode


def find_reputation_per_ep(stats, num_runs, num_episodes):
    all_runs_rep_per_ep = []
    for run in range(num_runs):
        rep_per_ep = []
        run_stats = stats[run]
        for ep in range(num_episodes):
            total_rep = 0
            for individual_agent_stats in run_stats:
                total_rep += np.sum(individual_agent_stats["reputation"][ep])
            rep_per_ep.append(total_rep)
        all_runs_rep_per_ep.append(rep_per_ep)
    mean_rep_per_episode = [np.mean(k) for k in
                            zip(*all_runs_rep_per_ep)]
    return mean_rep_per_episode


def find_centipede_avg_play_length_per_ep(stats, num_runs, num_episodes):
    all_play_length_per_ep = []
    for run in range(num_runs):
        play_length_per_ep = []
        run_stats = stats[run]
        for ep in range(num_episodes):
            play_length = 0
            for individual_agent_stats in run_stats:
                play_length += len(individual_agent_stats["play_history"][ep])
            play_length_per_ep.append(play_length)
        all_play_length_per_ep.append(play_length_per_ep)
    mean_play_length_per_ep = [np.mean(k) for k in
                               zip(*all_play_length_per_ep)]
    return mean_play_length_per_ep


def plotting_single(title, values, xlabel, ylabel, save_folder):
    plt.figure()
    smoothed_values = pd.DataFrame(values)
    smoothed_path = smoothed_values.rolling(window=100, center=True).mean()
    path_deviation = 1.96 * smoothed_values.rolling(window=100, center=True).std()
    under_line = (smoothed_path - path_deviation)[0]
    over_line = (smoothed_path + path_deviation)[0]
    plt.plot(smoothed_path, linewidth=2)
    plt.fill_between(path_deviation.index, under_line, over_line, color='b', alpha=.1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_title = title.replace(" ", "_")
    plt.savefig(f"{save_folder}/{save_title}")
    plt.show()


def plotting_multiple(title, values_list, xlabel, ylabel, save_folder, legend_list):
    # cmap = ["red", "green", "blue", "purple", "orange", "#e7298a", "black", "#00ff00"]
    cmap = ["red", "green", "blue", "purple", "orange", "black", "#e7298a", "#00ff00"]
    plt.figure()
    for i, values in enumerate(values_list):
        print(i)
        smoothed_values = pd.DataFrame(values)
        smoothed_path = smoothed_values.rolling(window=100, center=True).mean()
        path_deviation = 1.96 * smoothed_values.rolling(window=100, center=True).std()
        under_line = (smoothed_path - path_deviation)[0]
        over_line = (smoothed_path + path_deviation)[0]
        plt.plot(smoothed_path, linewidth=2, color=cmap[i], label=legend_list[i])
        plt.fill_between(path_deviation.index, under_line, over_line, color=cmap[i], alpha=.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_title = title.replace(" ", "_")
    plt.legend()
    plt.savefig(f"{save_folder}/{save_title}")
    plt.show()


if __name__ == "__main__":

    NUM_RUNS = 20
    NUM_EPISODES = 2000

    def analysis():
        plotting_folder = "" # Insert Path of Folder For Plots
        comparison_files = [] # Insert Results Files To Plot
        legend_list = [] # Insert Corresponding Legends
        selection = [True, True, True]
        punishment = [True, True, True]
        punish_selection = [True, True, True]
        play_selection = [True, True, True]
        tpdps = False
        centipede = False
        tpdps_non_separate = False
        title_suffix = "Prisoner's Dilemma With DP-S" # Title
        return plotting_folder, comparison_files, legend_list, selection, punishment, punish_selection, play_selection, \
               tpdps, centipede, title_suffix, tpdps_non_separate



    plotting_folder, comparison_files, legend_list, selection, punishment, punish_selection, play_selection, \
    tpdps, centipede, title_prefix, tpdps_non_separate = analysis()

    print("Loading Files")
    agent_stats_list = []
    for filename in tqdm(comparison_files):
        with open(filename) as f:
            agent_stats = json.load(f)
        agent_stats_list.append(agent_stats)

    print("Comparing Cooperative Selections")
    comparing_cooperation_selections_list = []
    for i, agent_stat in tqdm(enumerate(agent_stats_list)):
        if selection[i]:
            mean_percentage_cooperator_selections_per_episode = find_cooperator_select_per_ep(agent_stat, NUM_RUNS,
                                                                                              NUM_EPISODES)
            comparing_cooperation_selections_list.append(mean_percentage_cooperator_selections_per_episode)

    if not tpdps:
        print("Comparing Punisher Selections")
        comparing_punisher_selections_list = []
        for i, agent_stat in tqdm(enumerate(agent_stats_list)):
            if selection[i] and punish_selection[i]:
                mean_percentage_punisher_selections_per_ep = find_punish_select_per_ep(agent_stat, NUM_RUNS,
                                                                                       NUM_EPISODES)
                comparing_punisher_selections_list.append(mean_percentage_punisher_selections_per_ep)

        print("Comparing Just Punisher Selections")
        comparing_just_punisher_selections_list = []
        for i, agent_stat in tqdm(enumerate(agent_stats_list)):
            if selection[i] and punish_selection[i]:
                mean_percentage_just_punisher_selections_per_episode = find_just_punish_select_per_ep(agent_stat,
                                                                                                      NUM_RUNS,
                                                                                                      NUM_EPISODES)
                comparing_just_punisher_selections_list.append(mean_percentage_just_punisher_selections_per_episode)

    print("Comparing Cooperation")
    comparing_cooperation_list = []
    for agent_stat in tqdm(agent_stats_list):
        mean_percentage_cooperation_per_episode = find_cooperation_per_ep(agent_stat, NUM_RUNS, NUM_EPISODES)
        comparing_cooperation_list.append(mean_percentage_cooperation_per_episode)

    if not tpdps:
        print("Comparing Punishment")
        comparing_punishment_list = []
        for i, agent_stat in tqdm(enumerate(agent_stats_list)):
            if punishment[i]:
                mean_percentage_punishment_per_episode = find_punishment_per_ep(agent_stat, NUM_RUNS, NUM_EPISODES)
                comparing_punishment_list.append(mean_percentage_punishment_per_episode)

        print("Comparing Just Punishment")
        comparing_just_punishment_list = []
        for i, agent_stat in tqdm(enumerate(agent_stats_list)):
            if punishment[i]:
                mean_percentage_punish_justly_per_episode = find_punish_justly_per_ep(agent_stat, NUM_RUNS,
                                                                                      NUM_EPISODES)
                comparing_just_punishment_list.append(mean_percentage_punish_justly_per_episode)

    print("Comparing Reward")
    comparing_reward_list = []
    for agent_stat in tqdm(agent_stats_list):
        mean_reward_per_episode = find_reward_per_ep(agent_stat, NUM_RUNS, NUM_EPISODES)
        comparing_reward_list.append(mean_reward_per_episode)

    print("Comparing Reputation")
    comparing_reputation_list = []
    for agent_stat in tqdm(agent_stats_list):
        mean_rep_per_ep = find_reputation_per_ep(agent_stat, NUM_RUNS, NUM_EPISODES)
        comparing_reputation_list.append(mean_rep_per_ep)

    if centipede:
        print("Comparing Centipede Play Length")
        comparing_centipede_play_length = []
        for agent_stat in tqdm(agent_stats_list):
            mean_play_length_per_ep = find_centipede_avg_play_length_per_ep(agent_stat, NUM_RUNS, NUM_EPISODES)
            comparing_centipede_play_length.append(mean_play_length_per_ep)

    if tpdps:
        if tpdps_non_separate:
            mean_percentage_tpp_s_punisher_selections_per_ep = find_punish_select_per_ep(agent_stats_list[0], NUM_RUNS,
                                                                                         NUM_EPISODES)
            mean_percentage_direct_s_punisher_selections_per_ep = find_punish_select_per_ep(agent_stats_list[2],
                                                                                            NUM_RUNS,
                                                                                            NUM_EPISODES)
            mean_percentage_combined_tppdps_punisher_selections_per_ep = find_combined_direct_tpp_punish_select_per_ep(
                agent_stats_list[4],
                NUM_RUNS,
                NUM_EPISODES)

            mean_percentage_tpp_s_just_punisher_selections_per_ep = find_just_punish_select_per_ep(agent_stats_list[0],
                                                                                                   NUM_RUNS,
                                                                                                   NUM_EPISODES)

            mean_percentage_direct_s_just_punisher_selections_per_ep = find_just_punish_select_per_ep(
                agent_stats_list[2],
                NUM_RUNS,
                NUM_EPISODES)
            mean_percentage_combined_tppdps_just_punisher_selections_per_ep = find_just_combined_direct_tpp_punish_select_per_ep(
                agent_stats_list[4],
                NUM_RUNS, NUM_EPISODES)

            mean_percentage_tpp_s_punishment_per_ep = find_punishment_per_ep(agent_stats_list[0], NUM_RUNS,
                                                                             NUM_EPISODES)
            mean_percentage_tpp_punishment_per_ep = find_punishment_per_ep(agent_stats_list[1], NUM_RUNS, NUM_EPISODES)
            mean_percentage_direct_s_punishment_per_ep = find_punishment_per_ep(agent_stats_list[2], NUM_RUNS,
                                                                                NUM_EPISODES)
            mean_percentage_direct_punishment_per_ep = find_punishment_per_ep(agent_stats_list[3], NUM_RUNS,
                                                                              NUM_EPISODES)

            mean_percentage_combined_tppdps_punishment_per_ep = find_combined_direct_tpp_punishment_per_ep(
                agent_stats_list[4],
                NUM_RUNS,
                NUM_EPISODES)

            mean_percentage_combined_tppdp_punishment_per_ep = find_combined_direct_tpp_punishment_per_ep(agent_stats_list[5],
                                                                                                          NUM_RUNS,
                                                                                                          NUM_EPISODES)

            mean_percentage_tpp_s_punish_justly_per_ep = find_punish_justly_per_ep(agent_stats_list[0], NUM_RUNS,
                                                                                   NUM_EPISODES)
            mean_percentage_tpp_punish_justly_per_ep = find_punish_justly_per_ep(agent_stats_list[1], NUM_RUNS,
                                                                                 NUM_EPISODES)
            mean_percentage_direct_s_punish_justly_per_ep = find_punish_justly_per_ep(agent_stats_list[2], NUM_RUNS,
                                                                                      NUM_EPISODES)
            mean_percentage_direct_punish_justly_per_ep = find_punish_justly_per_ep(agent_stats_list[3], NUM_RUNS,
                                                                                    NUM_EPISODES)
            mean_percentage_combined_tppdps_punish_justly_per_ep = find_combined_tpp_direct_punish_justly_per_ep(
                agent_stats_list[4],
                NUM_RUNS,
                NUM_EPISODES)


            mean_percentage_combined_tppdp_punish_justly_per_ep = find_combined_tpp_direct_punish_justly_per_ep(
                agent_stats_list[5],
                NUM_RUNS,
                NUM_EPISODES)


            punish_justly_comparisons = [mean_percentage_tpp_s_punish_justly_per_ep,
                                         mean_percentage_tpp_punish_justly_per_ep,
                                         mean_percentage_direct_s_punish_justly_per_ep,
                                         mean_percentage_direct_punish_justly_per_ep,
                                         mean_percentage_combined_tppdps_punish_justly_per_ep,
                                         mean_percentage_combined_tppdp_punish_justly_per_ep]

            punish_comparisons = [mean_percentage_tpp_s_punishment_per_ep,
                                  mean_percentage_tpp_punishment_per_ep,
                                  mean_percentage_direct_s_punishment_per_ep,
                                  mean_percentage_direct_punishment_per_ep,
                                  mean_percentage_combined_tppdps_punishment_per_ep,
                                  mean_percentage_combined_tppdp_punishment_per_ep]

            just_punish_select_comparisons = [mean_percentage_tpp_s_just_punisher_selections_per_ep,
                                              mean_percentage_direct_s_just_punisher_selections_per_ep,
                                              mean_percentage_combined_tppdps_just_punisher_selections_per_ep]

            punish_select_comparisons = [mean_percentage_tpp_s_punisher_selections_per_ep,
                                         mean_percentage_direct_s_punisher_selections_per_ep,
                                         mean_percentage_combined_tppdps_punisher_selections_per_ep]

            tpdps_legend_list = [
                "TPP-S",
                "TPP",
                "DP-S",
                "DP",
                "TPPDP-S",
                "TPPDP",
            ]

            tpdps_legend_list_select = [
                "TPP-S",
                "DP-S",
                "TPPDP-S",
            ]

            plotting_multiple(f"{title_prefix} Just Punishment Per Episode", punish_justly_comparisons, "Episodes",
                              "Just Punishment (%)", plotting_folder, tpdps_legend_list)

            plotting_multiple(f"{title_prefix} Punishment Per Episode", punish_comparisons, "Episodes",
                              "Punishment (%)", plotting_folder, tpdps_legend_list)

            plotting_multiple(f"{title_prefix} Just Punisher Selections Per Episode", just_punish_select_comparisons,
                              "Episodes", "Just Punisher Selections (%)", plotting_folder,
                              tpdps_legend_list_select)

            plotting_multiple(f"{title_prefix} Punisher Selections Per Episode", punish_select_comparisons,
                              "Episodes", "Punisher Selections (%)", plotting_folder, tpdps_legend_list_select)

        else:
            mean_percentage_tpp_s_punisher_selections_per_ep = find_punish_select_per_ep(agent_stats_list[0], NUM_RUNS,
                                                                                         NUM_EPISODES)
            mean_percentage_direct_s_punisher_selections_per_ep = find_punish_select_per_ep(agent_stats_list[2],
                                                                                            NUM_RUNS,
                                                                                            NUM_EPISODES)
            mean_percentage_tpp_tppdps_punisher_selections_per_ep = find_tpp_punish_select_per_ep(agent_stats_list[4],
                                                                                                  NUM_RUNS,
                                                                                                  NUM_EPISODES)
            mean_percentage_direct_tppdps_punisher_selections_per_ep = find_direct_punish_select_per_ep(
                agent_stats_list[4],
                NUM_RUNS,
                NUM_EPISODES)

            mean_percentage_tpp_s_just_punisher_selections_per_ep = find_just_punish_select_per_ep(agent_stats_list[0],
                                                                                                   NUM_RUNS,
                                                                                                   NUM_EPISODES)

            mean_percentage_direct_s_just_punisher_selections_per_ep = find_just_punish_select_per_ep(
                agent_stats_list[2],
                NUM_RUNS,
                NUM_EPISODES)
            mean_percentage_tpp_tppdps_just_punisher_selections_per_ep = find_just_tpp_punish_select_per_ep(
                agent_stats_list[4],
                NUM_RUNS, NUM_EPISODES)
            mean_percentage_direct_tppdps_just_punisher_selections_per_ep = find_just_direct_punish_select_per_ep(
                agent_stats_list[4],
                NUM_RUNS, NUM_EPISODES)

            mean_percentage_tpp_s_punishment_per_ep = find_punishment_per_ep(agent_stats_list[0], NUM_RUNS,
                                                                             NUM_EPISODES)
            mean_percentage_tpp_punishment_per_ep = find_punishment_per_ep(agent_stats_list[1], NUM_RUNS, NUM_EPISODES)
            mean_percentage_direct_s_punishment_per_ep = find_punishment_per_ep(agent_stats_list[2], NUM_RUNS,
                                                                                NUM_EPISODES)
            mean_percentage_direct_punishment_per_ep = find_punishment_per_ep(agent_stats_list[3], NUM_RUNS,
                                                                              NUM_EPISODES)

            mean_percentage_tpp_tppdps_punishment_per_ep = find_tpp_punishment_per_ep(agent_stats_list[4], NUM_RUNS,
                                                                                      NUM_EPISODES)
            mean_percentage_direct_tppdps_punishment_per_ep = find_direct_punishment_per_ep(agent_stats_list[4],
                                                                                            NUM_RUNS,
                                                                                            NUM_EPISODES)

            mean_percentage_tpp_tppdp_punishment_per_ep = find_tpp_punishment_per_ep(agent_stats_list[5], NUM_RUNS,
                                                                                     NUM_EPISODES)
            mean_percentage_direct_tppdp_punishment_per_ep = find_direct_punishment_per_ep(agent_stats_list[5],
                                                                                           NUM_RUNS,
                                                                                           NUM_EPISODES)

            mean_percentage_tpp_s_punish_justly_per_ep = find_punish_justly_per_ep(agent_stats_list[0], NUM_RUNS,
                                                                                   NUM_EPISODES)
            mean_percentage_tpp_punish_justly_per_ep = find_punish_justly_per_ep(agent_stats_list[1], NUM_RUNS,
                                                                                 NUM_EPISODES)
            mean_percentage_direct_s_punish_justly_per_ep = find_punish_justly_per_ep(agent_stats_list[2], NUM_RUNS,
                                                                                      NUM_EPISODES)
            mean_percentage_direct_punish_justly_per_ep = find_punish_justly_per_ep(agent_stats_list[3], NUM_RUNS,
                                                                                    NUM_EPISODES)
            mean_percentage_tpp_tppdps_punish_justly_per_ep = find_tpp_punish_justly_per_ep(agent_stats_list[4],
                                                                                            NUM_RUNS,
                                                                                            NUM_EPISODES)
            mean_percentage_direct_tppdps_punish_justly_per_ep = find_direct_punish_justly_per_ep(agent_stats_list[4],
                                                                                                  NUM_RUNS,
                                                                                                  NUM_EPISODES)

            mean_percentage_tpp_tppdp_punish_justly_per_ep = find_tpp_punish_justly_per_ep(agent_stats_list[5],
                                                                                           NUM_RUNS,
                                                                                           NUM_EPISODES)
            mean_percentage_direct_tppdp_punish_justly_per_ep = find_direct_punish_justly_per_ep(agent_stats_list[5],
                                                                                                 NUM_RUNS,
                                                                                                 NUM_EPISODES)

            punish_justly_comparisons = [mean_percentage_tpp_s_punish_justly_per_ep,
                                         mean_percentage_tpp_punish_justly_per_ep,
                                         mean_percentage_direct_s_punish_justly_per_ep,
                                         mean_percentage_direct_punish_justly_per_ep,
                                         mean_percentage_tpp_tppdps_punish_justly_per_ep,
                                         mean_percentage_direct_tppdps_punish_justly_per_ep,
                                         mean_percentage_tpp_tppdp_punish_justly_per_ep,
                                         mean_percentage_direct_tppdp_punish_justly_per_ep]

            punish_comparisons = [mean_percentage_tpp_s_punishment_per_ep,
                                  mean_percentage_tpp_punishment_per_ep,
                                  mean_percentage_direct_s_punishment_per_ep,
                                  mean_percentage_direct_punishment_per_ep,
                                  mean_percentage_tpp_tppdps_punishment_per_ep,
                                  mean_percentage_direct_tppdps_punishment_per_ep,
                                  mean_percentage_tpp_tppdp_punishment_per_ep,
                                  mean_percentage_direct_tppdp_punishment_per_ep]

            just_punish_select_comparisons = [mean_percentage_tpp_s_just_punisher_selections_per_ep,
                                              mean_percentage_direct_s_just_punisher_selections_per_ep,
                                              mean_percentage_tpp_tppdps_just_punisher_selections_per_ep,
                                              mean_percentage_direct_tppdps_just_punisher_selections_per_ep]

            punish_select_comparisons = [mean_percentage_tpp_s_punisher_selections_per_ep,
                                         mean_percentage_direct_s_punisher_selections_per_ep,
                                         mean_percentage_tpp_tppdps_punisher_selections_per_ep,
                                         mean_percentage_direct_tppdps_punisher_selections_per_ep]

            tpdps_legend_list = [
                "TPP-S",
                "TPP",
                "DP-S",
                "DP",
                "TPP in TPPDP-S",
                "DP in TPPDP-S",
                "TPP in TPPDP",
                "DP in TPPDP",
            ]

            tpdps_legend_list_select = [
                "TPP-S",
                "DP-S",
                "TPP in TPPDP-S",
                "DP in TPPDP-S",
            ]

            plotting_multiple(f"{title_prefix} Just Punishment Per Episode", punish_justly_comparisons, "Episodes",
                              "Just Punishment (%)", plotting_folder, tpdps_legend_list)

            plotting_multiple(f"{title_prefix} Punishment Per Episode", punish_comparisons, "Episodes",
                              "Punishment (%)", plotting_folder, tpdps_legend_list)

            plotting_multiple(f"{title_prefix} Just Punisher Selections Per Episode", just_punish_select_comparisons,
                              "Episodes", "Just Punisher Selections (%)", plotting_folder,
                              tpdps_legend_list_select)

            plotting_multiple(f"{title_prefix} Punisher Selections Per Episode", punish_select_comparisons,
                              "Episodes", "Punisher Selections (%)", plotting_folder, tpdps_legend_list_select)

    plotting_multiple(f"{title_prefix} Cooperator Selections Per Episode", comparing_cooperation_selections_list,
                      "Episodes",
                      "Cooperator Selections (%)", plotting_folder, np.array(legend_list)[np.array(selection)])

    if not tpdps:
        plotting_multiple(f"{title_prefix} Punisher Selections Per Episode", comparing_punisher_selections_list,
                          "Episodes",
                          "Punisher Selections (%)", plotting_folder,
                          np.array(legend_list)[np.array(selection and punish_selection)])

        plotting_multiple(f"{title_prefix} Just Punisher Selections Per Episode",
                          comparing_just_punisher_selections_list,
                          "Episodes",
                          "Just Punisher Selections (%)", plotting_folder,
                          np.array(legend_list)[np.array(selection and punish_selection)])

    plotting_multiple(f"{title_prefix} Cooperation Per Episode", comparing_cooperation_list, "Episodes",
                      "Cooperation (%)", plotting_folder, legend_list)

    if not tpdps:
        plotting_multiple(f"{title_prefix} Punishment Per Episode", comparing_punishment_list, "Episodes",
                          "Punishment (%)", plotting_folder, legend_list)

        plotting_multiple(f"{title_prefix} Just Punishment Per Episode", comparing_just_punishment_list, "Episodes",
                          "Just Punishment (%)", plotting_folder, legend_list)

    plotting_multiple(f"{title_prefix} Combined Reward Per Episode", comparing_reward_list, "Episodes",
                      "Combined Reward", plotting_folder, legend_list)

    plotting_multiple(f"{title_prefix} Combined Reputation Per Episode", comparing_reputation_list, "Episodes",
                      "Combined Reputation", plotting_folder, legend_list)

    plotting_multiple(f"{title_prefix} Combined Centipede Play Length Per Episode", comparing_reputation_list,
                      "Episodes",
                      "Combined Centipede Play Length", plotting_folder, legend_list)