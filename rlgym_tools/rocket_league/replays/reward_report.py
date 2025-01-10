import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rlgym.api import RewardFunction, SharedInfoProvider

from rlgym_tools.rocket_league.math.gamma import half_life_to_gamma
from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay


def get_reward_df(replay: ParsedReplay, reward_functions: List[Tuple[RewardFunction, str]],
                  shared_info_provider: Optional[SharedInfoProvider] = None):
    """
    Get a pandas dataframe of rewards for each player and reward function.

    :param replay: The replay to get rewards from.
    :param reward_functions: A list of tuples of reward functions and tags to separate them.
    :param shared_info_provider: A shared info provider to provide additional information to the reward functions.
    :return: A tuple of two dataframes: one for metadata (game time and frame number) and one for all the rewards.
    """
    reward_functions = {
        (reward_fn.__class__.__name__ + tag
         if not hasattr(reward_fn, "reward_fn")
         else reward_fn.reward_fn.__class__.__name__ + tag): reward_fn
        for reward_fn, tag in reward_functions
    }

    pid_to_name = {int(p["unique_id"]): p["name"]
                   for p in replay.metadata["players"]
                   if p["unique_id"] in replay.player_dfs}

    print(pid_to_name)

    # To be put into a pandas dataframe
    data = {
        "game_time": [],
        "frame_number": [],
        **{
            f"{player_name}/{reward_name}": []
            for reward_name in reward_functions.keys()
            for player_name in pid_to_name.values()
        }
    }

    mean_delta = replay.game_df["delta"].mean()
    shared_info = shared_info_provider.create({}) if shared_info_provider is not None else {}
    iterator = replay_to_rlgym(replay)
    reset = True
    for frame_number, replay_frame in enumerate(iterator):
        game_state = replay_frame.state
        agent_ids = sorted(game_state.cars.keys())

        if reset:
            if shared_info_provider is not None:
                shared_info_provider.set_state(agent_ids, game_state, shared_info)
            for reward_fn in reward_functions.values():
                reward_fn.reset(agent_ids, game_state, shared_info)
            reset = False

        if replay_frame.scoreboard.go_to_kickoff or replay_frame.scoreboard.is_over:
            reset = True

        terminated = {agent_id: reset for agent_id in agent_ids}
        truncated = {agent_id: False for agent_id in agent_ids}

        timer = replay_frame.scoreboard.game_timer_seconds
        if math.isinf(timer):
            prev_time = data["game_time"][-1] if data["game_time"] else 0
            data["game_time"].append(prev_time - mean_delta)
        else:
            data["game_time"].append(timer)
        data["frame_number"].append(frame_number)

        if shared_info_provider is not None:
            shared_info_provider.step(agent_ids, game_state, shared_info)
        shared_info["scoreboard"] = replay_frame.scoreboard
        for reward_name, reward_fn in reward_functions.items():
            reward = reward_fn.get_rewards(agent_ids, game_state, terminated, truncated, shared_info)
            for agent_id, r in reward.items():
                player_name = pid_to_name[agent_id]
                data[f"{player_name}/{reward_name}"].append(r)

    data = pd.DataFrame(data)

    meta = data[['game_time', 'frame_number']]
    multicol = data.loc[:, data.columns.str.contains("/")]
    # Make columns multiindex (player_name, reward_name)
    multicol.columns = pd.MultiIndex.from_tuples(
        [tuple(col.split("/", maxsplit=1)) for col in multicol.columns]
    )

    return meta, multicol


def generate_report(meta: pd.DataFrame, rewards: pd.DataFrame, per_player: bool = False):
    """
    Use the metadata and rewards dataframes to generate a report on the rewards.

    :param meta: The metadata dataframe.
    :param rewards: The rewards dataframe.
    :param per_player: Whether to generate a report for each player.
    :return:
    """
    if per_player:
        for player in rewards.columns.levels[0]:
            player_rewards = rewards[player]
            print(player)
            generate_report(meta, player_rewards)
            print()
    for reward_name in rewards.columns.levels[1]:
        reward_df = rewards.xs(reward_name, axis=1, level=1)

        # Flatten
        stacked = pd.concat([reward_df[col] for col in reward_df.columns], axis=0)
        num_players = len(stacked) // len(meta)
        times = np.repeat(meta['game_time'].values, num_players)

        print(reward_name)

        # First, some basic stats
        game_time = meta['game_time'].max() - meta['game_time'].min()
        t_tot = num_players * game_time
        print(f"\tPlayer count: {num_players}")
        print(f"\tGame length: {game_time:.2f} seconds")
        print("\tReward:")
        print(f"\t\tAverage (per-second): {stacked.sum() / t_tot:.3g}")
        print(f"\t\tMax: {stacked.max():.3g}")
        print(f"\t\tMin: {stacked.min():.3g}")

        print()

        # Next, some more intricate stats
        nonzero = stacked[stacked != 0]
        c = nonzero.count()
        density = c / stacked.count()
        unique = np.unique(nonzero.values)
        print("\tClassification:", end=" ")
        if density < 0.5:
            # sparsity = 1 - density
            regional_density = 1 - np.diff(stacked != 0).sum() / c
            density_str = f"{density:.1%}"
            if density_str == "0.0%":
                density_str = f"<0.1% ({c} in {stacked.count()})"
            if c < 10:
                print(f"sparse, with density={density_str}")
            elif regional_density > 0.5:
                print(f"regionally dense, with density={density_str} "
                      f"and regional density={regional_density:.3g})")
            else:
                print(f"sparse, with density={density_str}")
        else:
            print(f"dense, with density={density:.1%}")

        print()

        print("\tCount of non-zero rewards:", c)
        print("\tAverage of non-zero rewards:", nonzero.mean())
        print("\tCount of positive rewards:", (nonzero > 0).sum())
        print("\tAverage of positive rewards:", nonzero[nonzero > 0].mean())
        print("\tCount of negative rewards:", (nonzero < 0).sum())
        print("\tAverage of negative rewards:", nonzero[nonzero < 0].mean())
        if len(unique) <= 5 and c > len(unique):
            print("\tUnique values:")
            for val in unique:
                if np.isnan(val):
                    continue
                count = (nonzero == val).sum()
                print(f"\t\t{val}: {count} occurrences")

        print()


def generate_plot(meta_df, reward_df, gamma=None):
    """
    Generate a plot of the rewards for each player.

    :param meta_df: The metadata dataframe.
    :param reward_df: The rewards dataframe.
    :param gamma: The discount factor for the discounted future sum.
    :return:
    """
    if gamma is None:
        gamma = half_life_to_gamma(30, 4)
    xs = meta_df["game_time"]
    # Make a plot of each reward for each player
    fig, axs = plt.subplots(nrows=len(reward_df.columns.levels[1]),
                            ncols=len(reward_df.columns.levels[0]),
                            figsize=(10 * len(reward_df.columns.levels[0]), 5 * len(reward_df.columns.levels[1])),
                            sharey="row")
    goals = np.where(reward_df[reward_df.columns.levels[0][0], "EpisodeEnd"] != 0)[0]
    edges = sorted(set([0] + (goals + 1).tolist() + [len(xs)]))
    for i, player in enumerate(reward_df.columns.levels[0]):
        for j, reward in enumerate(reward_df.columns.levels[1]):
            ax = axs[j, i]
            # Cumulative reward
            tot = pd.concat([reward_df[player, reward].iloc[start:end].cumsum()
                             for start, end in zip(edges, edges[1:])]).values
            ax.plot(xs, tot, label="cumulative")
            # Gamma discounted cumulative reward
            disc_sum = pd.concat([reward_df[player, reward].iloc[start:end][::-1].ewm(alpha=1 - gamma).sum()[::-1]
                                  for start, end in zip(edges, edges[1:])]).values
            ax.plot(xs, disc_sum, linestyle="--", label="discounted future sum")
            # Vertical lines for goals
            for g, goal in enumerate(goals):
                ax.axvline(xs[goal], color="red", linestyle="--", alpha=0.5, label="goal" if g == 0 else None)
            ax.legend()
            ax.set_ylabel("Cumulative reward")
            ax.set_title(f"{player}/{reward}")
            ax.grid(True)

            # Invert x-axis for time
            ax.invert_xaxis()
            ax.set_xlabel("Time (s)")

            # Make secondary x-axis tick labels that show mm:ss
            a_sec = ax.secondary_xaxis("top")
            a_sec.set_xlabel("Time (mm:ss)")
            a_sec.set_xticks(ax.get_xticks())
            a_sec.set_xticklabels([f"{int(t // 60):02d}:{int(t % 60):02d}" for t in ax.get_xticks()])
    fig.tight_layout()
    return fig
