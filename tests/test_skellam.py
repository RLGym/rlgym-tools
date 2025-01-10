import numpy as np
from matplotlib import pyplot as plt

from rlgym_tools.rocket_league.math.skellam import GOALS_PER_MINUTE, win_prob


def plot():
    xs = np.arange(0, 300) / 60

    fig, axs = plt.subplots(nrows=len(GOALS_PER_MINUTE), sharex=True, sharey=True, figsize=(10, 10))

    for i, goals_per_minute in enumerate(GOALS_PER_MINUTE):
        goal_prob = 0.5
        goal_diff = 1
        for goal_diff in range(1, 3):
            for goal_prob in (0.0, 0.1, 0.4, 0.5, 0.6, 0.9, 1.0):
                ys = [win_prob(GOALS_PER_MINUTE[i], x * 60, goal_diff, goal_prob) for x in xs]
                axs[i].plot(xs, ys, label=f"diff={goal_diff},prob={goal_prob}", linestyle='dashed' if goal_diff == 2 else 'solid')
        # for goal_diff in range(6):
        #     ys = [(2 * win_prob_scalar(GOALS_PER_MINUTE[i], x * 60, -goal_diff, goal_prob) - 1) / x for x in xs]
        #     axs[i].plot(xs, ys, label=f"Goal diff: {goal_diff}")
        axs[i].set_xlim([xs.max(), xs.min()])
        axs[i].set_ylim([0.5, 1.0])
        axs[i].set_title(f"{i + 1}v{i + 1}")
        axs[i].set_xlabel("Minutes Remaining")
        axs[i].set_ylabel("Win Probability")
        axs[i].grid()
        axs[i].legend()
    fig.tight_layout()
    plt.show()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    plot()
