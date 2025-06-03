import numpy as np
from rlgym.rocket_league.api import Car


def filter_action_options(car: Car, replay_action: np.ndarray, action_options: np.ndarray, greedy_continuous: bool,
                          deadzone: float = 0.5):
    # Filters out actions that are definitely not correct

    candidates = action_options.copy()

    # There's a tiny chance it could be wrong if it lands holding jump and is supposed to jump the next step,
    # but doing it like this lets it use actions meant for flips to rotate in the air, giving greater precision.
    jump_matters = car.on_ground or car.is_jumping or car.has_flip
    boost_matters = car.boost_amount > 0
    is_grounded = car.on_ground and replay_action[5] == 0
    is_using_dodge = replay_action[5] == 1 and car.can_flip
    is_aerial = not is_grounded and not is_using_dodge

    if jump_matters:
        # Jump
        jump_error = np.abs(candidates[:, 5] - replay_action[5])
        candidates = candidates[jump_error == jump_error.min()]

        if is_using_dodge:
            # Check if it's a dodge or a double jump
            is_dodge = np.abs(replay_action[2:5]).sum() >= deadzone
            is_dodges = np.abs(candidates[:, 2:5]).sum(axis=1) >= deadzone  # Jump is already filtered
            dodge_error = np.abs(is_dodges - float(is_dodge))
            candidates = candidates[dodge_error == dodge_error.min()]

            if is_dodge:
                # Check that the dodge direction error is minimized
                dodge_dir = np.array([replay_action[2], replay_action[3] + replay_action[4]])
                dodge_dirs = np.array([candidates[:, 2], candidates[:, 3] + candidates[:, 4]]).T
                if np.all(dodge_dir == 0):
                    # Stall
                    dir_dot = np.all(dodge_dirs == 0, axis=1)
                else:
                    dodge_dir /= np.linalg.norm(dodge_dir)
                    m = np.any(dodge_dirs != 0, axis=1)
                    dodge_dirs[m] /= np.linalg.norm(dodge_dirs[m], axis=1, keepdims=True)
                    dir_dot = np.dot(dodge_dirs, dodge_dir)
                candidates = candidates[dir_dot == dir_dot.max()]
            else:
                is_aerial = True  # Optimize aerial direction for valid candidates

    if boost_matters:
        # Boost
        boost_error = np.abs(candidates[:, 6] - replay_action[6])
        candidates = candidates[boost_error == boost_error.min()]

    if is_grounded:  # Grounded
        # Handbrake
        handbrake_error = np.abs(candidates[:, 7] - replay_action[7])
        candidates = candidates[handbrake_error == handbrake_error.min()]

        # Throttle
        margin = 1e-3  # Below this is regarded as 0
        throttle = replay_action[0] if (replay_action[6] == 0 or not boost_matters) else 1
        if abs(throttle) >= margin:
            # There is a braking effect when opposing the current velocity that does not depend on magnitude
            throttle_dir = np.sign(throttle)
            throttle_dirs = np.sign(candidates[:, 0]) * (abs(candidates[:, 0]) >= margin)
            dir_error = throttle_dirs == -throttle_dir
            candidates = candidates[dir_error == dir_error.min()]

        if greedy_continuous:
            throttle_error = np.abs(candidates[:, 0] - throttle)
            candidates = candidates[throttle_error == throttle_error.min()]

            steer_error = np.abs(candidates[:, 1] - replay_action[1])
            candidates = candidates[steer_error == steer_error.min()]
    elif is_aerial:  # Aerial
        # Pitch, yaw, roll
        rotate_dir = replay_action[2:5]
        rotate_dirs = candidates[:, 2:5]
        if greedy_continuous:
            dir_error = np.linalg.norm(rotate_dirs - rotate_dir, axis=1)
        else:
            dir_dot = np.dot(rotate_dirs, rotate_dir)
            dir_error = dir_dot < 0
        candidates = candidates[dir_error == dir_error.min()]

    # Steer (on ground) and pitch/yaw/roll (in air) are ignored due to being continuous
    return candidates


def get_best_action_options(car: Car, replay_action: np.ndarray, action_options: np.ndarray, dodge_deadzone=0.5,
                            greedy=True) -> np.ndarray:
    """
    Produces a probability distribution over the action options, where the best options are given the highest
    probability.

    :param car: The car state
    :param replay_action: The action to match
    :param action_options: The available action options
    :param dodge_deadzone: The deadzone for dodges
    :param greedy: If True, after filtering candidates it picks the closest ones by Euclidean distance.
    :return: The probabilities for each action option
    """
    candidates = filter_action_options(car, replay_action, action_options, True, dodge_deadzone)

    if greedy:
        naive_error = np.linalg.norm(candidates - replay_action, axis=1)
        candidates = candidates[naive_error == naive_error.min()]

    probs = np.zeros(len(action_options), dtype=np.float32)
    for i, c in enumerate(candidates):
        probs[np.all(action_options == c, axis=1)] = 1
    probs /= probs.sum()

    return probs


def get_weighted_action_options(car: Car, replay_action: np.ndarray, action_options: np.ndarray, dodge_deadzone=0.5):
    """
    Produces weights for each action such that the weighted sum of the action options is as close as possible to
    replay_action, and that the sum of the weights is 1 (e.g. they represent probabilities).

    We want to combine only the two available values closest to the true value, e.g. if we can have 0.5 and 1,
    then 0.75 would be a 50/50 mix of the two.

    :param car: The car state
    :param replay_action: The action to match
    :param action_options: The available action options
    :param dodge_deadzone: The deadzone for dodges
    :return: The weights for each action option
    """
    # Mostly the same as get_best_action_option, but slightly more lenient with candidates,
    # so we can get closer with the weighted average
    margin = 0.01
    candidates = filter_action_options(car, replay_action, action_options, False, dodge_deadzone)

    button_weights = np.ones(8)

    def _validate_continuous(button):
        # Returns a boolean mask of valid candidates (e.g. between the bins in the action options)
        replay_val = replay_action[button]
        o = candidates[:, button]
        diff = np.abs(o - replay_val)
        if diff.min() <= margin:
            v = diff <= margin
        else:
            # Include the bin borders surrounding the replay value
            unique = np.unique(o)
            bins = np.digitize(o, unique)
            diff = bins - np.digitize(replay_val, unique)
            v = (diff == 0) | (diff == 1)
        return v

    is_grounded = car.on_ground and replay_action[5] == 0
    is_using_dodge = replay_action[5] == 1 and car.can_flip
    is_aerial = not is_grounded and not is_using_dodge

    if is_grounded:
        handbrake_error = np.abs(candidates[:, 7] - replay_action[7])
        candidates = candidates[handbrake_error == handbrake_error.min()]

        candidates = candidates[_validate_continuous(0)]  # Throttle
        candidates = candidates[_validate_continuous(1)]  # Steer
        button_weights[[2, 3, 4]] = 1e-2  # Basically telling it to only optimize this if it has options
        button_weights[[5, 6]] = 1e-4
    elif is_aerial:
        button_weights[[0, 1, 7]] = 1e-2
        button_weights[[5, 6]] = 1e-4
        candidates = candidates[_validate_continuous(2)]  # Pitch
        candidates = candidates[_validate_continuous(3)]  # Yaw
        candidates = candidates[_validate_continuous(4)]  # Roll

        if len(candidates) > 1 and np.all(candidates[:, [2, 3, 4]] == candidates[0, [2, 3, 4]]):
            candidates = candidates[_validate_continuous(0)]  # Throttle

    if len(candidates) == 1:
        weights = np.zeros(len(action_options))
        idx = np.where(np.all(action_options == candidates[0], axis=1))[0]
        weights[idx] = 1
        return weights

    # Perform least squares with sum(weights) = 1 constraint
    scale = 1000  # Scaling factor to control the importance of the constraint
    coefs = np.concatenate([(candidates * button_weights).T, scale * np.ones((1, len(candidates)))])
    target = np.concatenate([(replay_action * button_weights), [scale]])

    last_weights = np.zeros(len(candidates))
    while True:
        weights = np.linalg.lstsq(coefs, target, rcond=None)[0]
        valid = weights >= -1e-5  # Numerical instability can cause very small negative weights
        if valid.all() or np.allclose(weights, last_weights):
            break
        coefs[:, ~valid] = 0  # Set the vectors to 0, so they are effectively removed
        last_weights = weights

    weights = weights / weights.sum()  # It should be very close already but just to make sure
    smol = (0 != weights) & (weights < 0.01)
    while smol.any():
        weights[smol] = 0
        weights = weights / weights.sum()
        smol = (0 != weights) & (weights < 0.01)

    probs = np.zeros(len(action_options), dtype=np.float32)
    for i, c in enumerate(candidates):
        idx = np.where(np.all(action_options == c, axis=1))[0]
        probs[idx] = weights[i]

    return probs
