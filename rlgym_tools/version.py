# From https://github.com/RLBot/RLBot/blob/master/src/main/python/rlbot/version.py
# Store the version here so:
# 1) we don't load dependencies by storing it in __init__.py
# 2) we can import it in setup.py for the same reason
# 3) we can import it into your module module
# https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package

__version__ = '1.0.0'

release_notes = {
    '1.0.0': """
    - Added replay to rlgym GameState converter
    - Moved SB3 environments from rlgym (now called SB3SingleInstanceEnv and SB3MultipleInstanceEnv) and fixed some bugs
    - Added SB3MultiDiscreteWrapper, SB3DistributeRewardsWrapper and SB3LogReward 
    - Added extra reward functions (DiffReward, DistributeRewards and MultiplyRewards)
    - Added RLLibEnv
    - Added working example code for SB3 and rllib
    """
}


def get_current_release_notes():
    if __version__ in release_notes:
        return release_notes[__version__]
    return ''


def print_current_release_notes():
    print(f"Version {__version__}")
    print(get_current_release_notes())
    print("")
