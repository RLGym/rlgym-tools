# From https://github.com/RLBot/RLBot/blob/master/src/main/python/rlbot/version.py
# Store the version here so:
# 1) we don't load dependencies by storing it in __init__.py
# 2) we can import it in setup.py for the same reason
# 3) we can import it into your module module
# https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package

__version__ = '1.5.0'

release_notes = {
    '1.5.0': {
        """
        - Add SB3CombinedLogReward (LuminousLamp)
        - Add SB3InstantaneousFPSCallback (LuminousLamp)
        - Add KickoffLikeSetter (Carrot)
        - Add GameCondition (Rolv)
        """
    },
    '1.4.1': {
        """
        - Remove SB3MultiDiscreteWrapper
        - Update SB3 examples to include action parsers
        """
    },
    '1.4.0': {
        """
        - Add KBM action parser
        """
    },
    '1.3.0': 
    """
    - Add KickoffReward (Impossibum)
    - Add SequentialReward (Impossibum)
    - Better individual reward logging for SB3 (LuminousLamp)
    - Add launch_preference as parameter to SB3MultipleInstanceEnv
    """,
    '1.2.0':
    """
    - Added multi-model support for SB3
    - Added weighted sample setter
    - Added general stacker
    - Various bug fixes
    """,
    '1.1.4':
    """
    - Bug fixes for AdvancedStacker
    """,
    '1.1.3':
    """
    - Fix RLGym 1.0 incompatibility
    """,
    '1.1.2':
    """
    - Fix version import (was absolute, now relative)
    """,
    '1.1.1':
    """
    - Fixed an invalid Python 3.7 import
    - AnnealRewards now has initial_count parameter
    """,
    '1.1.0': """
    - Added functionality in SB3MultipleInstanceEnv to take multiple match objects (one for each instance).
    - Added AnnealRewards for transitioning between reward functions.
    - Added granularity option to SB3MultiDiscreteWrapper.
    - Added negative slope parameter to DiffReward (negative values are multiplied by this).
    """,
    '1.0.0': """
    - Added replay to rlgym GameState converter
    - Moved SB3 environments from rlgym (now called SB3SingleInstanceEnv and SB3MultipleInstanceEnv) and fixed some bugs
    - Added SB3MultiDiscreteWrapper, SB3DistributeRewardsWrapper and SB3LogReward 
    - Added extra reward functions (DiffReward, DistributeRewards and MultiplyRewards)
    - Added RLLibEnv
    - Added working example code for SB3 and RLlib
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
