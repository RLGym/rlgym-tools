# From https://github.com/RLBot/RLBot/blob/master/src/main/python/rlbot/version.py
# Store the version here so:
# 1) we don't load dependencies by storing it in __init__.py
# 2) we can import it in setup.py for the same reason
# 3) we can import it into your module module
# https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package

__version__ = '2.3.5'


release_notes = {
    '2.3.5': """
    - Fix bad pitch/yaw/roll error calculation in pick_action.py
    """,
    '2.3.4': """
    - Improve conditions in pick_action.py
    """,
    '2.3.3': """
    - Fix changing dictionary entries in ReplayFrame actions after yield.
    """,
    '2.3.2': """
    - In pick_action.py: better logic for when jump matters, and fix for throttle logic when no boost.
    """,
    '2.3.1': """
    - Fix flip reset obtain detection in FlipResetReward
    """,
    '2.3.0': """
    - Update RLGym requirement to >=2.0.1
    - Add AutoRewardNormalizer,SimpleZNormalizer and SQLiteNormalizer for automatically normalizing rewards
    - Add hitboxes.py containing a Hitbox dataclass and the standard in-game hitboxes
    - Add surface.py with a Surface enum methods for getting distance to different kinds of arena surfaces
    - Fix AugmentMutator not adjusting flip torque and autoflip drection
    - Fix DemoReward rewarding the same demo multiple times
    """,
    '2.2.5': """
    - Add new carball.exe
    """,
    '2.2.4': """
    - Fix dodge torque adjustment by normalizing torque in replay converter.
    """,
    '2.2.3': """
    - Fix pitch and yaw on dodges by normalizing torque in replay converter.
    """,
    '2.2.2': """
    - Fix RandomPhysicsMutator giving invalid rotation and disabling ball collision.
    """,
    '2.2.1': """
    - Fix error in RandomPhysicsMutator
    """,
    '2.2.0': """
    - Add V1 objects and V2 to V1 conversion functions.
    """,
    '2.1.2': """
    - Fix activation function in BoostKeepReward
    - Add checks to RandomPhysicsMutator to make sure we place cars and ball inside the field.
    - Use copying in AugmentMutator to circumvent a bug in base RLGym library when using KickoffMutator
    - Fix DelayedAction for delays longer than action length
    - Set delayed_actions in shared_info on reset in DelayedAction
    - Set go_to_kickoff and is_over in RandomScoreboardMutator
    """,
    '2.1.1': """
    - Improve accuracy of the replay to RLGym converter
    """,
    '2.1.0': """
    - Add reward reporting tools for inspecting rewards with replays
    - Add some convenience methods to the Action dataclass
    """,
    '2.0.3': """
        - Update MANIFEST.in to include carball.exe
    """,
    '2.0.2': """
       - Add carball.exe to package_data
    """,
    '2.0.1': """
       - Fix serialization (added wheel contacts and padded replay frame support, and tested with new script)
       - Fix ReplayMutator
       - Add RandomPhysicsMutator
    """,
    '2.0.0': """
       Move to RLGym v2:
       - Remove all v1 code
       - Add action parsers: ActionHistoryWrapper, AdvancedLookupTableAction, DelayedAction, QueuedAction
       - Add done conditions: BallHitGroundCondition, CarOnGroundCondition, GameCondition
       - Add math utils for ball, gamma, inverse aerial controls, relative physics, skellam distribution and solid angle
       - Add Action dataclass and serialization
       - Add obs builder: RelativeDefaultObs
       - Add renderer: RocketSimVisRenderer
       - Add new replay parser and replay converter to new ReplayFrame object
       - Add rewards: AdvancedTouchReward, AerialDistanceReward, BallTravelReward, BoostChangeReward, BoostKeepReward, 
                      DemoReward, EpisodeEndReward, FlipResetReward, GoalProbReward, StackReward, 
                      TeamSpiritRewardWrapper, VelocityPlayerToBallReward, WaveDashReward
       - Add shared info providers: BallPredictionProvider, MultiProvider, ScoreboardProvider, SerializedProvider
       - Add state mutators: AugmentMutator, ConfigMutator, GameMutator, HitboxMutator, RandomScoreboardMutator, 
                             ReplayMutator, VariableTeamSizeMutator, WeightedSampleMutator
       - Add transition engine: MultiEnvEngine
    """,
    '1.8.3': """
       - Update parse_actions in lookup_act.py (Jeff)
    """,
    '1.8.2': """
       - Fix no touch timer in GameCondition (Rolv)
       - Update RLLib example (Aech)
    """,
    '1.8.1': """
        - Refactor GameCondition (Rolv, Impossibum)
        - Fix a small mistake in LookupAction (Rolv)
    """,
    '1.8.0': """
        - Add lookup parser as used by Nexto/Tecko (Rolv)
        - Add customizable odds to WallPracticeState (Soren)
        - Add code for reducing SB3 model size for RLBot botpack, with example (DI3D)
        - Update AdvancedPadder for RLGym 1.2 (Kaiyotech)
        - Update example code for RLGym 1.2 (mstuettgen)
        - Fix AdvancedStacker (Some Rando)
        - Fix broken imports in SequentialRewards (Some Rando)
        - Fix bad indent in JumpTouchReward (Some Rando)
        """,
    '1.7.0':
        """
        - Add AdvancedObsPadder (Impossibum)
        - Add JumpTouchReward (Impossibum)
        - Fix NameError in KickoffReward (benjamincburns)
        - Add generate_probabilities as a method to ReplaySetter (Rolv)
        - Upgrade WallSetter (Soren) and fix bug when num_cars == 1 (Kaiyotech)
        - Add max overtime to GameCondition (Rolv)
        """,
    '1.6.6':
        """
        -WallStateSetter now has airdribble setups and harder wall plays
        """,
    '1.6.5':
        """
        -GoalieStateSetter now better emulates incoming airdribbles
        -fixed WallStateSetter bug and increased starting diversity
        """,
    '1.6.4':
        """
        -Added wall play state setter (Soren)
        """,
    '1.6.3':
        """
        - Fix hoops-like setter for multiple players (Carrot)
        """,
    '1.6.2':
        """
        - Added hoops-like setter (Carrot)
        - Fixed kickoff-like setter (Carrot)
        - Fixed pitch in KBMAction (Rolv)
        - Added include_frame option in replay converter (wrongu)
        """,
    '1.6.1':
        """
        - Fixed angular velocities in replay augmenter
        """,
    '1.6.0':
        """
        - Added GoaliePracticeState setter (Soren)
        - Added ReplaySetter (Carrot)
        - Added AugmentSetter (NeverCast)
        - Fixed an error in replay converter
        """,
    '1.5.3':
        """
        - Yet another fix for GameCondition
        """,
    '1.5.2':
        """
        - Another fix for GameCondition
        """,
    '1.5.1':
        """
        - Fix for GameCondition
        """,
    '1.5.0':
        """
        - Add SB3CombinedLogReward (LuminousLamp)
        - Add SB3InstantaneousFPSCallback (LuminousLamp)
        - Add KickoffLikeSetter (Carrot)
        - Add GameCondition (Rolv)
        """,
    '1.4.1':
        """
        - Remove SB3MultiDiscreteWrapper
        - Update SB3 examples to include action parsers
        """,
    '1.4.0':
        """
        - Add KBM action parser
        """,
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
    '1.1.0':
        """
        - Added functionality in SB3MultipleInstanceEnv to take multiple match objects (one for each instance).
        - Added AnnealRewards for transitioning between reward functions.
        - Added granularity option to SB3MultiDiscreteWrapper.
        - Added negative slope parameter to DiffReward (negative values are multiplied by this).
        """,
    '1.0.0':
        """
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
