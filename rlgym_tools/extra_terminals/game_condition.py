from rlgym.utils import TerminalCondition
from rlgym.utils.gamestates import GameState


class GameCondition(TerminalCondition):  # Mimics a Rocket League game
    def __init__(self, tick_skip=8, seconds_left=300, seconds_per_goal_forfeit=None,
                 max_overtime_seconds=float("inf"), max_no_touch_seconds=float("inf")):
        # NOTE: Since game isn't reset to kickoff by default,
        # you need to keep this outside the main loop as well,
        # checking the done variable after each terminal
        super().__init__()
        self.tick_skip = tick_skip
        self.seconds_left = seconds_left
        self.timer = seconds_left
        self.overtime = False
        self.done = True
        self.initial_state = None
        self.seconds_per_goal_forfeit = seconds_per_goal_forfeit  # SPG = Seconds Per Goal
        self.max_overtime = max_overtime_seconds
        self.max_no_touch = max_no_touch_seconds
        self.last_touch = None
        self.differential = None

    def reset(self, initial_state: GameState):
        if self.done:  # New game
            self.timer = self.seconds_left
            self.initial_state = initial_state
            self.overtime = False
            self.done = False
            self.last_touch = self.seconds_left
            self.differential = 0

    def is_terminal(self, current_state: GameState) -> bool:
        reset = False

        differential = ((current_state.blue_score - self.initial_state.blue_score)
                        - (current_state.orange_score - self.initial_state.orange_score))

        if differential != self.differential:  # Goal scored
            reset = True

        if self.overtime:
            if differential != 0:
                self.done = True
            elif self.timer >= self.max_overtime:
                self.done = True  # Call it a draw
            else:
                self.timer += self.tick_skip / 120
                self.done = False
        else:
            if self.timer <= 0 and current_state.ball.position[2] <= 110:
                # Can't detect ball on ground directly, should be an alright approximation.
                # Anything below z vel of ~690uu/s should be detected. 50% for 1380 etc.
                if differential != 0:
                    self.done = True
                else:
                    self.overtime = True
                    self.last_touch = self.timer  # Just for convenience
                    self.done = False
                    reset = True
            elif (self.seconds_per_goal_forfeit is not None
                  and abs(differential) >= 3
                  and self.timer / abs(differential) < self.seconds_per_goal_forfeit):
                # Too few seconds per goal to realistically win
                self.done = True
            else:
                self.timer -= self.tick_skip / 120

        if abs(self.last_touch - self.timer) >= self.max_no_touch:
            self.done = True
        elif any(p.ball_touched for p in current_state.players):
            self.last_touch = self.timer

        self.differential = differential

        return reset or self.done
