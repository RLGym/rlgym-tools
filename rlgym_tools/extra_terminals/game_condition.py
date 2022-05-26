from rlgym.utils import TerminalCondition
from rlgym.utils.gamestates import GameState


class GameCondition(TerminalCondition):  # Mimics a Rocket League game
    def __init__(self, tick_skip=8, seconds_left=300, forfeit_spg_limit=None, max_overtime=300):
        # NOTE: Since game isn't reset to kickoff by default,
        # you need to keep this outside the main loop as well,
        # checking the done variable after each GoalTerminal
        super().__init__()
        self.tick_skip = tick_skip
        self.seconds_left = seconds_left
        self.timer = seconds_left
        self.overtime = False
        self.done = True
        self.initial_state = None
        self.forfeit_spg_limit = forfeit_spg_limit  # SPG = Seconds Per Goal
        self.max_overtime = max_overtime

    def reset(self, initial_state: GameState):
        if self.done:
            self.timer = self.seconds_left
            self.initial_state = initial_state
            self.overtime = False
            self.done = False

    def is_terminal(self, current_state: GameState) -> bool:
        differential = (current_state.blue_score - self.initial_state.blue_score) \
                       - (current_state.orange_score - self.initial_state.orange_score)
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
                    self.done = False
            elif self.forfeit_spg_limit is not None \
                    and abs(differential) >= 3 \
                    and self.timer / abs(differential) < self.forfeit_spg_limit:
                self.done = True
            else:
                self.timer -= self.tick_skip / 120

        return self.done
