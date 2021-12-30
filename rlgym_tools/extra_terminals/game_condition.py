from rlgym.utils import TerminalCondition
from rlgym.utils.gamestates import GameState


class GameCondition(TerminalCondition):  # Mimics a Rocket League game
    def __init__(self, tick_skip=8, seconds_left=300):
        # NOTE: Since game isn't reset to kickoff by default,
        # you need to keep this outside the main loop as well,
        # checking the done variable after each GoalTerminal
        super().__init__()
        self.tick_skip = tick_skip
        self.timer = seconds_left
        self.overtime = False
        self.done = True
        self.initial_state = None

    def reset(self, initial_state: GameState):
        if self.done:
            self.timer = 300 * 120 / self.tick_skip
            self.initial_state = initial_state
            self.overtime = False
            self.done = False

    def is_terminal(self, current_state: GameState) -> bool:
        differential = (current_state.blue_score - self.initial_state.blue_score) \
                       - (current_state.orange_score - self.initial_state.orange_score)
        if self.overtime:
            if differential != 0:
                self.done = True
            else:
                self.timer += self.tick_skip / 120
                self.done = False
        else:
            if self.timer <= 0 and current_state.ball.position[3] <= 100:
                # Can't detect ball on ground directly, should be an alright approximation
                if differential != 0:
                    self.done = True
                else:
                    self.overtime = True
                    self.done = False
        self.timer -= self.tick_skip / 120
        return self.done
