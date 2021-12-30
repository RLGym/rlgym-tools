from rlgym.utils import TerminalCondition
from rlgym.utils.gamestates import GameState


class GameCondition(TerminalCondition):  # Mimics a Rocket League game
    def __init__(self, tick_skip=8, seconds_left=300):
        super().__init__()
        self.tick_skip = tick_skip
        self.timer = seconds_left
        self.overtime = False
        self.initial_state = None

    def reset(self, initial_state: GameState):
        self.timer = 300 * 120 / self.tick_skip
        self.initial_state = initial_state

    def is_terminal(self, current_state: GameState) -> bool:
        differential = (current_state.blue_score - self.initial_state.blue_score) \
                       - (current_state.orange_score - self.initial_state.orange_score)
        if self.overtime:
            if differential != 0:
                return True
            else:
                self.timer += self.tick_skip / 120
                return False
        else:
            if self.timer <= 0 and current_state.ball.position[3] <= 100:
                # Can't detect ball on ground directly, should be an alright approximation
                if differential != 0:
                    return True
                else:
                    self.overtime = True
                    return False
        self.timer -= self.tick_skip / 120
        return False
