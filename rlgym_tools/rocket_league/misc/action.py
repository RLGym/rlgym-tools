from dataclasses import dataclass

import numpy as np

INVERT_ACTION = np.array([1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)  # Invert steer, yaw and roll


@dataclass(slots=True)
class Action:
    """
    Dataclass representing an action in the environment.
    RLGym uses numpy arrays, but this class can be used in place of them for better readability,
    type hints and pretty printing.
    """
    throttle: float = 0.0
    steer: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    jump: bool = False
    boost: bool = False
    handbrake: bool = False

    @classmethod
    def from_numpy(cls, action: np.ndarray):
        return cls(*action[:5], *action[5:].astype(bool))

    def to_numpy(self):
        return np.array(
            (self.throttle, self.steer, self.pitch, self.yaw, self.roll, self.jump, self.boost, self.handbrake),
            dtype=np.float32
        )

    def mirror(self):
        return ~self

    def __invert__(self):
        # ~ operator mirrors the action
        return Action.from_numpy(self.to_numpy() * INVERT_ACTION)

    def __array__(self):
        # Make this class compatible with numpy functions
        return self.to_numpy()

    def __repr__(self):
        def format_bool(b):
            return int(b)

        def format_float(f):
            return f"{f:+.2f}"

        return (f"Action("
                f"throttle={format_float(self.throttle)}, "
                f"steer={format_float(self.steer)}, "
                f"pitch={format_float(self.pitch)}, "
                f"yaw={format_float(self.yaw)}, "
                f"roll={format_float(self.roll)}, "
                f"jump={format_bool(self.jump)}, "
                f"boost={format_bool(self.boost)}, "
                f"handbrake={format_bool(self.handbrake)}"
                f")")
