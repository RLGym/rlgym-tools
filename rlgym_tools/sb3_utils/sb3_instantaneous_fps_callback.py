"""
A module containing an accurate instantaneous fps logging callback
"""

from time import time

from stable_baselines3.common.callbacks import BaseCallback


class SB3InstantaneousFPSCallback(BaseCallback):
  
  def __init__(self, average_over_n_rollouts:int=5):
    if average_over_n_rollouts < 1:
      raise ValueError("average_over_n_rollouts must be greater than 0")
    super().__init__()
    self.last_steps = []
    self.last_times = []
    self.capacity = average_over_n_rollouts + 1

  def _on_step(self):
    return True

  def _init_callback(self):
    self.last_steps.append(self.model.num_timesteps)
    self.last_times.append(time())

  def _on_rollout_end(self):
    if len(self.last_steps) == self.capacity:
      self.last_steps.pop(0)
      self.last_times.pop(0)
    self.last_steps.append(self.model.num_timesteps)
    self.last_times.append(time())
    
    fps = (self.last_steps[-1] - self.last_steps[0]) / (self.last_times[-1] - self.last_times[0])

    self.model.logger.record('time/instantaneous_fps', fps)
    
