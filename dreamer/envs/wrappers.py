import numpy as np
import gymnasium as gym
from gym import spaces


class UnscaleAction(gym.ActionWrapper):
    """Unscales an action from range [a,b] back to range [env_min, env_max]

    Useful as often neural networks (including dreamer for continuous actions)
    used as actors output actions in a normalised range.

    """

    def __init__(self, env: gym.Env, min_scaled: float = -1.0, max_scaled: float = 1.0):
        super().__init__(env)

        if isinstance(env.action_space, spaces.Box):
            self._env_low = env.action_space.low
            self._env_high = env.action_space.high
        else:
            raise ValueError("Action space must be a Box to unscale")

        self._min_scaled = min_scaled
        self._max_scaled = max_scaled

        self.action_space = spaces.Box(
            low=np.ones_like(self._env_low) * self._min_scaled,
            high=np.ones_like(self._env_high) * self._max_scaled,
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Maps the scaled actions in range [self._min_scaled, self._max_scaled]
        back to the environmnet range [self._env_low, self._env_high]
        """
        action = np.asarray(action, dtype=np.float32)
        unscaled_action = (action - self._min_scaled) * (
            (self._env_high - self._env_low) / self._max_scaled - self._min_scaled
        ) + self._env_low
        return np.clip(unscaled_action, self._env_low, self._env_high)
