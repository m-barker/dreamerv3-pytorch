from typing import Tuple, Dict, Union
import gymnasium as gym
import torch
import numpy as np
import crafter


class CrafterWrapper:
    def __init__(self, task: str, image_res: Tuple[int, int], seed: int) -> None:
        assert task in ("reward", "noreward"), f"Invalid task {task}"

        self._env = crafter.Env(size=image_res, reward=(task == "reward"), seed=seed)
        self._seed = seed
        self._image_res = image_res

    @property
    def observation_space(self) -> gym.spaces.Dict:
        image_size = self._image_res + (3,)
        return gym.spaces.Dict(
            {"image": gym.spaces.Box(0, 255, image_size, dtype=np.uint8)}
        )

    @property
    def action_space(self):
        return gym.spaces.Discrete(self._env.action_space.n)

    def reset(self) -> Tuple[Dict, Dict]:
        # Crafter uses old gymnasium that only returns obs on reset
        # not obs, info
        obs = self._env.reset()
        return {"image": obs}, {}

    def step(
        self, action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        # Handle tensor/numpy discrete actions
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy().squeeze()
            action = action.argmax()
        if isinstance(action, np.ndarray):
            action = action.squeeze().argmax()
        # Crafter uses old gymnasium
        obs, reward, done, info = self._env.step(action)

        obs = {"image": obs}
        terminated = info["discount"] == 0
        truncated = bool(done and (not terminated))

        return obs, float(reward), terminated, truncated, info
