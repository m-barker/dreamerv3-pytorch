from typing import Tuple, Dict, Optional, Union

import torch
import gymnasium as gym
from gymnasium.wrappers import RescaleAction, ClipAction
import numpy as np
import shimmy  # import needed to trigger the registrations of dm environments

from dreamer.utils.utils import resize_image


class DMCWrapper:
    def __init__(
        self,
        task_name: str,
        image_res: Tuple[int, int],
        seed: int,
        return_high_res_img: bool = False,
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Args:
            task_name (str): name of the dmc task. Used in gym.make as 'dmc_control/{task_name}'

            image_res (Tuple[int, int]): height, width of the rgb observation to return. Resizes if
            neccessary.

            seed (int): environment seed.

            return_high_res_image (bool, optional): whether the original, high-res rgb observation
            should be returned. Useful for evaluation. Defaults to False.

            max_steps (int, optional): optional maximum number of steps before environment truncates.
            defaults to None.
        """

        self._task_name = task_name
        self._image_res = image_res
        self._seed = seed
        self._return_high_res_image = return_high_res_img

        self._env = gym.make(f"dm_control/{self._task_name}", render_mode="rgb_array")
        self._env = RescaleAction(self._env, min_action=-1.0, max_action=1.0)
        self._env = ClipAction(self._env)
        self._step_count = 0
        self._max_steps = max_steps

    @property
    def observation_space(self) -> gym.spaces.Dict:
        # H,W,C
        image_size = self._image_res + (3,)
        return gym.spaces.Dict(
            {"image": gym.spaces.Box(0, 255, image_size, dtype=np.uint8)}
        )

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self) -> Tuple[Dict, Dict]:
        """Resets the environment using the seed passed into the constructor

        Returns:
            Tuple[Dict, Dict] obs, info
        """
        obs, info = self._env.reset(seed=self._seed)
        image_obs = self._env.render()

        resized_img = image_obs
        if image_obs.shape[:-1] != self._image_res:
            resized_img = resize_image(image_obs, self._image_res)

        self._step_count = 0

        observation = {
            "image": resized_img,
        }
        if self._return_high_res_image:
            observation["high_res_image"] = image_obs

        return observation, info

    def step(
        self, action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Takes a step in the environment using the provided action.

        Args:
            action
        """
        # Try to squeeze off any singular batch dimensions
        if action.shape != self._env.action_space.shape:
            action = action.squeeze()

        assert action.shape == self._env.action_space.shape, (
            f"Given action shape {action.shape} does not match required environment shape: {self._env.action_space.shape}"
        )
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1

        if self._max_steps is not None:
            if self._step_count >= self._max_steps:
                truncated = True

        image_obs = self._env.render()

        resized_img = image_obs
        if image_obs.shape[:-1] != self._image_res:
            resized_img = resize_image(image_obs, self._image_res)

        observation = {
            "image": resized_img,
        }
        if self._return_high_res_image:
            observation["high_res_image"] = image_obs

        return observation, reward, terminated, truncated, info
