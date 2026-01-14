from typing import Tuple, Dict, Union
import gymnasium as gym
import numpy as np
import torch
import ale_py

from dreamer.utils.utils import resize_image

gym.register_envs(ale_py)


class AtariWrapper:
    def __init__(
        self,
        task_name: str,
        image_res: Tuple[int, int],
        seed: int,
        return_high_res_image: bool = False,
        obs_type: str = "rgb",
    ) -> None:
        """
        Args:
            task_name (str): name of the Atari environment, used in gym.make
            as "ALE/{task_name}

            image_res (Tuple[int, int]): desired observation image H,W

            seed (int): seed for randomness

            return_high_res_image (optional, bool): whether to return the
            higher res, un-resized image. Defaults to False.

            obs_type (str, optinoal): observation type. Can be either "rgb",
            "ram" or "grayscale". Defaults to rgb.

        """
        self._image_res = image_res
        self._seed = seed
        self._return_high_res = return_high_res_image

        self._env = gym.make(f"ALE/{task_name}", obs_type=obs_type)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        image_size = self._image_res + (3,)
        space = gym.spaces.Dict(
            {"image": gym.spaces.Box(0, 255, image_size, dtype=np.uint8)}
        )
        if self._return_high_res:
            space["high_res_image"] = self._env.observation_space
        return space

    @property
    def action_space(self):
        return self._env.action_space

    def _process_obs(self, rgb_obs: np.ndarray) -> Dict:
        """Resizes the rgb obs to self._image_res"""

        high_res_obs = rgb_obs
        image_obs = rgb_obs

        if image_obs.shape[:-1] != self._image_res:
            image_obs = resize_image(image_obs, self._image_res)

        obs = {"image": image_obs}
        if self._return_high_res:
            obs["high_res_image"] = high_res_obs
        return obs

    def reset(self) -> Tuple[Dict, Dict]:
        obs, info = self._env.reset(seed=self._seed)

        return self._process_obs(obs), info

    def step(
        self, action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy().squeeze()
            action = action.argmax()
        if isinstance(action, np.ndarray):
            action = action.squeeze().argmax()

        obs, reward, terminated, truncated, info = self._env.step(action)

        return self._process_obs(obs), float(reward), terminated, truncated, info
