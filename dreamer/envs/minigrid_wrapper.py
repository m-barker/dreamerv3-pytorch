from typing import Tuple, Dict, Optional, Union

from gymnasium.spaces.discrete import Discrete
import torch
import gymnasium as gym
import numpy as np
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper

from dreamer.utils.utils import resize_image


class MiniGridFullObsWrapper:
    def __init__(
        self,
        task_name: str,
        image_res: Tuple[int, int],
        seed: int,
        tile_size: int = 8,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = "rgb_array",
    ) -> None:
        """
        Args:
            task_name (str): MiniGrid task name.
            image_res (Tuple[int, int]): Target (H, W) for Dreamer.
            seed (int): Environment seed.
            tile_size (int): Resolution of each grid cell. Default 8
                            gives a 64x64 image for an 8x8 grid.
            max_steps (int): Optional manual truncation.
            render_mode (optional, str) optional render mode. Defaults
            to RGB array.
        """
        self._image_res = image_res
        self._seed = seed
        self._max_steps = max_steps
        self._step_count = 0

        # 1. Create the base environment
        env = gym.make(task_name, render_mode=render_mode)

        # 2. Make the observation include the full grid (no partial view)
        env = FullyObsWrapper(env)

        # 3. Convert that full symbolic grid into an RGB pixel image
        # This puts the image in obs["image"]
        self._env = RGBImgObsWrapper(env, tile_size=tile_size)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        image_size = self._image_res + (3,)
        return gym.spaces.Dict(
            {"image": gym.spaces.Box(0, 255, image_size, dtype=np.uint8)}
        )

    @property
    def action_space(self):
        return self._env.action_space

    def _process_obs(self, obs: Dict) -> Dict:
        # The RGBImgObsWrapper already provides pixels in obs["image"]
        image_obs = obs["image"]
        high_res_obs = obs["image"]

        if image_obs.shape[:-1] != self._image_res:
            image_obs = resize_image(image_obs, self._image_res)

        return {"image": image_obs, "high_res_image": high_res_obs}

    def reset(self) -> Tuple[Dict, Dict]:
        obs, info = self._env.reset(seed=self._seed)
        self._step_count = 0
        return self._process_obs(obs), info

    def step(
        self, action: Union[np.ndarray, torch.Tensor, int]
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        # Handle tensor/numpy discrete actions
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy().squeeze()
            action = action.argmax()
        if isinstance(action, np.ndarray):
            action = action.squeeze().argmax()

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1

        if self._max_steps is not None and self._step_count >= self._max_steps:
            truncated = True

        return self._process_obs(obs), float(reward), terminated, truncated, info
