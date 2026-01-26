from typing import Tuple, Dict, Union
import gymnasium as gym
import numpy as np
import torch
import ale_py
from PIL import Image
import collections

gym.register_envs(ale_py)


class AtariWrapper:
    def __init__(
        self,
        task_name: str,
        image_res: Tuple[int, int],
        seed: int,
        return_high_res_image: bool = False,
        obs_type: str = "rgb",
        sticky: bool = True,
        use_max_pooling: bool = True,
        use_noop_reset: bool = True,
    ) -> None:
        self._image_res = image_res
        self._seed = seed
        self._return_high_res = return_high_res_image
        self._use_max_pooling = use_max_pooling
        self._use_noop_reset = use_noop_reset

        repeat_prob = 0.25 if sticky else 0.0

        self._env = gym.make(
            f"ALE/{task_name}",
            obs_type=obs_type,
            frameskip=4,
            repeat_action_probability=repeat_prob,
            full_action_space=False,
        )

        self._rng = np.random.default_rng(seed)

        # frame pooling buffer (last 2 frames)
        if self._use_max_pooling:
            self._frames = collections.deque(maxlen=2)
        else:
            self._frames = None

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

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] == self._image_res:
            return image
        img = Image.fromarray(image)
        img = img.resize(self._image_res, Image.BILINEAR)
        return np.array(img)

    def _process_obs(self, obs: np.ndarray) -> Dict:
        if self._use_max_pooling:
            assert self._frames is not None
            # max-pool last 2 frames
            self._frames.append(obs)
            if len(self._frames) == 2:
                pooled = np.maximum(self._frames[0], self._frames[1])
            else:
                pooled = obs
        else:
            pooled = obs

        image = self._resize(pooled)

        out = {"image": image}
        if self._return_high_res:
            out["high_res_image"] = obs
        return out

    def reset(self) -> Tuple[Dict, Dict]:
        obs, info = self._env.reset(seed=self._seed)
        if self._use_max_pooling:
            assert self._frames is not None
            self._frames.clear()

        if self._use_noop_reset:
            # random no-op reset (0–30)
            noops = self._rng.integers(0, 31)
            for _ in range(noops):
                obs, _, terminated, truncated, _ = self._env.step(0)
                if terminated or truncated:
                    obs, info = self._env.reset()
                    if self._use_max_pooling:
                        assert self._frames is not None
                        self._frames.clear()
                    break

        if self._use_max_pooling:
            assert self._frames is not None
            self._frames.append(obs)

        return self._process_obs(obs), info

    def step(
        self, action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy().squeeze()
            action = action.argmax()
        elif isinstance(action, np.ndarray):
            action = action.squeeze().argmax()

        obs, reward, terminated, truncated, info = self._env.step(action)

        return self._process_obs(obs), float(reward), terminated, truncated, info
