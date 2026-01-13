from typing import Tuple
from dataclasses import fields

import numpy as np
import random
import torch
import cv2


class PercentNorm:
    def __init__(
        self,
        low: float = 0.05,
        high: float = 0.95,
        rate: float = 0.01,
        limit: float = 1.0,
    ):
        self._low = low
        self._high = high
        self._rate = rate
        self._limit = limit

        self._range = torch.tensor([self._low, self._high])

    def __call__(self, unnormed_input: torch.Tensor, ema_vals: torch.Tensor):
        unnormed_in_flat = torch.flatten(unnormed_input.detach())
        x_quantile = torch.quantile(
            input=unnormed_in_flat, q=self._range.to(unnormed_in_flat.device)
        )

        ema_vals[:] = self._rate * x_quantile + (1 - self._rate) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=self._limit)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def combine_det_and_stoch(deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
    """
    Combines the deterministic and stochastic latent state components into a single
    tensor.

    Args:
        deter (torch.Tensor): of shape (B, D) or (B, T, D)

        stoch (torch.Tensor): of shape (B, N, S) or (B, T, N, S)

    Returns:
        torch.Tensor of shape (B, D*N*S) or (B, T, D*N*S)
    """
    T = None
    if len(deter.shape) == 2:
        B, D = deter.shape
    elif len(deter.shape) == 3:
        B, T, D = deter.shape
    else:
        raise ValueError(f"Invalid deter shape: {deter.shape}")
    t = None
    if len(stoch.shape) == 3:
        b, N, S = stoch.shape
    elif len(stoch.shape) == 4:
        b, t, N, S = stoch.shape
    else:
        raise ValueError(f"Invalid stoch shape: {stoch.shape}")

    assert B == b, "Deter and Stoch must have the same batch dim"
    assert T == t, "Deter and Stoch must either have no or the same time dim"

    if T is not None:
        stoch = stoch.reshape((B, T, N * S))
    else:
        stoch = stoch.reshape((B, N * S))

    return torch.concatenate([deter, stoch], dim=-1)


def asdict_shallow(dc):
    """Helper for not converting nested dataclasses into dicts"""
    result = {}
    for f in fields(dc):
        value = getattr(dc, f.name)
        # do NOT recurse into nested dataclasses
        result[f.name] = value
    return result


def resize_image(
    source_image: np.ndarray,
    target_size: Tuple[int, int],
    resize_method=cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resizes the given image using openCV.

    Args:
        source_image (np.ndarray): source image of shape (H, W, C)

        target_size (Tuple[int, int]): desired H, W

        resize_method (optional): openCV resizing method. Defaults
        to cv2.INTER_AREA
    """

    return cv2.resize(source_image, target_size, interpolation=resize_method)
