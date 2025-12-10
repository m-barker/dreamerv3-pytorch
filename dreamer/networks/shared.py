from typing import Optional, Union, List

import torch
import torch.nn as nn
import numpy as np


def truncated_normal_weight_init(layer: nn.Module) -> None:
    """
    Applies truncated normal distribution weights initlisation,
    as done in the official Dreamer V3 paper. If the layer has
    a bias term, fills this with zeros.

    The official Dreamer paper uses the "fan in" method across
    all weight initilisation, which scales the variance depending
    on the size of the input to the layer. Such that all weights
    have a variance of 1 / fan.

    This currently only truncates to +-2 std deviations. The
    magic number is used to rescale the variance to be unit
    after the truncation.

    Args:
        layer (nn.Module): layer to initialise in-place.
    """

    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        with torch.no_grad():
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            # We assume data is sampled from a standard normal N(0,1)
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=1.0, a=-2.0, b=2.0)
            # Magic number rescaling for +-2stdev truncation back to unit var.
            layer.weight.mul_(1.1368 * np.sqrt(1 / fan_in))
        if layer.bias is not None:
            with torch.no_grad():
                layer.bias.zero_()
    elif isinstance(layer, nn.RMSNorm):
        with torch.no_grad():
            # Initalise to 1.0 as initally we don't scale the
            # normalisation
            layer.weight.fill_(1.0)
    elif isinstance(layer, nn.GRU):
        for name, param in layer.named_parameters():
            if "weight" in name:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(param)
                nn.init.trunc_normal_(param, mean=0.0, std=1.0, a=-2.0, b=2.0)
                param.mul_(1.1368 * np.sqrt(1 / fan_in))
            elif "bias" in name:
                with torch.no_grad():
                    param.zero_()


class RMSNormWrapper(nn.Module):
    """Implements Pytorch's RMS norm, but with the option
    of reshaping the input
    """

    def __init__(
        self,
        norm_size: Union[List[int], int],
        eps: float = 1e-4,
        permute: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            norm_dims (Union[Tuple[int, int]]): the size to normalise
            over. Normalises over the last dimensions.

            eps (float, optional) epsilon float added to the
            rms demoninator for numerical stability

            permute (optional, List[int]): Optional permutation dimensions
        """
        super().__init__()
        self._norm = nn.RMSNorm(norm_size, eps=eps)
        self._permute = permute
        self._inverse_permute = None
        if self._permute:
            self._inverse_permute = self._inverse_permutation(self._permute)

    def _inverse_permutation(self, permutation: List[int]) -> List[int]:
        """
        Computes the inverse permutation

        Args:
            permutation (List[int]) permutation to undo
        """

        inverse_permute = [0] * len(permutation)
        for i, p in enumerate(permutation):
            inverse_permute[p] = i
        return inverse_permute

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes RMS Normalisation over x, which can be any shape.
        """

        if self._permute:
            assert len(x.shape) == len(self._permute), (
                f"Permutation length {len(self._permute)} does not match input length {len(x.shape)}"
            )
            x = x.permute(*self._permute)

        x = self._norm(x)
        if self._inverse_permute:
            x = x.permute(*self._inverse_permute)
        return x
