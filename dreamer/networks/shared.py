from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import numpy as np


def truncated_normal_weight_init(layer: nn.Module, weight_scale: float = 1.0) -> None:
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

        weight_scale (float, optional): amount to scale weights by.
        Defaults to 1.0
    """

    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        with torch.no_grad():
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            # We assume data is sampled from a standard normal N(0,1)
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=1.0, a=-2.0, b=2.0)
            # Magic number rescaling for +-2stdev truncation back to unit var.
            layer.weight.mul_(1.1368 * np.sqrt(1 / fan_in) * weight_scale)
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
                param.mul_(1.1368 * np.sqrt(1 / fan_in) * weight_scale)
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


class MLP(nn.Module):
    """Multi-layer perceptron network"""

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        n_layers: int,
        layer_width: int,
        act_func: str,
        layer_norm: bool,
        bias: bool,
        winit_scale: float = 1.0,
    ) -> None:
        """
        Args:
            input_dim (int): dim of input to the network

            out_dim (int): dim of the network output

            n_layers (int): number of MLP layers

            layer_width (int): number of neurons in each hidden layer.

            act_func (str): name of activation to apply after each layer,
            including the output layer. Must match a torch activation function
            name

            layer_norm (bool): whether to apply rms normalisation after each
            layer.

            bias (bool): whether to have a bias in the linear layers

            winit_scale (float): amount to scale the winit by. Especially useful
            for when wanting to zero the weights for reward and value heads.


        """
        super().__init__()

        self._input_dim = input_dim
        self._out_dim = out_dim
        self._n_layers = n_layers
        self._layer_width = layer_width
        self._act_func = act_func
        self._layer_norm = layer_norm
        self._winit_scale = winit_scale
        self._bias = bias

        self._network = self._build_network()
        # Have to use a lambda to pass the winit_scale arg in
        self._network.apply(
            lambda m: truncated_normal_weight_init(m, self._winit_scale)
        )

    def _build_network(self) -> nn.Sequential:
        layers = []

        act_func = getattr(nn, self._act_func)

        for i in range(self._n_layers - 1):
            if i == 0:
                layers.append(
                    nn.Linear(self._input_dim, self._layer_width, bias=self._bias)
                )
            else:
                layers.append(
                    nn.Linear(self._layer_width, self._layer_width, bias=self._bias)
                )
            if self._layer_norm:
                layers.append(RMSNormWrapper([self._layer_width]))
            layers.append(act_func())

        layers.append(nn.Linear(self._layer_width, self._out_dim, bias=self._bias))
        if self._layer_norm:
            layers.append(RMSNormWrapper([self._out_dim]))
        layers.append(act_func())

        return nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor) of shape (B, D)

        Returns:
            torch.Tensor of shape (B, out_dim)
        """

        return self._network(input)
