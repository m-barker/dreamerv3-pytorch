from dataclasses import dataclass

import torch
import torch.nn as nn

from .shared import MLP, MLPDistHead, TwoHotDistParams, TwoHotDist


@dataclass
class CriticParams:
    latent_state_size: int
    n_layers: int
    layer_width: int
    act_func: str
    layer_norm: bool
    bias: bool
    winit_scale: float
    two_hot_params: TwoHotDistParams


class Critic(nn.Module):
    """Critic class that takes in a latent state and outputs a two-hot dist
    for the value of the state.
    """

    def __init__(
        self,
        latent_state_size: int,
        n_layers: int,
        layer_width: int,
        act_func: str,
        layer_norm: bool,
        bias: bool,
        winit_scale: float,
        two_hot_params: TwoHotDistParams,
    ) -> None:
        super().__init__()

        output_dim = two_hot_params.n_bins

        self._network = MLP(
            input_dim=latent_state_size,
            out_dim=output_dim,
            n_layers=n_layers,
            layer_width=layer_width,
            act_func=act_func,
            layer_norm=layer_norm,
            bias=bias,
            winit_scale=winit_scale,
        )
        self._value_head = MLPDistHead(two_hot_params)

    def forward(self, latent_state: torch.Tensor) -> TwoHotDist:
        """
        Takes in a latent state and returns a TwoHotDist of the value
        for the given latent state.

        Args:
           latent_state (torch.Tensor): shape (B, D) or (B, D, T)
        """

        logits = self._network(latent_state)
        value_dist = self._value_head(logits)

        return value_dist

    def forward_and_pred(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Takes in a bathc of latent states and returns the predicted value
        for each state in the batch

        Args:
            latent_state (torch.Tensor) of shape (B, D) or (B, T, D)

        Returns:
            torch.Tensor of shape (B, 1) or (B, T, 1)
        """
        if len(latent_state.shape) == 3:
            B, T, D = latent_state.shape
        else:
            assert len(latent_state.shape) == 2, (
                f"Invalid latent state shape provided: {latent_state.shape}"
            )
            B, D = latent_state.shape
            T = None

        if T is not None:
            latent_state = latent_state.reshape(B * T, D)

        value_dist = self.forward(latent_state)
        value_pred = value_dist.predict()

        if T is not None:
            value_pred = value_pred.reshape(B, T, 1)

        return value_pred
