import torch
import torch.nn as nn

from .shared import MLP, MLPDistHead, TwoHotDistParams, TwoHotDist


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
           latent_state (torch.Tensor): shape (B, D)
        """

        logits = self._network(latent_state)
        value_dist = self._value_head(logits)

        return value_dist
