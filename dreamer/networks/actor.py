from typing import Union

import torch
import torch.nn as nn

from dreamer.distributions.distributions import BoundedNormalDist, OneHotDist

from .shared import OneHotParams, BoundedNormalParams, MLP, MLPDistHead


class Actor(nn.Module):
    """
    Class containing the actor network that takes in a latent
    state and outputs a policy
    """

    def __init__(
        self,
        n_actions: int,
        action_dim: int,
        latent_state_size: int,
        n_layers: int,
        layer_width: int,
        act_func: str,
        layer_norm: bool,
        bias: bool,
        winit_scale: float,
        dist_params: Union[OneHotParams, BoundedNormalParams],
    ) -> None:
        """
        Args:
            n_actions (int): Number of possible actions.

            action_dim (int): Number of dimensions per action. For now,
            we assume that this is 1 for discrete actions (e.g., no parameterised
            actions), and constant across continuous actions, e.g., there isn't an
            actuator that takes a scalar value and another actuator that takes a
            multi-dimensional vector, etc.

            latent_state_size (int): the number of dimensions of the latent state

            n_layers (int): number of layers in the actor network.

            layer_width (int): the number of neurons in each hidden layer.

            act_func (str): name of the activation function to use after each layer.

            layer_norm (bool): whether to apply layer normalisation after each layer.

            bias (bool): whether each layer should have a bias or not.

            winit_scale (float): amount to scale the weight initialisation by.

            dist_params (Union[OneHotParams, BoundedNormalParams]): parameters for the
            one-hot distribution or the bounded-normal distribution for continuous acts.

        """
        super().__init__()

        output_dimension = n_actions * action_dim
        if isinstance(dist_params, BoundedNormalParams):
            # As we have to parameterise the mean and stddev
            # of each action dimension
            output_dimension *= 2

        self._network = MLP(
            input_dim=latent_state_size,
            out_dim=output_dimension,
            n_layers=n_layers,
            layer_width=layer_width,
            act_func=act_func,
            layer_norm=layer_norm,
            bias=bias,
            winit_scale=winit_scale,
        )

        self._n_actions = n_actions
        self._action_dim = action_dim

        self._policy_head = MLPDistHead(dist_params)

    def forward(
        self, latent_state: torch.Tensor
    ) -> Union[BoundedNormalDist, OneHotDist]:
        """
        Forward pass that takes in a batch of latent states and returns a batch of policies.

        Args:
            latent_state (torch.Tensor): of dimension (B, D)

        Returns:
            Union[BoundedNormalDist, OneHotDist] a continuous or discrete distribution over
            actions, essentially a set of independent policies over the set of batches,
            but sampling from this distribution needs to handle re-shaping back to the
            appropriate action dimension
        """

        logits = self._network(latent_state)
        policy = self._policy_head(logits)

        return policy

    def draw_from_policy(
        self, policy: Union[BoundedNormalDist, OneHotDist], sample: bool = True
    ) -> torch.Tensor:
        """
        Draws an action from the policy and does appropriate reshaping to convert
        back to environment's action space

        Args:
            policy (Union[BoundedNormalDist, OneHotDist]): a continuous or discrete
            policy.

            sample (bool, optional): whether to sample from the policy, or return
            the modal action. Defaults to True.

        Returns:
            torch.Tensor of shape (B, n_actions, action_dim)
        """

        if sample:
            action = policy.sample()
        else:
            action = policy.mode

        batch_size = action.shape[0]

        return action.reshape((batch_size, self._n_actions, self._action_dim))


if __name__ == "__main__":
    pass
