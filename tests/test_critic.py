import torch

from dreamer.networks.critic import Critic
from dreamer.networks.shared import TwoHotDistParams


def test_critic_initially_zero():
    dist_params = TwoHotDistParams()
    critic = Critic(
        latent_state_size=64,
        n_layers=3,
        layer_width=32,
        act_func="ReLU",
        layer_norm=True,
        bias=True,
        winit_scale=0.0,
        two_hot_params=dist_params,
    )

    latent_state = torch.randn((16, 64))
    value_dist = critic.forward(latent_state)

    values = value_dist.predict()

    assert torch.all(values == 0.0)


def test_critic_shapes():
    dist_params = TwoHotDistParams()
    critic = Critic(
        latent_state_size=64,
        n_layers=3,
        layer_width=32,
        act_func="ReLU",
        layer_norm=True,
        bias=True,
        winit_scale=0.0,
        two_hot_params=dist_params,
    )

    latent_state = torch.randn((16, 64))
    value_dist = critic.forward(latent_state)

    values = value_dist.predict()

    assert values.shape == (16,)
