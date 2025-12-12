import torch

from dreamer.networks.actor import Actor, OneHotParams, BoundedNormalParams, OneHotDist


def test_forward_pass():
    actor = Actor(
        n_actions=5,
        action_dim=1,
        latent_state_size=64,
        n_layers=3,
        layer_width=32,
        act_func="ReLU",
        layer_norm=True,
        bias=True,
        winit_scale=0.01,
        dist_params=OneHotParams(0.0),
    )
    latent_state = torch.randn((16, 64))
    policy = actor.forward(latent_state)

    assert isinstance(policy, OneHotDist)
    assert policy.sample().shape == ((16, 5))


def test_policy_drawing_continuous():
    actor = Actor(
        n_actions=5,
        action_dim=2,
        latent_state_size=64,
        n_layers=3,
        layer_width=32,
        act_func="ReLU",
        layer_norm=True,
        bias=True,
        winit_scale=0.01,
        dist_params=BoundedNormalParams(0.0, 2.0),
    )
    latent_state = torch.randn((16, 64))
    policy = actor.forward(latent_state)

    action = actor.draw_from_policy(policy)

    assert action.shape == ((16, 5, 2))


def test_policy_drawing_discrete():
    actor = Actor(
        n_actions=10,
        action_dim=1,
        latent_state_size=64,
        n_layers=3,
        layer_width=32,
        act_func="ReLU",
        layer_norm=True,
        bias=True,
        winit_scale=0.01,
        dist_params=OneHotParams(0.5),
    )
    latent_state = torch.randn((16, 64))
    policy = actor.forward(latent_state)

    action = actor.draw_from_policy(policy)

    assert action.shape == ((16, 10, 1))
