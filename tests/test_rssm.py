import torch

from dreamer.networks.rssm import DeterministicModule, RSSM


def test_block_gru():
    block_gru = DeterministicModule(
        deter_size=128,
        stoch_size=64,
        action_size=10,
        hidden_size=32,
        n_layers=1,
        layer_norm=True,
        bias=True,
        n_blocks=8,
        winit_scale=1.0,
        act_func="ReLU",
    )

    prev_deter = torch.randn((8, 128))
    prev_stoch = torch.randn((8, 64))
    prev_act = torch.randn(8, 10)

    deter = block_gru.forward(prev_deter, prev_stoch, prev_act)
    assert deter.shape == (8, 128)


def test_prior_logits():
    rssm = RSSM(
        deter_size=128,
        n_stoch_dists=16,
        n_stoch_cats=8,
        encoded_size=64,
        hidden_size=32,
        act_func="ReLU",
        n_prior_layers=1,
        n_post_layers=2,
        n_deter_layers=1,
        layer_norm=True,
        bias=True,
        unimix=0.01,
        winit_scale=1.0,
        n_blocks=8,
        action_dim=10,
    )

    deter = torch.randn((8, 128))
    prior_logits = rssm._prior_logit_network(deter)
    assert prior_logits.shape == (8, 8 * 16)


def test_post_logits():
    rssm = RSSM(
        deter_size=128,
        n_stoch_dists=16,
        n_stoch_cats=8,
        encoded_size=64,
        hidden_size=32,
        act_func="ReLU",
        n_prior_layers=1,
        n_post_layers=2,
        n_deter_layers=1,
        layer_norm=True,
        bias=True,
        unimix=0.01,
        winit_scale=1.0,
        n_blocks=8,
        action_dim=10,
    )

    deter = torch.randn((8, 128))
    embed = torch.randn((8, 64))
    state = torch.concatenate([deter, embed], dim=-1)
    post_logits = rssm._post_logit_network(state)
    assert post_logits.shape == (8, 8 * 16)


def test_prior_dist():
    rssm = RSSM(
        deter_size=128,
        n_stoch_dists=16,
        n_stoch_cats=8,
        encoded_size=64,
        hidden_size=32,
        act_func="ReLU",
        n_prior_layers=1,
        n_post_layers=2,
        n_deter_layers=1,
        layer_norm=True,
        bias=True,
        unimix=0.01,
        winit_scale=1.0,
        n_blocks=8,
        action_dim=10,
    )

    deter = torch.randn((8, 128))
    prior_dist = rssm._get_prior_dist(deter)

    stoch_state = prior_dist.sample()
    assert stoch_state.shape == (8, 16, 8)


def test_post_dist():
    rssm = RSSM(
        deter_size=128,
        n_stoch_dists=16,
        n_stoch_cats=8,
        encoded_size=64,
        hidden_size=32,
        act_func="ReLU",
        n_prior_layers=1,
        n_post_layers=2,
        n_deter_layers=1,
        layer_norm=True,
        bias=True,
        unimix=0.01,
        winit_scale=1.0,
        n_blocks=8,
        action_dim=10,
    )

    deter = torch.randn((8, 128))
    embed = torch.randn((8, 64))
    post_dist = rssm._get_post_dist(deter, embed)

    stoch_state = post_dist.sample()
    assert stoch_state.shape == (8, 16, 8)
