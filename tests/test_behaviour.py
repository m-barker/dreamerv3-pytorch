import torch

from dreamer.behaviour import (
    Behaviour,
    BehaviourTrainingParams,
    ActorParams,
    CriticParams,
)
from dreamer.networks.shared import OneHotParams, TwoHotDistParams


def get_mock_behaviour() -> Behaviour:
    training_params = BehaviourTrainingParams(slow_val=True, lam=1.0)
    actor_dist_params = OneHotParams(0.01)
    critic_dist_params = TwoHotDistParams()
    actor_params = ActorParams(
        n_actions=1,
        action_dim=10,
        latent_state_size=256,
        n_layers=1,
        layer_width=64,
        act_func="ReLU",
        layer_norm=True,
        bias=True,
        winit_scale=0.01,
        dist_params=actor_dist_params,
    )
    critic_params = CriticParams(
        latent_state_size=256,
        n_layers=1,
        layer_width=64,
        act_func="ReLU",
        layer_norm=True,
        bias=True,
        winit_scale=0.0,
        two_hot_params=critic_dist_params,
    )
    return Behaviour(
        actor_params=actor_params,
        critic_params=critic_params,
        training_params=training_params,
    )


def test_lambda_ret_shape():
    behaviour = get_mock_behaviour()

    reward = torch.randn((8, 16))
    value = torch.randn((8, 16))
    cont = torch.randn((8, 16))

    ret = behaviour._lambda_return(reward, value, cont)
    assert ret.shape == ((8, 15))


def test_lambda_ret_val():
    behaviour = get_mock_behaviour()

    reward = torch.randn((8, 16))
    value = torch.randn((8, 16))
    cont = torch.randn((8, 16))

    ret = behaviour._lambda_return(reward, value, cont)
    # Lambda is 1
    final_ret = reward[:, -1] + cont[:, -1] * value[:, -1]
    assert torch.all(ret[:, -1] == final_ret)

    # Lambda is 0
    behaviour._training_params.lam = 0.0
    ret = behaviour._lambda_return(reward, value, cont)

    for t in range(15):
        true_ret = reward[:, t + 1] + cont[:, t + 1] * value[:, t + 1]
        assert torch.all(ret[:, t] == true_ret)
