import sys

import torch
from dreamer.networks.shared import (
    RMSNormWrapper,
    MLP,
    truncated_normal_weight_init,
    OneHotDist,
    OneHotParams,
    MLPDistHead,
)
from dreamer.networks.encoder import CNNEncoder


def test_inverse_permute():
    permute = [0, 3, 2, 1]
    norm = RMSNormWrapper(4, permute=permute)

    # Original: (A, B, C, D)
    # Permute:  (A, D, C, B)
    # Inverse:  (0, 3, 2, 1)

    inv_permute = norm._inverse_permutation(permute)
    assert inv_permute == permute

    permute = [0, 2, 3, 1]
    norm = RMSNormWrapper(2, permute=permute)

    # Original: (A, B, C, D)
    # Permute:  (A, C, D, B)
    # Inverse:  (0, 3, 1, 2)

    inv_permute = norm._inverse_permutation(permute)
    assert inv_permute == [0, 3, 1, 2]

    x = torch.randn((1, 2, 3, 4))
    y = norm(x)
    assert y.shape == x.shape


def test_norm_weight_init():
    permute = [0, 3, 2, 1]
    norm = RMSNormWrapper(4, permute=permute)
    norm.apply(truncated_normal_weight_init)
    for name, param in norm.named_parameters():
        assert torch.all(param == 1)


def test_encoder_weight_init():
    encoder = CNNEncoder(
        image_shape=(64, 64, 3),
        initial_depth=24,
        kernel_size=3,
        stride=1,
        min_res=4,
        bias=True,
        norm="rms",
        act_func="ReLU",
        depth_mult=(2, 3, 4, 4),
        max_pool=True,
        max_pool_kernel=2,
        max_pool_stride=2,
        dilation=1,
    )
    encoder.apply(truncated_normal_weight_init)
    for name, param in encoder.named_parameters():
        if "rms" in name.lower() and "weight" in name:
            assert torch.all(param == 1), f"{name} RMSNorm weight not ones"
        elif "bias" in name.lower():
            assert torch.all(param == 0)
        elif "weight" in name:
            max_val = param.abs().max().item()
            # 2.3 as we scale [-2, 2] by 1.1368 * sqrt(1/something)
            assert max_val <= 2.3, f"{name} exceeds truncation range: {max_val}"


def test_mlp_shapes():
    network = MLP(16, 64, 3, 8, "ReLU", True, True)
    input = torch.randn((2, 16))
    output = network(input)
    assert output.shape == ((2, 64))


def test_mlp_winit():
    network = MLP(16, 64, 3, 8, "ReLU", True, True, winit_scale=0.0)

    input = torch.randn((2, 16))
    output = network(input)

    assert torch.all(output == 0)

    network = MLP(16, 64, 3, 8, "ReLU", True, True, winit_scale=1.0)

    input = torch.randn((2, 16))
    output = network(input)

    assert not torch.all(output == 0)


def test_mlp_dist_head():
    one_hot_params = OneHotParams(0.0)
    logits = torch.randn((10, 20))

    head = MLPDistHead(one_hot_params)

    dist = head.forward(logits)

    assert isinstance(dist, OneHotDist)
    sample = dist.sample()
    assert sample.shape == ((10, 20))
