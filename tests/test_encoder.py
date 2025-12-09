import sys
import torch
from dreamer.networks.encoder import CNNEncoder


def test_same_padding_calc():
    """
    Tests the reimplementaiton of the JAX same padding calculation
    """

    encoder = CNNEncoder(
        image_shape=(7, 7, 3),
        initial_depth=24,
        kernel_size=3,
        stride=2,
        min_res=3,
        bias=False,
        norm="none",
        act_func="ReLU",
        max_pool=False,
        depth_mult=(2,),
        dilation=1,
    )
    pad_left, pad_right, pad_top, pad_bot = encoder._calculate_same_padding((7, 7))

    assert pad_left == 1
    assert pad_right == 1
    assert pad_top == 1
    assert pad_bot == 1

    pad_left, pad_right, pad_top, pad_bot = encoder._calculate_same_padding((8, 8))

    assert pad_left == 0
    assert pad_right == 1
    assert pad_top == 0
    assert pad_bot == 1


def test_output_dim_calc():
    """
    Tests the calcualtion of output resolution of one conv layer
    """

    encoder = CNNEncoder(
        image_shape=(64, 64, 3),
        initial_depth=24,
        kernel_size=3,
        stride=2,
        min_res=3,
        bias=False,
        norm="none",
        act_func="ReLU",
        max_pool=False,
        depth_mult=(2, 3, 4, 4),
        dilation=1,
    )

    out_res = encoder._calculate_output_dim(64)

    assert out_res == 32

    out_res = encoder._calculate_output_dim(32)

    assert out_res == 16

    out_res = encoder._calculate_output_dim(31)

    assert out_res == 16


def test_n_conv_layers_cal():
    encoder = CNNEncoder(
        image_shape=(64, 64, 3),
        initial_depth=24,
        kernel_size=3,
        stride=2,
        min_res=2,
        bias=False,
        norm="none",
        act_func="ReLU",
        max_pool=False,
        depth_mult=(2, 3, 4, 4, 4),
        dilation=1,
    )

    n_layers = encoder._calculate_n_conv_layers()

    assert n_layers == 5

    encoder = CNNEncoder(
        image_shape=(64, 64, 3),
        initial_depth=24,
        kernel_size=3,
        stride=2,
        min_res=3,
        bias=False,
        norm="none",
        act_func="ReLU",
        max_pool=False,
        depth_mult=(2, 3, 4, 4),
        dilation=1,
    )

    n_layers = encoder._calculate_n_conv_layers()

    assert n_layers == 4

    encoder = CNNEncoder(
        image_shape=(60, 60, 3),
        initial_depth=24,
        kernel_size=3,
        stride=3,
        min_res=3,
        bias=False,
        norm="none",
        act_func="ReLU",
        max_pool=False,
        depth_mult=(2, 3, 4),
        dilation=1,
    )

    n_layers = encoder._calculate_n_conv_layers()

    assert n_layers == 3

    encoder = CNNEncoder(
        image_shape=(64, 64, 3),
        initial_depth=24,
        kernel_size=3,
        stride=1,
        min_res=4,
        bias=False,
        norm="none",
        act_func="ReLU",
        depth_mult=(2, 3, 4, 4),
        max_pool=True,
        max_pool_kernel=2,
        max_pool_stride=2,
        dilation=1,
    )

    n_layers = encoder._calculate_n_conv_layers()

    assert n_layers == 4


def test_get_encoded_res():
    encoder = CNNEncoder(
        image_shape=(64, 64, 3),
        initial_depth=24,
        kernel_size=3,
        stride=2,
        min_res=2,
        bias=False,
        norm="none",
        act_func="ReLU",
        depth_mult=(2, 3, 4, 4, 5),
        max_pool=False,
        dilation=1,
    )
    res = encoder.encoded_res
    assert res == 2

    encoder = CNNEncoder(
        image_shape=(64, 64, 3),
        initial_depth=24,
        kernel_size=3,
        stride=2,
        min_res=3,
        bias=False,
        norm="none",
        act_func="ReLU",
        depth_mult=(2, 3, 4, 4),
        max_pool=False,
        dilation=1,
    )
    res = encoder.encoded_res
    assert res == 4
    dim = encoder.encoded_dim
    assert dim == 24 * 4 * 16


def test_encoder_foward_pass_shape():
    encoder = CNNEncoder(
        image_shape=(64, 64, 3),
        initial_depth=24,
        kernel_size=3,
        stride=1,
        min_res=4,
        bias=False,
        norm="rms",
        act_func="ReLU",
        depth_mult=(2, 3, 4, 4),
        max_pool=True,
        max_pool_kernel=2,
        max_pool_stride=2,
        dilation=1,
    )

    x = torch.randn((2, 4, 64, 64, 3))
    x = encoder.forward(x)
    assert x.shape == (2, 4, 96 * 4 * 4)
