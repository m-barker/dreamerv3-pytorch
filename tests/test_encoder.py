import sys

print(sys.path)
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
        layer_norm=False,
        act_func="ReLU",
        dilation=1,
    )
    pad_left, pad_right, pad_top, pad_bot = encoder._calulate_same_padding((7, 7))

    assert pad_left == 1
    assert pad_right == 1
    assert pad_top == 1
    assert pad_bot == 1

    pad_left, pad_right, pad_top, pad_bot = encoder._calulate_same_padding((8, 8))

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
        layer_norm=False,
        act_func="ReLU",
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
        layer_norm=False,
        act_func="ReLU",
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
        layer_norm=False,
        act_func="ReLU",
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
        layer_norm=False,
        act_func="ReLU",
        dilation=1,
    )

    n_layers = encoder._calculate_n_conv_layers()

    assert n_layers == 3


def test_get_encoded_res():
    encoder = CNNEncoder(
        image_shape=(64, 64, 3),
        initial_depth=24,
        kernel_size=3,
        stride=2,
        min_res=2,
        bias=False,
        layer_norm=False,
        act_func="ReLU",
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
        layer_norm=False,
        act_func="ReLU",
        dilation=1,
    )
    res = encoder.encoded_res
    assert res == 4
