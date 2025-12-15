import sys
import torch
from dreamer.networks.encoder import CNNEncoder, Encoder, MLPParams, CNNParams


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
        norm=False,
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
        norm=True,
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
        norm=True,
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
        norm=True,
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
        norm=True,
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
        norm=True,
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
        norm=True,
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
        norm=True,
        act_func="ReLU",
        depth_mult=(2, 3, 4, 4),
        max_pool=False,
        dilation=1,
    )
    res = encoder.encoded_res
    assert res == 4
    dim = encoder.encoded_dim
    assert dim == 24 * 4 * 16

    image = torch.randn((16, 64, 64, 3))
    out = encoder.forward(image)
    assert out.shape == (16, dim)


def test_encoder_foward_pass_shape():
    encoder = CNNEncoder(
        image_shape=(64, 64, 3),
        initial_depth=24,
        kernel_size=3,
        stride=1,
        min_res=4,
        bias=False,
        norm=True,
        act_func="ReLU",
        depth_mult=(2, 3, 4, 4),
        max_pool=True,
        max_pool_kernel=2,
        max_pool_stride=2,
        dilation=1,
    )

    x = torch.randn((2, 64, 64, 3))
    x = encoder.forward(x)
    assert x.shape == (2, 24 * 4 * 4 * 4)
    assert x.shape[-1] == encoder.encoded_dim


def test_encoder_single_image_no_vec():
    cnn_params = CNNParams(
        24, 3, 1, 4, True, True, "ReLU", True, (2, 3, 4, 4), 2, 2, 1, (64, 64, 3)
    )
    image_keys = ["image"]
    image_shapes = [(64, 64, 3)]

    encoder = Encoder(image_keys, image_shapes, cnn_params)

    image_data = torch.randn((16, 64, 64, 3))
    embed_dim = encoder._encoded_dim

    encoded_obs = encoder.forward({"image": image_data})
    assert encoded_obs.shape == (16, embed_dim)

    image_data = torch.randn((16, 32, 64, 64, 3))
    encoded_obs = encoder.forward({"image": image_data})
    assert encoded_obs.shape == (16, 32, embed_dim)


def test_encoder_multi_image_no_vec():
    cnn_params = CNNParams(
        24, 3, 1, 4, True, True, "ReLU", True, (2, 3, 4, 4), 2, 2, 1, (64, 64, 3)
    )
    image_keys = ["image", "image2"]
    image_shapes = [(64, 64, 3), (64, 64, 12)]

    encoder = Encoder(image_keys, image_shapes, cnn_params)

    image_data = torch.randn((16, 64, 64, 3))
    image_2_data = torch.randn((16, 64, 64, 12))
    embed_dim = encoder._encoded_dim

    encoded_obs = encoder.forward({"image": image_data, "image2": image_2_data})
    assert encoded_obs.shape == (16, embed_dim)

    image_data = torch.randn((16, 32, 64, 64, 3))
    image_2_data = torch.randn((16, 32, 64, 64, 12))
    encoded_obs = encoder.forward({"image": image_data, "image2": image_2_data})
    assert encoded_obs.shape == (16, 32, embed_dim)


def test_encoder_single_image_single_vec():
    cnn_params = CNNParams(
        24, 3, 1, 4, True, True, "ReLU", True, (2, 3, 4, 4), 2, 2, 1, (64, 64, 3)
    )
    mlp_params = MLPParams(
        input_dim=32,
        out_dim=128,
        n_layers=1,
        layer_width=64,
        act_func="ReLU",
        layer_norm=True,
        bias=True,
    )
    image_keys = ["image"]
    image_shapes = [(64, 64, 3)]
    vec_keys = ["vec"]
    vec_shapes = [32]

    encoder = Encoder(
        image_keys, image_shapes, cnn_params, vec_keys, vec_shapes, mlp_params
    )

    image_data = torch.randn((16, 64, 64, 3))
    vec_data = torch.randn((16, 32))
    embed_dim = encoder._encoded_dim

    encoded_obs = encoder.forward({"image": image_data, "vec": vec_data})
    assert encoded_obs.shape == (16, embed_dim)

    image_data = torch.randn((16, 32, 64, 64, 3))
    vec_data = torch.randn((16, 32, 32))
    encoded_obs = encoder.forward({"image": image_data, "vec": vec_data})
    assert encoded_obs.shape == (16, 32, embed_dim)


def test_encoder_multi_image_multi_vec():
    cnn_params = CNNParams(
        24, 3, 1, 4, True, True, "ReLU", True, (2, 3, 4, 4), 2, 2, 1, (64, 64, 3)
    )
    mlp_params = MLPParams(
        input_dim=32,
        out_dim=128,
        n_layers=1,
        layer_width=64,
        act_func="ReLU",
        layer_norm=True,
        bias=True,
    )
    image_keys = ["image", "image_2"]
    image_shapes = [(64, 64, 3), (64, 64, 12)]
    vec_keys = ["vec", "vec_2"]
    vec_shapes = [32, 64]

    encoder = Encoder(
        image_keys, image_shapes, cnn_params, vec_keys, vec_shapes, mlp_params
    )

    image_data = torch.randn((16, 64, 64, 3))
    image_2_data = torch.randn(16, 64, 64, 12)
    vec_data = torch.randn((16, 32))
    vec_2_data = torch.randn((16, 64))
    embed_dim = encoder._encoded_dim

    encoded_obs = encoder.forward(
        {
            "image": image_data,
            "vec": vec_data,
            "image_2": image_2_data,
            "vec_2": vec_2_data,
        }
    )
    assert encoded_obs.shape == (16, embed_dim)

    image_data = torch.randn((16, 32, 64, 64, 3))
    image_2_data = torch.randn((16, 32, 64, 64, 12))
    vec_data = torch.randn((16, 32, 32))
    vec_2_data = torch.randn((16, 32, 64))
    encoded_obs = encoder.forward(
        {
            "image": image_data,
            "vec": vec_data,
            "image_2": image_2_data,
            "vec_2": vec_2_data,
        }
    )
    assert encoded_obs.shape == (16, 32, embed_dim)
