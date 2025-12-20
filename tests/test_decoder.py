import torch
from dreamer.distributions.distributions import MSEDist, SymlogDist
from dreamer.networks.decoder import CNNDecoder, Decoder, DecoderCNNParams, MLPParams


def test_decoder_output():
    decoder = CNNDecoder(
        image_shape=(64, 64, 3),
        starting_res=4,
        latent_dim=2048,
        kernel_size=3,
        starting_depth=24,
        depth_mults=(2, 3, 4, 4),
        bias=False,
        norm=True,
        act_func="ReLU",
        final_sigmoid=True,
    )
    latent_states = torch.randn((2, 2048))
    decoded_images = decoder(latent_states)
    assert decoded_images.shape == (2, 64, 64, 3)


def test_single_img_decoder_shape():
    cnn_decoder_params = DecoderCNNParams(
        starting_res=4,
        latent_dim=256,
        kernel_size=3,
        starting_depth=24,
        depth_mults=(2, 3, 4, 4),
        bias=True,
        norm=True,
        act_func="ReLU",
        final_sigmoid=True,
        image_shape=(64, 64, 3),
    )

    image_keys = ["image"]
    image_shapes = [(64, 64, 3)]

    decoder = Decoder(
        image_keys=image_keys, image_shapes=image_shapes, cnn_params=cnn_decoder_params
    )

    latent_states = torch.randn(16, 256)

    out = decoder(latent_states)

    assert isinstance(out["image"], MSEDist)
    mean = out["image"].mean()
    assert mean.shape == (16, 64, 64, 3)

    target_img = torch.randn((16, 64, 64, 3))
    loss = out["image"].loss(target_img)
    assert loss.shape == (16,)


def test_multi_img_decoder_shape():
    cnn_decoder_params = DecoderCNNParams(
        starting_res=4,
        latent_dim=256,
        kernel_size=3,
        starting_depth=24,
        depth_mults=(2, 3, 4, 4),
        bias=True,
        norm=True,
        act_func="ReLU",
        final_sigmoid=True,
        image_shape=(64, 64, 3),
    )

    image_keys = ["image", "image_2", "image_3"]
    image_shapes = [(64, 64, 3), (64, 64, 10), (64, 64, 13)]

    decoder = Decoder(
        image_keys=image_keys, image_shapes=image_shapes, cnn_params=cnn_decoder_params
    )

    latent_states = torch.randn(16, 256)

    out = decoder(latent_states)

    assert isinstance(out["image"], MSEDist)
    mean = out["image"].mean()
    assert mean.shape == (16, 64, 64, 3)
    mean = out["image_2"].mean()
    assert mean.shape == (16, 64, 64, 10)
    mean = out["image_3"].mean()
    assert mean.shape == (16, 64, 64, 13)

    latent_states = torch.randn(2, 4, 256)

    out = decoder(latent_states)

    assert isinstance(out["image"], MSEDist)
    mean = out["image"].mean()
    assert mean.shape == (2, 4, 64, 64, 3)
    mean = out["image_2"].mean()
    assert mean.shape == (2, 4, 64, 64, 10)
    mean = out["image_3"].mean()
    assert mean.shape == (2, 4, 64, 64, 13)


def test_single_img_single_vec_decoder_shape():
    cnn_decoder_params = DecoderCNNParams(
        starting_res=4,
        latent_dim=256,
        kernel_size=3,
        starting_depth=24,
        depth_mults=(2, 3, 4, 4),
        bias=True,
        norm=True,
        act_func="ReLU",
        final_sigmoid=True,
        image_shape=(64, 64, 3),
    )

    image_keys = ["image"]
    image_shapes = [(64, 64, 3)]

    vector_keys = ["vec"]
    vector_dims = [10]

    mlp_params = MLPParams(
        input_dim=256,
        out_dim=12,
        n_layers=2,
        layer_width=64,
        act_func="ReLU",
        layer_norm=True,
        bias=True,
    )

    decoder = Decoder(
        image_keys=image_keys,
        image_shapes=image_shapes,
        cnn_params=cnn_decoder_params,
        vector_keys=vector_keys,
        vector_shapes=vector_dims,
        mlp_params=mlp_params,
        symlog_vecs=True,
    )

    latent_states = torch.randn(16, 256)

    out = decoder(latent_states)

    assert isinstance(out["image"], MSEDist)
    mean = out["image"].mean()
    assert mean.shape == (16, 64, 64, 3)

    target_img = torch.randn((16, 64, 64, 3))
    loss = out["image"].loss(target_img)
    assert loss.shape == (16,)

    assert isinstance(out["vec"], SymlogDist)
    mean = out["vec"].mean()
    assert mean.shape == (16, 10)

    target_vec = torch.randn((16, 10))
    loss = out["vec"].loss(target_vec)
    assert loss.shape == (16,)


def test_multi_img_multi_vec_decoder_shape():
    cnn_decoder_params = DecoderCNNParams(
        starting_res=4,
        latent_dim=256,
        kernel_size=3,
        starting_depth=24,
        depth_mults=(2, 3, 4, 4),
        bias=True,
        norm=True,
        act_func="ReLU",
        final_sigmoid=True,
        image_shape=(64, 64, 3),
    )

    image_keys = ["image", "image_2", "image_3"]
    image_shapes = [(64, 64, 3), (64, 64, 10), (64, 64, 5)]

    vector_keys = ["vec", "vec_2", "vec_3"]
    vector_dims = [10, 22, 4]

    mlp_params = MLPParams(
        input_dim=256,
        out_dim=12,
        n_layers=2,
        layer_width=64,
        act_func="ReLU",
        layer_norm=True,
        bias=True,
    )

    decoder = Decoder(
        image_keys=image_keys,
        image_shapes=image_shapes,
        cnn_params=cnn_decoder_params,
        vector_keys=vector_keys,
        vector_shapes=vector_dims,
        mlp_params=mlp_params,
        symlog_vecs=True,
    )

    latent_states = torch.randn(16, 256)

    out = decoder(latent_states)

    assert isinstance(out["image"], MSEDist)
    mean = out["image"].mean()
    assert mean.shape == (16, 64, 64, 3)

    assert isinstance(out["image_2"], MSEDist)
    mean = out["image_2"].mean()
    assert mean.shape == (16, 64, 64, 10)

    assert isinstance(out["image_3"], MSEDist)
    mean = out["image_3"].mean()
    assert mean.shape == (16, 64, 64, 5)

    assert isinstance(out["vec"], SymlogDist)
    mean = out["vec"].mean()
    assert mean.shape == (16, 10)

    assert isinstance(out["vec_2"], SymlogDist)
    mean = out["vec_2"].mean()
    assert mean.shape == (16, 22)

    assert isinstance(out["vec_3"], SymlogDist)
    mean = out["vec_3"].mean()
    assert mean.shape == (16, 4)
