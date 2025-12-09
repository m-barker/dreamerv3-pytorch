import torch
from dreamer.networks.decoder import CNNDecoder


def test_decoder_output():
    decoder = CNNDecoder(
        image_shape=(64, 64, 3),
        starting_res=4,
        latent_dim=2048,
        kernel_size=3,
        starting_depth=24,
        depth_mults=(2, 3, 4, 4),
        bias=False,
        norm="rms",
        act_func="ReLU",
        final_sigmoid=True,
    )
    latent_states = torch.randn((2, 4, 2048))
    decoded_images = decoder(latent_states)
    assert decoded_images.shape == (2, 4, 64, 64, 3)
