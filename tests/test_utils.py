import torch

from dreamer.utils.utils import combine_det_and_stoch


def test_combine_latents():
    B = 16
    T = 32

    deter = torch.randn((B, T, 8))
    stoch = torch.randn((B, T, 16, 16))

    latent = combine_det_and_stoch(deter, stoch)
    assert latent.shape == (B, T, 8 + (16 * 16))

    deter = torch.randn((B, 8))
    stoch = torch.randn((B, 16, 16))

    latent = combine_det_and_stoch(deter, stoch)
    assert latent.shape == (B, 8 + (16 * 16))
