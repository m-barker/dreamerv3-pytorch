import torch

from dreamer.distributions.dist_utils import symexp, symlog
from dreamer.distributions.distributions import OneHotDist, MSEDist


def test_symlog():
    """Tests the symlog function produces correct
    output"""
    x = torch.Tensor([0, 1, 10, -100, 1000])
    expected_result = torch.Tensor([0, 0.693147, 2.397895, -4.61512, 6.90875])
    result = symlog(x)
    assert torch.allclose(result, expected_result, atol=1e-5)


def test_symexp():
    """Tests the symexp function produces correct
    output"""
    x = torch.Tensor([0, 1, -10])
    expected_result = torch.Tensor([0, 1.71828, -22025.465794807])
    result = symexp(x)
    assert torch.allclose(result, expected_result, atol=1e-5)


def test_inverse_equal():
    """Tests that applying symlog and then symexp
    recovers the initial data.
    """
    x = torch.randn(1000)
    symlog_x = symlog(x)
    symexp_symlog_x = symexp(symlog_x)
    assert torch.allclose(x, symexp_symlog_x, atol=1e-5)


def test_one_hot_dist_shapes():
    """Tests correctness of the mode and sample shapes of the
    one-hot distribution.
    """
    batch_shape = (10, 20)
    num_categories = (5,)
    sample_shape = (100,)
    dist = OneHotDist(logits=torch.randn((batch_shape) + num_categories))
    mode = dist.mode()
    sample = dist.sample(sample_shape=sample_shape)
    assert mode.shape == (batch_shape) + (num_categories)
    assert sample.shape == (sample_shape) + (batch_shape) + (num_categories)


def test_one_hot_unimix():
    """Tests the correctness of the uniform mixing functionality for the one-
    hot categorical distribution.
    """

    logits = torch.Tensor([[100.0, 0.0, 0.0, 0.0]])
    full_mix_dist = OneHotDist(logits=logits, unimix_ratio=1.0)
    no_mix_dist = OneHotDist(logits=logits, unimix_ratio=0.0)
    half_mix_dist = OneHotDist(logits=logits, unimix_ratio=0.5)

    assert torch.allclose(full_mix_dist.probs, torch.Tensor([[0.25, 0.25, 0.25, 0.25]]))
    assert torch.allclose(no_mix_dist.probs, torch.Tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(
        half_mix_dist.probs, torch.Tensor([[0.625, 0.125, 0.125, 0.125]])
    )


def test_mse_dist_shapes():
    """Tests the the aggregation of the MSE preserves the
    correct shapes"""

    decoder_output = torch.randn((10, 5, 64, 64, 3))
    mse_dist = MSEDist(decoder_output)
    loss = mse_dist.log_prob(torch.randn((10, 5, 64, 64, 3)))
    assert loss.shape == (10, 5)


def test_mse_values():
    """Checks the MSE caluclation values, for both summation and
    mean aggregation"""

    decoder_output = torch.Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
    true_values = torch.Tensor([[[[2, 3, 4], [5, 6, 7], [8, 9, 10]]]])
    mse_dist_sum = MSEDist(decoder_output, agg="sum")
    mse_dist_mean = MSEDist(decoder_output, agg="mean")

    loss_sum = mse_dist_sum.log_prob(true_values)
    loss_mean = mse_dist_mean.log_prob(true_values)

    assert torch.allclose(loss_sum, torch.Tensor([[-9]]))
    assert torch.allclose(loss_mean, torch.Tensor([[-1]]))
