import torch

from dreamer.distributions.dist_utils import symexp, symlog
from dreamer.distributions.distributions import (
    OneHotDist,
    MSEDist,
    SymlogDist,
    TwoHotDist,
    BoundedNormalDist,
    BernoulliDist,
)


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
    mode = dist.mode
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

    decoder_output = torch.randn((10, 64, 64, 3))
    mse_dist = MSEDist(decoder_output)
    loss = mse_dist.loss(torch.randn((10, 64, 64, 3)))
    assert loss.shape == (10,)


def test_mse_values():
    """Checks the MSE caluclation values, for both summation and
    mean aggregation"""

    decoder_output = torch.Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
    true_values = torch.Tensor([[[[2, 3, 4], [5, 6, 7], [8, 9, 10]]]])
    mse_dist_sum = MSEDist(decoder_output, agg="sum")
    mse_dist_mean = MSEDist(decoder_output, agg="mean")

    loss_sum = mse_dist_sum.loss(true_values)
    loss_mean = mse_dist_mean.loss(true_values)

    assert torch.allclose(loss_sum, torch.Tensor([-9]))
    assert torch.allclose(loss_mean, torch.Tensor([-1]))


def test_symlog_dist_shapes():
    """Tests that the aggregation and distance metric calculations
    of the symlog dist preserves the correct shapes"""

    # (batch_size, D)
    decoder_output = torch.randn((10, 20))
    symlog_dist = SymlogDist(decoder_output)
    loss = symlog_dist.loss(torch.randn((10, 20)))
    assert loss.shape == (10,)

    # (B, T, D)
    decoder_output = torch.randn((10, 15, 20))
    symlog_dist = SymlogDist(decoder_output)
    loss = symlog_dist.loss(torch.randn((10, 15, 20)))
    assert loss.shape == (10, 15)


def test_symlog_dist_values():
    """Checks that the symlog distance metric is calculated correctly
    for abs and mse, and for sum and mean aggregations
    """

    decoder_output = torch.Tensor(
        [
            [1, 2, 3, 4, 5, 6, 7],
        ]
    )
    # Symlog of true values is taken when calculating the loss
    true_values = symexp(
        torch.Tensor(
            [
                [3, 4, 5, 6, 7, 8, 10],
            ]
        )
    )

    symlog_dist_mse_sum = SymlogDist(decoder_output, dist="mse", agg="sum")
    symlog_dist_mse_mean = SymlogDist(decoder_output, dist="mse", agg="mean")
    symlog_dist_abs_sum = SymlogDist(decoder_output, dist="abs", agg="sum")
    symlog_dist_abs_mean = SymlogDist(decoder_output, dist="abs", agg="mean")

    loss_mse_sum = symlog_dist_mse_sum.loss(true_values)
    loss_mse_mean = symlog_dist_mse_mean.loss(true_values)
    loss_abs_sum = symlog_dist_abs_sum.loss(true_values)
    loss_abs_mean = symlog_dist_abs_mean.loss(true_values)

    assert torch.allclose(loss_mse_sum, torch.Tensor([[-33]]))
    assert torch.allclose(loss_mse_mean, torch.Tensor([[-4.7143]]))
    assert torch.allclose(loss_abs_sum, torch.Tensor([[-15]]))
    assert torch.allclose(loss_abs_mean, torch.Tensor([[-2.14286]]))


def test_twohot_shapes():
    logits = torch.randn((16, 255))
    dist = TwoHotDist(logits)

    target = torch.randn((16, 1))

    res = dist.loss(target)

    assert res.shape == (16,)


def test_twohot_bins():
    logits = torch.randn(1, 7)
    dist = TwoHotDist(logits, symexp_bins=False, n_bins=7)
    bins = dist._bins
    midpoint = 7 // 2
    # Should be symetric around 0.0
    assert bins[midpoint] == 0.0
    assert bins[0] == -bins[-1]


def test_bounded_normal_shapes():
    mean = torch.randn((16, 4))
    std = torch.randn((16, 4))

    dist = BoundedNormalDist(mean, std, 0.0, 1.0)
    sample = dist.sample()

    assert sample.shape == (16, 4)

    entropy = dist.entropy()
    assert entropy.shape == (16, 4)

    vals = torch.randn((16, 4))
    log_prob = dist.log_prob(vals)

    assert log_prob.shape == (16, 4)


def test_bounded_normal_bounds():
    mean = torch.randn((16, 4))
    std = torch.randn((16, 4))

    dist = BoundedNormalDist(mean, std, 0.0, 1.0)

    stdev = dist._dist.stddev

    assert torch.all(stdev >= 0.0)
    assert torch.all(stdev <= 1.0)


def test_bernoulli_shape():
    logits = torch.randn((10, 1))
    dist = BernoulliDist(logits)

    pred = dist.pred()
    assert pred.shape == (10,)

    sample = dist.sample()
    assert sample.shape == (10,)

    vals = torch.ones((10,))
    log_prob = dist.log_prob(vals)
    assert log_prob.shape == (10,)
