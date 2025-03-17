import torch

from dreamer.distributions.dist_utils import symexp, symlog


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
