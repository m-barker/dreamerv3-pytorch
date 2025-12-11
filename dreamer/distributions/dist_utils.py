import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Computes the elementwise symmetric logarithm of
    an input tensor, as defined by:

    x = sign(x) * ln(|x| + 1)

    Args:
        x (torch.Tensor): Input tensor of any shape

    Returns:
        torch.Tensor: Elementwise symmetric logarithm of x
    """

    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Computes the elementwise symmetric exponential of
    an input tensor (the inverse of symlog), as defined by:

    x = sign(x) * (exp(|x|) - 1)

    Args:
        x (torch.Tensor): Input tensor of any shape

    Returns:
        torch.Tensor: Elementwise symmetric exponential of x
    """

    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
