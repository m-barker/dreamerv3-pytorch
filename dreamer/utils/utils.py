import torch


def combine_det_and_stoch(deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
    """
    Combines the deterministic and stochastic latent state components into a single
    tensor.

    Args:
        deter (torch.Tensor): of shape (B, D) or (B, T, D)

        stoch (torch.Tensor): of shape (B, N, S) or (B, T, N, S)

    Returns:
        torch.Tensor of shape (B, D*N*S) or (B, T, D*N*S)
    """
    T = None
    if len(deter.shape) == 2:
        B, D = deter.shape
    elif len(deter.shape) == 3:
        B, T, D = deter.shape
    else:
        raise ValueError(f"Invalid deter shape: {deter.shape}")
    t = None
    if len(stoch.shape) == 3:
        b, N, S = stoch.shape
    elif len(stoch.shape) == 4:
        b, t, N, S = stoch.shape
    else:
        raise ValueError(f"Invalid stoch shape: {stoch.shape}")

    assert B == b, "Deter and Stoch must have the same batch dim"
    assert T == t, "Deter and Stoch must either have no or the same time dim"

    if T is not None:
        stoch = stoch.reshape((B, T, N * S))
    else:
        stoch = stoch.reshape((B, N * S))

    return torch.concatenate([deter, stoch], dim=-1)
