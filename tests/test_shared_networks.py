import sys

print(sys.path)
import torch
from dreamer.networks.shared import RMSNormWrapper


def test_inverse_permute():
    permute = [0, 3, 2, 1]
    norm = RMSNormWrapper(4, permute=permute)

    # Original: (A, B, C, D)
    # Permute:  (A, D, C, B)
    # Inverse:  (0, 3, 2, 1)

    inv_permute = norm._inverse_permutation(permute)
    assert inv_permute == permute

    permute = [0, 2, 3, 1]
    norm = RMSNormWrapper(2, permute=permute)

    # Original: (A, B, C, D)
    # Permute:  (A, C, D, B)
    # Inverse:  (0, 3, 1, 2)

    inv_permute = norm._inverse_permutation(permute)
    assert inv_permute == [0, 3, 1, 2]

    x = torch.randn((1, 2, 3, 4))
    y = norm(x)
    assert y.shape == x.shape
