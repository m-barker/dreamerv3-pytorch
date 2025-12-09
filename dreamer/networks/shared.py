from typing import Optional, Union, List

import torch
import torch.nn as nn


class RMSNormWrapper(nn.Module):
    """Implements Pytorch's RMS norm, but with the option
    of reshaping the input
    """

    def __init__(
        self,
        norm_size: Union[List[int], int],
        eps: float = 1e-4,
        permute: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            norm_dims (Union[Tuple[int, int]]): the size to normalise
            over. Normalises over the last dimensions.

            eps (float, optional) epsilon float added to the
            rms demoninator for numerical stability

            permute (optional, List[int]): Optional permutation dimensions
        """
        super().__init__()
        self._norm = nn.RMSNorm(norm_size, eps=eps)
        self._permute = permute
        self._inverse_permute = None
        if self._permute:
            self._inverse_permute = self._inverse_permutation(self._permute)

    def _inverse_permutation(self, permutation: List[int]) -> List[int]:
        """
        Computes the inverse permutation

        Args:
            permutation (List[int]) permutation to undo
        """

        inverse_permute = [0] * len(permutation)
        for i, p in enumerate(permutation):
            inverse_permute[p] = i
        return inverse_permute

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes RMS Normalisation over x, which can be any shape.
        """

        if self._permute:
            assert len(x.shape) == len(self._permute), (
                f"Permutation length {len(self._permute)} does not match input length {len(x.shape)}"
            )
            x = x.permute(*self._permute)

        x = self._norm(x)
        if self._inverse_permute:
            x = x.permute(*self._inverse_permute)
        return x
