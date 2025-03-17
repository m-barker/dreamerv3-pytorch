from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributions as torchd


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        unimix_ratio: float = 0.0,
    ):
        """A one-hot categorical distribution with straight-through
        gradients as proposed in https://arxiv.org/abs/1308.3432, with optional
        uniform mixing. The distribution can be initialised with either logits
        or probabilities, but not both.

        The last dimension of the input tensor is assumed to be the number of
        categories. Sampling returns independent one-hot samples that numbers
        the sum of all non-final dimensions of the input tensor.

        Args:
            logits (Optional[torch.Tensor], optional): . Tensor of unormalised probabilities
            Defaults to None. Either logits or probs must be provided.

            probs (Optional[torch.Tensor], optional): . Tensor of probabilities. Defaults to None.

            unimix_ratio (float, optional): Proportion of probabilities that
            should come from a uniform distribution. Defaults to 0.0.
        """
        if logits is None and probs is None:
            raise ValueError("Either logits or probs must be provided")
        if not 0.0 <= unimix_ratio <= 1.0:
            raise ValueError("unimix_ratio must be in [0, 1]")
        if logits is not None and unimix_ratio > 0.0:
            # Normalise, mix with uniform, and convert to log-probs.
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        """Returns the mode of the distribution in one-hot format.

        Returns:
            torch.Tensor of shape [batch_shape, num_categories]: The mode of the distribution.
        """
        # Shape (batch_shape, num_categories)
        _mode = F.one_hot(
            torch.argmax(super().logits, dim=-1), super().logits.shape[-1]
        )
        # straight-through gradients
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=()):
        sample = super().sample(sample_shape)
        probs = super().probs
        # Add empty dimensions to probs to match sample shape
        # to enable broadcasting for straight-through grads.
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample
