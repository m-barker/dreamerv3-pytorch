from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributions as torchd

from .dist_utils import symexp, symlog


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

    @property
    def mode(self) -> torch.Tensor:
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


class MSEDist:
    def __init__(self, mode: torch.Tensor, agg: str = "sum"):
        """A proxy mean squared error distribution used to allow
        for the use of the same interface for all other network
        distribution outputs I.e., loss = -dist.log_prob(value).

        Args:
            mode (torch.Tensor): The output of the decoder network.
            Used to proxy the mean/mode of the distribution. Can think
            of it as a distribution with zero variance. Shape should
            be (batch_length, batch_size, h, w, c) where h, w, c are the
            height, width, and number of channels of the output respectively.

            agg (str, optional): Aggregation metric across batch dimensions.
            Can be either "mean" or "sum". Defaults to "sum".

        Raises:
            ValueError: If the mode shape is less than three dimensions.
        """

        # Needed as we strip off the batch dimensions [2:]
        if len(mode.shape) < 3:
            raise ValueError("Mode shape must have at least three dimensions.")

        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Calculates the mean squared error between the decoder output
        and the target value. Log_prob is used to allow for the use of
        the same interface for all other network distribution outputs.

        Args:
            value (torch.Tensor): The target value to calculate the
            mean squared error against. Shape should be the same as
            the mode.

        Raises:
            NotImplementedError: If the aggregation metric is not
            either "mean" or "sum".

            ValueError: If the value shape is less than three dimensions.

        Returns:
            torch.Tensor: The negative mean squared error loss. The
            Tensor is of shape (batch_length, batch_size), with the
            loss aggregated over the height, width, and channel dimensions.
        """
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)

        # Needed as we strip off the batch dimensions
        if len(value.shape) < 3:
            raise ValueError("Value shape must have at least three dimensions.")

        distance = (self._mode - value) ** 2
        # [2:] to aggregate over pixels.
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(
        self, mode: torch.Tensor, dist: str = "mse", agg: str = "sum", tol: float = 1e-8
    ):
        """Symlog proxy distribution to enable the use of the same API as actual distributions
        when calculating the world model's losses. In reality, this is either the mean
        squared error or mean absolute error between the decoder's outputs and the target.

        This is used for the vector components of the world model's output; where the output
        of the network is first passed through the symlog function. This is used to ensure that

        Args:
            mode (torch.Tensor): The output of the decoder network. Used to proxy the mean/mode
            of the 'distribution'. Tensor of shape (batch_length, batch_size, D) where D is the
            dimensionality of the vector that is being reconstructed.

            dist (str, optional): Distance metric to use between predicted and actual.
            Defaults to "mse". Must be either "mse" or "abs" (mean squared error or mean
            absolute error).

            agg (str, optional): Aggregation metric to use for the loss across vector dimensions.
            Defaults to "sum". Must be either "mean" or "sum".

            tol (float, optional): If any distance is less than this tolerance, set the
            distance to zero. Defaults to 1e-8.

        Raises:
            ValueError: If the mode shape is less than three dimensions.

        """

        # Needed as strip off the batch dimensions [2:]
        if len(mode.shape) < 3:
            raise ValueError("Mode shape must have at least three dimensions.")

        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self) -> torch.Tensor:
        """Returns the symexp transformation as all inputs to the vector encoder are first
        transformed via the symlog function, so this maps to the original data space."""
        return symexp(self._mode)

    def mean(self) -> torch.Tensor:
        return symexp(self._mode)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Calculates the mean squared error or mean absolute error between the decoder output
        and the target value. Log_prob is used to allow for the use of the same interface for
        all other network distribution outputs when computing the world model's loss.

        Args:
            value (torch.Tensor): The target value (i.e., the ground truth vector observation)

        Raises:
            NotImplementedError: If incorrect distance metric or aggregation metric is provided.

            ValueError: If the value shape is less than 3 dimensions.

        Returns:
            torch.Tensor: The negative mean squared error or mean absolute error loss. The
            Tensor is of shape (batch_length, batch_size), with the loss aggregated over the
            vector dimensions, using the given aggregation metric.
        """
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)

        # Needed as strip off the batch dimensions [2:]
        if len(value.shape) < 3:
            raise ValueError("Value shape must have at least three dimensions.")

        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)

        # [2:] to aggregate over vector dimensions, i.e., strip off the
        # batch_length and batch_size dimensions.
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)

        return -loss


class TwoHotDist:
    """
    Two-hot distribution
    """

    def __init__(
        self,
        logits: torch.Tensor,
        n_bins: int = 255,
        min_bin_val: float = -20.0,
        max_bin_val: float = 20.0,
        symexp_bins: bool = True,
    ) -> None:
        """
        Builds the bins of the two hot index.

        Args:
            logits (torch.Tensor) of shape (B, n_bins)

            n_bins (int, optional) number of bins. Defaults
            to 255.

            min_bin_val (float, optional): minimum, non-transformed
            bin value. Defaults to -20.0

            max_bin_val (int, optional): maximum, non-transformed
            bin value. Defaults to +20.0.

            symexp_bins (bool, optional): whether to apply a symexp
            transformation to the bin values, to make them exponentially
            spaced. Defaults to True.

        """
        assert logits.shape[-1] == n_bins, (
            f"Logits shape of {logits.shape} is inconsistent with the given number of bins {n_bins}"
        )
        self._logits = logits
        self._probs = torch.softmax(logits, dim=-1)
        self._n_bins = n_bins
        self._min_bin_val = min_bin_val
        self._max_bin_val = max_bin_val
        self._symexp_bins = symexp_bins

        self._bins = self._construct_bins()

    def _construct_bins(self) -> torch.Tensor:
        """
        Builds the bins, given the information from the constructor
        """

        # We have to split this in two to ensure one of the bins has a
        # value of zero, as done in the JAX official codebase

        # Case with odd number of bins
        if self._n_bins % 2 == 1:
            lower_bins = torch.linspace(
                self._min_bin_val, 0, (self._n_bins - 1) // 2 + 1, dtype=torch.float32
            )
            if self._symexp_bins:
                lower_bins = symexp(lower_bins)

            upper_bins = -torch.flip(lower_bins, [-1])
            # We have flipped, so last bin is now 0, we already have a 0 so drop it
            return torch.concatenate([lower_bins, upper_bins[:-1]], dim=-1)
        else:
            lower_bins = torch.linspace(
                self._min_bin_val, 0, self._n_bins // 2, dtype=torch.float32
            )
            if self._symexp_bins:
                lower_bins = symexp(lower_bins)

            upper_bins = -torch.flip(lower_bins, [-1])
            # For even bins, to keep symetry we have two bins that are zero
            return torch.concatenate([lower_bins, upper_bins], dim=-1)

    def predict(self, symlog_ret: bool = True) -> torch.Tensor:
        """
        Computes the weighted average of bins multiplied by their probabilities.
        As with the official JAX implementation, uses a symetric sum to prevent
        floating point errors from compounding. This is because the sum operation
        goes left to right, meaning we start adding lots of large negative numbers
        which can cause fp errors.

        Args:
            symlog_ret (bool, optional): whether to return the values back to their original
            data dim by passing through the symlog function. Defaults to true
        """

        if self._n_bins % 2 == 1:
            midpoint = (self._n_bins - 1) // 2
            lower_probs = self._probs[..., :midpoint]
            mid_prob = self._probs[..., midpoint : midpoint + 1]
            upper_probs = self._probs[..., midpoint + 1 :]
            lower_bins = self._bins[..., :midpoint]
            mid_bin = self._bins[..., midpoint : midpoint + 1]
            upper_bins = self._bins[..., midpoint + 1 :]

            # Expected value calculation
            # We flip the lower ones as sum goes left to right. So, we want always add
            # a negative magnitude with the corresponding positive magnitude, to reduce
            # floating point errors.
            weighted_avg = (mid_prob * mid_bin).sum(-1) + (
                (lower_probs * lower_bins).flip([-1]) + (upper_probs * upper_bins)
            ).sum(-1)
        else:
            midpoint = (self._n_bins) // 2
            lower_probs = self._probs[..., :midpoint]
            upper_probs = self._probs[..., midpoint:]
            lower_bins = self._bins[..., :midpoint]
            upper_bins = self._bins[..., midpoint:]

            weighted_avg = (
                (lower_probs * lower_bins).flip([-1]) + (upper_probs * upper_bins)
            ).sum(-1)

        if symlog_ret:
            return symlog(weighted_avg)
        return weighted_avg

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability of the target given the networks
        probabilities.

        Target is the true values, i.e., not passed through symexp

        Args:
           target (torch.Tensor): of shape (B, 1)

        Returns:
           log_prob (torch.Tensor) of shape (B)
        """
        assert len(target.shape) == 2, (
            f"TwoHotDist target has incorrect shape of {target.shape}"
        )
        with torch.no_grad():
            target = symexp(target)

        # Find the index of the lowest bin
        # [1, 1, 1, 1, 0, 0, 0 , ...] sum of ones is 4, -1 to convert to idx
        lower_bin_idx = (self._bins <= target).to(torch.int32).sum(-1) - 1
        upper_bin_idx = self._n_bins - (self._bins > target).to(torch.int32).sum(-1)

        # Clip lower to 0, if target is smaller than all bins
        # Clip upper to max_idx, if target is larger than all bins
        lower_bin_idx = torch.clip(lower_bin_idx, 0, self._n_bins - 1)
        upper_bin_idx = torch.clip(upper_bin_idx, 0, self._n_bins - 1)

        # Only happens (I think) when target is > max bin value
        equal = lower_bin_idx == upper_bin_idx

        # Set distance to 1 if max upper bin
        dist_to_below = torch.where(
            equal, 1, torch.abs(self._bins[lower_bin_idx] - target.squeeze())
        )
        dist_to_above = torch.where(
            equal, 1, torch.abs(self._bins[upper_bin_idx] - target.squeeze())
        )

        total_dist = dist_to_below + dist_to_above
        # further from above = more weight to below
        # and vice versa
        weight_below = (dist_to_above / total_dist).unsqueeze(-1)
        weight_above = (dist_to_below / total_dist).unsqueeze(-1)

        one_hot_target = (
            F.one_hot(lower_bin_idx, num_classes=self._n_bins) * weight_below
            + F.one_hot(upper_bin_idx, num_classes=self._n_bins) * weight_above
        )

        # Get log probs in stable way
        log_pred = self._logits - torch.logsumexp(self._logits, -1, keepdim=True)
        return (one_hot_target * log_pred).sum(-1)
