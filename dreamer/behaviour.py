from dataclasses import dataclass

import torch

from dreamer.utils.utils import asdict_shallow, combine_det_and_stoch
from dreamer.networks.actor import ActorParams, Actor
from dreamer.networks.critic import CriticParams, Critic


@dataclass
class BehaviourTrainingParams:
    slow_val: bool
    lam: float


class Behaviour:
    """Contains the actor-critic networks and training"""

    def __init__(
        self,
        actor_params: ActorParams,
        critic_params: CriticParams,
        training_params: BehaviourTrainingParams,
    ) -> None:
        self._actor = Actor(**asdict_shallow(actor_params))
        self._critic = Critic(**asdict_shallow(critic_params))
        self._training_params = training_params

        self._slow_val_target = None
        if self._training_params.slow_val:
            self._slow_val_target = Critic(**asdict_shallow(critic_params))

    def _lambda_return(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        cont: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes lambda returns, i.e., TD(lambda), a mix between monte-carlo and bootstrapped values
        estimates.

        Args:
            reward (torch.Tensor) of shape (B, H + 1)

            value (torch.Tensor) of shape (B, H + 1)

            cont (torch.Tensor) of shape (B, H + 1). The probability that the sequence continues
            at the current timestep. Includes discounting.

        Returns:
            torch.Tensor lambda returh of shape (B, H)
        """

        # This contains the reversed time returns i.e., L_T, L_T-1, ..., L_0
        # Final timestep is just bootstrapped with the value.
        lambda_returns = [value[:, -1]]
        lam = self._training_params.lam
        for t in reversed(range(reward.shape[1] - 1)):
            next_ret = lambda_returns[-1]
            ret = reward[:, t + 1] + cont[:, t + 1] * (
                (1 - lam) * value[:, t + 1] + lam * next_ret
            )

            lambda_returns.append(ret)

        return torch.stack(list(reversed(lambda_returns))[:-1], 1)
