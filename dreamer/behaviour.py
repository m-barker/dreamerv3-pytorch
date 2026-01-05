from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


from dreamer.utils.utils import asdict_shallow, combine_det_and_stoch, PercentNorm
from dreamer.networks.actor import ActorParams, Actor
from dreamer.networks.critic import CriticParams, Critic
from dreamer.world_model import WorldModel


@dataclass
class BehaviourTrainingParams:
    slow_val: bool
    lam: float
    imag_horizon: int = 15
    ret_norm: bool = True
    ret_norm_min: float = 0.05
    ret_norm_max: float = 0.95
    ret_norm_limit: float = 1.0
    ret_norm_rate: float = 0.01


class Behaviour(nn.Module):
    """Contains the actor-critic networks and training"""

    def __init__(
        self,
        actor_params: ActorParams,
        critic_params: CriticParams,
        training_params: BehaviourTrainingParams,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._device = device if device is not None else torch.device("cpu")
        self._actor = Actor(**asdict_shallow(actor_params))
        self._critic = Critic(**asdict_shallow(critic_params))
        self._training_params = training_params

        self._slow_val_target = None
        if self._training_params.slow_val:
            self._slow_val_target = Critic(**asdict_shallow(critic_params))

        if self._training_params.ret_norm:
            self._ret_norm = PercentNorm(
                self._training_params.ret_norm_min,
                self._training_params.ret_norm_max,
                self._training_params.ret_norm_rate,
                self._training_params.ret_norm_limit,
            )
            self.register_buffer("ret_norm_vals", torch.zeros((2,)).to(self._device))

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

    def imag_train(
        self,
        world_model: WorldModel,
        starting_deter: torch.Tensor,
        starting_stoch: torch.Tensor,
        reward_func: Optional[Callable] = None,
        continue_func: Optional[Callable] = None,
    ):
        out = world_model.imagine_sequence(
            starting_deter,
            starting_stoch,
            self._training_params.imag_horizon,
            self._actor.forward_sample,  # TODO: make this function
        )
        # Shape (B, H, D)
        imagined_actions = out["action"]
        # Shape (B, H + 1, D)
        imagined_latents = combine_det_and_stoch(out["deter"], out["prior_sample"])

        # TODO: flatten and reshape
        policy = self._actor(imagined_latents)
        imagined_value = self._critic(imagined_latents)

        if reward_func is not None:
            imagined_reward = reward_func(imagined_latents)
        else:
            imagined_reward = world_model.predict_reward(imagined_latents)

        if continue_func is not None:
            imagined_cont = continue_func(imagined_latents)
        else:
            imagined_cont = world_model.predict_cont(imagined_latents)

        ret = self._lambda_return(imagined_reward, imagined_value, imagined_cont)
