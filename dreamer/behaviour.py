from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict

import torch

torch.set_float32_matmul_precision("high")
import torch.nn as nn


from dreamer.utils.utils import asdict_shallow, combine_det_and_stoch, PercentNorm
from dreamer.networks.actor import ActorParams, Actor
from dreamer.networks.critic import CriticParams, Critic
from dreamer.world_model import WorldModel


@dataclass
class BehaviourTrainingParams:
    lam: float
    imag_horizon: int = 15
    ret_norm: bool = True
    ret_norm_min: float = 0.05
    ret_norm_max: float = 0.95
    ret_norm_limit: float = 1.0
    ret_norm_rate: float = 0.01
    ent_weight: float = 3e-4
    slow_reg: float = 1.0
    slow_val: bool = True
    slow_target_update: int = 1
    slow_target_frac: float = 0.02


class Behaviour(nn.Module):
    """Contains the actor-critic networks and training"""

    def __init__(
        self,
        actor_params: ActorParams,
        critic_params: CriticParams,
        training_params: BehaviourTrainingParams,
        device: Optional[torch.device] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self._device = device if device is not None else torch.device("cpu")
        self._actor = Actor(**asdict_shallow(actor_params)).to(self._device)
        if compile:
            self._actor = torch.compile(self._actor)
        self._critic = Critic(**asdict_shallow(critic_params)).to(self._device)
        if compile:
            self._critic = torch.compile(self._critic)
        self._training_params = training_params

        self._slow_val_target = None
        self._slow_val_updates = 0
        if self._training_params.slow_val:
            self._slow_val_target = Critic(**asdict_shallow(critic_params)).to(
                self._device
            )
            if compile:
                self._slow_val_target = torch.compile(self._slow_val_target)

        self._ret_norm = None
        if self._training_params.ret_norm:
            self._ret_norm = PercentNorm(
                self._training_params.ret_norm_min,
                self._training_params.ret_norm_max,
                self._training_params.ret_norm_rate,
                self._training_params.ret_norm_limit,
            )
            self.register_buffer("ret_norm_vals", torch.zeros((2,)).to(self._device))

    def get_parameters(self) -> List[nn.Parameter]:
        params = []
        params += self._actor.parameters()
        params += self._critic.parameters()

        return params

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

    def _update_slow_target(self) -> None:
        """Updates the slow target network, which is an expoential moving
        average of the critic network's weights.
        """
        if self._slow_val_target is None:
            return
        if self._slow_val_updates % self._training_params.slow_target_update == 0:
            mix = self._training_params.slow_target_frac
            for s, d in zip(
                self._critic.parameters(), self._slow_val_target.parameters()
            ):
                d.data = mix * s.data + (1 - mix) * d.data
            self._slow_val_updates += 1

    def predict_values(self, latent_states: torch.Tensor) -> torch.Tensor:
        """
        Predicts and returns the Critic's estimate of the value of each
        given latent state
        """

        return self._critic.forward_and_pred(latent_states)

    def act(self, latent_state: torch.Tensor) -> torch.Tensor:
        return self._actor.forward_sample(latent_state)

    def imag_train(
        self,
        world_model: WorldModel,
        starting_deter: torch.Tensor,
        starting_stoch: torch.Tensor,
        reward_func: Optional[Callable] = None,
        continue_func: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Trains the actor and critic inside of the world model.

        Args:
            world_model (WorldModel): world model instance.

            starting_deter (torch.Tensor): starting deterministic latent
            state component of shape (B, D)

            starting_stoch (torch.Tensor): starting stochastic latent
            state component of shape (B, D)

            reward_func (callable, optional): optional reward function to use
            to predict the reward of the latent states. If None, uses the
            world model's reward head. Defaults to None.

            continue_func (callable, optional): optional continue function to
            use to predict whether the trjactory continues at each latent state.
            If None, uses the world model's continue head. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict]: actor loss and critic loss, averaged
            over batch and time dimensions, and metrics used for logging.
        """
        with torch.autocast(device_type="cuda"):
            self._update_slow_target()
            metrics = {}
            with torch.no_grad():
                out = world_model.imagine_sequence(
                    starting_deter.detach(),
                    starting_stoch.detach(),
                    self._training_params.imag_horizon,
                    self._actor.forward_sample,
                )
            # Shape (B, H, D) contains the prev_action, so out["action"][0] corresponds to the
            # prev action for out["deter"][1]
            imagined_actions = out["action"]

            # Shape (B, H + 1, D)
            imagined_latents = combine_det_and_stoch(
                out["deter"], out["prior_sample"]
            ).detach()

            policy = self._actor(imagined_latents[:, :-1])

            # Shape (B, H + 1, 1)
            imagined_value = self._critic.forward_and_pred(imagined_latents)

            metrics["mean_imagined_value"] = imagined_value.mean()
            metrics["max_imagined_value"] = imagined_value.max()
            metrics["min_imagined_value"] = imagined_value.min()

            if reward_func is not None:
                imagined_reward = reward_func(imagined_latents)
            else:
                with torch.no_grad():
                    imagined_reward = world_model.predict_reward(imagined_latents)

            metrics["mean_imagined_reward"] = imagined_reward.mean()
            metrics["max_imagined_reward"] = imagined_reward.max()
            metrics["min_imagined_reward"] = imagined_reward.min()

            if continue_func is not None:
                imagined_cont = continue_func(imagined_latents)
            else:
                with torch.no_grad():
                    imagined_cont = world_model.predict_cont(
                        imagined_latents, soft=True
                    )

            metrics["mean_imagined_continue"] = imagined_cont.mean()
            metrics["max_imagined_continue"] = imagined_cont.max()
            metrics["min_imagined_continue"] = imagined_cont.min()

            # Shape (B, H)
            ret = self._lambda_return(
                imagined_reward.squeeze(), imagined_value.squeeze(), imagined_cont
            )

            metrics["mean_lambda_return"] = ret.mean()
            metrics["max_lambda_return"] = ret.max()
            metrics["min_lambda_return"] = ret.min()

            ret_offset = 0.0
            ret_scale = 1.0
            if self._ret_norm is not None:
                ret_offset, ret_scale = self._ret_norm(ret, self.ret_norm_vals)
            advantage = (ret - imagined_value[:, :-1].squeeze()) / ret_scale

            metrics["mean_advantage"] = advantage.mean()
            metrics["max_advantage"] = advantage.max()
            metrics["min_advantage"] = advantage.min()

            logpi = policy.log_prob(imagined_actions.detach())
            policy_ent = policy.entropy()
            if len(logpi.shape) == 3:
                logpi = logpi.sum(-1)
                assert len(policy_ent.shape) == 3
                policy_ent = policy_ent.sum(-1)

            metrics["policy_entropy"] = policy_ent.mean()

            cum_continue = torch.cumprod(imagined_cont[:, :-1], dim=-1).detach()
            # Shift the continues right by one, to not mask out the terminal state.
            cum_continue = torch.cat(
                [torch.ones_like(cum_continue[:, :1]), cum_continue[:, :-1]], dim=-1
            ).detach()
            # REINFORCE - increase logprobs of actions with high advantage
            policy_loss = cum_continue * -(
                logpi * advantage.detach()
                + policy_ent * self._training_params.ent_weight
            )

            B, H, _ = imagined_latents[:, :-1].shape
            flattened_latents = imagined_latents[:, :-1].reshape((B * H, -1))
            value_dist = self._critic(flattened_latents)
            slow_val_dist = self._slow_val_target(flattened_latents)
            slow_val_pred = slow_val_dist.predict()

            metrics["mean_slow_value"] = slow_val_pred.mean()
            metrics["max_slow_value"] = slow_val_pred.max()
            metrics["min_slow_value"] = slow_val_pred.min()

            # Rehsape RET to (B*H, 1)
            val_loss = value_dist.loss(ret.detach().flatten().unsqueeze(-1))

            flattened_cont = cum_continue.reshape((B * H, -1))
            value_loss = flattened_cont * (
                val_loss
                + self._training_params.slow_reg
                * value_dist.loss(slow_val_pred.unsqueeze(-1))
            )

        return policy_loss.mean(), value_loss.mean(), metrics
