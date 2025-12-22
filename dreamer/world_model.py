from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass, asdict, fields

import torch
from torch._subclasses.fake_impls import dyn_shape
from torch.distributions.kl import kl_divergence
import torch.nn as nn

from dreamer.distributions.distributions import (
    BernoulliDist,
    BoundedNormalDist,
    MSEDist,
    OneHotDist,
    SymlogDist,
    TwoHotDist,
)
from dreamer.networks.shared import MLP, MLPandHeadParams, MLPDistHead
from dreamer.networks.encoder import Encoder, EncoderParams
from dreamer.networks.decoder import Decoder, DecoderParams
from dreamer.networks.rssm import RSSM, RSSMParams


@dataclass
class WorldModelTrainingParams:
    decoder_loss_scale: float = 1.0
    reward_loss_scale: float = 1.0
    continue_loss_scale: float = 1.0
    dynamics_loss_scale: float = 1.0
    representation_loss_scale: float = 0.1
    # floor for stochastic KL divergence loss
    free_nats: float = 1.0
    # Used for the continuation loss
    imagination_horizon: int = 15


def asdict_shallow(dc):
    """Helper for not converting nested dataclasses into dicts"""
    result = {}
    for f in fields(dc):
        value = getattr(dc, f.name)
        # do NOT recurse into nested dataclasses
        result[f.name] = value
    return result


class WorldModel:
    def __init__(
        self,
        training_params: WorldModelTrainingParams,
        rssm_params: RSSMParams,
        encoder_params: EncoderParams,
        decoder_params: DecoderParams,
        grad_components: List[str],
        reward_params: Optional[MLPandHeadParams] = None,
        continue_params: Optional[MLPandHeadParams] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self._training_params = training_params
        self._device = device
        if device is None:
            self._device = torch.device("cpu")
        rssm_params.device = self._device

        self._rssm = RSSM(**asdict(rssm_params))
        self._encoder = Encoder(**asdict_shallow(encoder_params)).to(self._device)
        self._decoder = Decoder(**asdict_shallow(decoder_params)).to(self._device)

        self._reward_network = None
        self._reward_head = None

        self._continue_network = None
        self._continue_head = None

        if reward_params is not None:
            self._reward_network = MLP(**asdict(reward_params.mlp_params)).to(
                self._device
            )
            self._reward_head = MLPDistHead(reward_params.head_params)
        if continue_params is not None:
            self._continue_network = MLP(**asdict(continue_params.mlp_params)).to(
                self._device
            )
            self._continue_head = MLPDistHead(continue_params.head_params)

        valid_grad_components = ("encoder", "decoder", "reward", "continue")
        for component in grad_components:
            assert component in valid_grad_components, (
                f"Invalid grad component {component}"
            )
        self._grad_components = grad_components

    def _compute_loss(
        self,
        data: Dict[str, torch.Tensor],
        latent_components: Dict[str, torch.Tensor],
        recon_obs: Dict[str, Union[MSEDist, SymlogDist]],
        reward_pred: Optional[TwoHotDist] = None,
        cont_pred: Optional[BernoulliDist] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the world model loss.
        """
        loss_dict = {}

        for obs_name, obs_recon in recon_obs.items():
            assert obs_name in data.keys(), f"Unreconised reconstructed obs {obs_name}"
            target = data[obs_name].detach()
            # Normalise images
            if obs_name in self._encoder._image_keys:
                target = target / 255.0
            # Average over batch and time dimensions
            loss = obs_recon.loss(target).mean(dim=(0, 1))
            loss *= self._training_params.decoder_loss_scale
            loss_dict[obs_name] = loss

        B, T = data["reward"].shape[0], data["reward"].shape[1]
        if reward_pred is not None:
            target = data["reward"].detach().reshape((B * T, *data["reward"].shape[2:]))
            loss = -reward_pred.loss(target).mean(dim=0)
            loss *= self._training_params.reward_loss_scale
            loss_dict["reward"] = loss

        if cont_pred is not None:
            target = data["continue"].detach()
            target = target.reshape((B * T, *target.shape[2:])).squeeze()
            # Used to assign a non-zero probability that the environment terminates
            # at each step, as otherwise when bootstrapping values beyond the imagination
            # horizon, the model may more easily assign certain continuation
            target *= 1 - 1 / self._training_params.imagination_horizon
            loss = cont_pred.log_prob(target).mean(dim=0)
            loss *= self._training_params.continue_loss_scale
            loss_dict["continue"] = loss

        # Dynamics loss
        # Shape (B, T, n_dists, n_classes)
        post_logits = latent_components["post_logits"]
        prior_logits = latent_components["prior_logits"]

        post_dist = OneHotDist(post_logits.detach(), unimix_ratio=self._rssm._unimix)
        prior_dist = OneHotDist(prior_logits, unimix_ratio=self._rssm._unimix)

        dynamics_loss = kl_divergence(post_dist, prior_dist)
        dynamics_loss = torch.clip(dynamics_loss, min=self._training_params.free_nats)
        dynamics_loss = dynamics_loss.mean()
        dynamics_loss *= self._training_params.dynamics_loss_scale

        loss_dict["dynamics"] = -dynamics_loss

        post_dist = OneHotDist(post_logits, unimix_ratio=self._rssm._unimix)
        prior_dist = OneHotDist(prior_logits.detach(), unimix_ratio=self._rssm._unimix)

        rep_loss = kl_divergence(post_dist, prior_dist)
        rep_loss = torch.clip(rep_loss, min=self._training_params.free_nats)
        rep_loss = rep_loss.mean()
        rep_loss *= self._training_params.representation_loss_scale

        loss_dict["representation"] = -rep_loss

        return loss_dict

    def train(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            data [Dict[str, torch.Tensor]]: pre-processed replay
            buffer data, each tensor is of shape (B, T, ...)

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor] dictionary of
            loss scalars for each components and computed posterior latent
            states by the world model, which are then used for the starting
            imagination states to train the actor-critic.

        """
        encoded_obs = self._encoder.forward(data)
        B, T = encoded_obs.shape[0], encoded_obs.shape[1]
        if "encoder" not in self._grad_components:
            encoded_obs = encoded_obs.detach()

        latent_components = self._rssm.observe_sequence(
            data["action"], encoded_obs, data["is_first"]
        )

        post_latent = torch.concatenate(
            [
                latent_components["deter"],
                latent_components["post_sample"].reshape((B, T, -1)),
            ],
            dim=-1,
        )

        decoder_in = (
            post_latent if "decoder" in self._grad_components else post_latent.detach()
        )
        reconstructed_obs = self._decoder.forward(decoder_in)

        reward_dist = None
        continue_dist = None

        if self._reward_network is not None:
            reward_in = (
                post_latent
                if "reward" in self._grad_components
                else post_latent.detach()
            )
            reward_in = reward_in.reshape((B * T, *reward_in.shape[2:]))
            reward_out = self._reward_network.forward(reward_in)
            assert self._reward_head is not None
            reward_dist = self._reward_head.forward(reward_out)
            assert isinstance(reward_dist, TwoHotDist)

        if self._continue_network is not None:
            continue_in = (
                post_latent
                if "continue" in self._grad_components
                else post_latent.detach()
            )
            continue_in = continue_in.reshape((B * T, *continue_in.shape[2:]))
            continue_out = self._continue_network.forward(continue_in)
            assert self._continue_head is not None
            continue_dist = self._continue_head.forward(continue_out)
            assert isinstance(continue_dist, BernoulliDist)

        loss = self._compute_loss(
            data, latent_components, reconstructed_obs, reward_dist, continue_dist
        )

        return loss, post_latent
