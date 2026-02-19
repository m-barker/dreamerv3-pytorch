from typing import Callable, Dict, Tuple, Optional, List, Union
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import numpy as np
import torch.distributions as torchd
from torch.distributions.kl import kl_divergence
from PIL import Image

from dreamer.utils.utils import asdict_shallow, combine_det_and_stoch
from dreamer.distributions.distributions import (
    BernoulliDist,
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
    cont_horizon: int = 333


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
        compile: bool = False,
    ) -> None:
        self._training_params = training_params
        self._device = device
        if device is None:
            self._device = torch.device("cpu")
        rssm_params.device = self._device

        self._rssm = RSSM(**asdict(rssm_params))
        self._encoder = Encoder(**asdict_shallow(encoder_params)).to(self._device)
        if compile:
            self._encoder = torch.compile(self._encoder)
        self._decoder = Decoder(**asdict_shallow(decoder_params)).to(self._device)
        if compile:
            self._decoder = torch.compile(self._decoder)

        self._reward_network = None
        self._reward_head = None

        self._continue_network = None
        self._continue_head = None

        if reward_params is not None:
            self._reward_network = MLP(**asdict(reward_params.mlp_params)).to(
                self._device
            )
            if compile:
                self._reward_network = torch.compile(self._reward_network)
            self._reward_head = MLPDistHead(reward_params.head_params).to(self._device)
        if continue_params is not None:
            self._continue_network = MLP(**asdict(continue_params.mlp_params)).to(
                self._device
            )
            if compile:
                self._continue_network = torch.compile(self._continue_network)
            self._continue_head = MLPDistHead(continue_params.head_params).to(
                self._device
            )

        valid_grad_components = ("encoder", "decoder", "reward", "continue")
        for component in grad_components:
            assert component in valid_grad_components, (
                f"Invalid grad component {component}"
            )
        self._grad_components = grad_components

    def get_parameters(self) -> List[nn.Parameter]:
        """
        Since this class isn't a nn.Module, collate all of the
        training parameters of the sub classes, and return in
        a single List, used to give to the optimiser.
        """
        params: List[nn.Parameter] = []

        params += list(self._rssm.parameters())
        params += list(self._encoder.parameters())
        params += list(self._decoder.parameters())

        if self._reward_network is not None:
            params += list(self._reward_network.parameters())
        if self._reward_head is not None:
            params += list(self._reward_head.parameters())

        if self._continue_network is not None:
            params += list(self._continue_network.parameters())
        if self._continue_head is not None:
            params += list(self._continue_head.parameters())

        return params

    def get_init_deter(self, batch_size: int = 1) -> torch.Tensor:
        init_deter, _ = self._rssm._get_initial_state(batch_size)
        return init_deter

    def get_init_stoch(self, batch_size: int = 1) -> torch.Tensor:
        _, init_stoch = self._rssm._get_initial_state(batch_size)
        return init_stoch

    def _compute_loss(
        self,
        data: Dict[str, torch.Tensor],
        latent_components: Dict[str, torch.Tensor],
        recon_obs: Dict[str, Union[MSEDist, SymlogDist]],
        reward_pred: Optional[TwoHotDist] = None,
        cont_pred: Optional[BernoulliDist] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Computes the world model loss.

        Args:
            data (Dict[str, torch.Tensor]): replay buffer sample
            where each value must be a Tensor of shape (B, T, ...)

            latent_components (Dict[str, torch.Tensor]): world model
            latent state components computed for the replay buffer sample.
            Each component must be of shape (B, T, ...)

            recon_obs: Dict[str, Union[MSEDist, SymlogDist]]: dictinoary
            containing the k,v pairs of the name of each reconstructed observation
            and the corresponding modelled distribiution for the decoded observation.

            reward_pred: Optional[TwoHotDist]: optional reward prediction.

            cont_pred: Optional[BernoulliDist]: optional continuation prediction.
        Returns:
            Tuple[Dict, Dict]: loss dictionary and logging metrics dictionary.
        """
        loss_dict = {}
        metrics = {}

        for obs_name, obs_recon in recon_obs.items():
            assert obs_name in data.keys(), f"Unreconised reconstructed obs {obs_name}"
            target = data[obs_name].detach()
            # Normalise images
            if obs_name in self._encoder._image_keys:
                target = target / 255.0
            # Reward, continuation can be stored as 1D tensors
            if len(target.shape) == 1:
                target = target.unsqueeze(-1)
            # Average over batch and time dimensions
            loss = -obs_recon.loss(target).mean(dim=(0, 1))
            loss *= self._training_params.decoder_loss_scale
            loss_dict[obs_name] = loss

        B, T = data["reward"].shape[0], data["reward"].shape[1]
        if reward_pred is not None:
            target = data["reward"].detach().reshape((B * T, *data["reward"].shape[2:]))
            if len(target.shape) == 1:
                target = target.unsqueeze(-1)
            loss = reward_pred.loss(target)
            loss = loss.mean(dim=0)
            loss *= self._training_params.reward_loss_scale
            loss_dict["reward"] = loss

        if cont_pred is not None:
            target = data["continue"].detach()
            target = target.reshape((B * T, *target.shape[2:])).squeeze()
            # Used to assign a non-zero probability that the environment terminates
            # at each step, as otherwise when bootstrapping values beyond the cont
            # horizon, the model may more easily assign certain continuation
            # Can be viewed as playing the same role as the discount rate
            target *= 1 - 1 / self._training_params.cont_horizon
            loss = -cont_pred.log_prob(target).mean(dim=0)
            loss *= self._training_params.continue_loss_scale
            loss_dict["continue"] = loss

        # Dynamics loss
        # Shape (B, T, n_dists, n_classes)
        post_logits = latent_components["post_logits"]
        prior_logits = latent_components["prior_logits"]

        post_dist = OneHotDist(post_logits.detach(), unimix_ratio=self._rssm._unimix)
        prior_dist = OneHotDist(prior_logits, unimix_ratio=self._rssm._unimix)
        post_dist = torchd.independent.Independent(post_dist, 1)
        prior_dist = torchd.independent.Independent(prior_dist, 1)

        dynamics_loss = kl_divergence(post_dist, prior_dist)  # (B, T)
        dynamics_loss = torch.clip(dynamics_loss, min=self._training_params.free_nats)
        dynamics_loss = dynamics_loss.mean()

        dynamics_loss *= self._training_params.dynamics_loss_scale

        loss_dict["dynamics"] = dynamics_loss

        post_dist = OneHotDist(post_logits, unimix_ratio=self._rssm._unimix)
        prior_dist = OneHotDist(prior_logits.detach(), unimix_ratio=self._rssm._unimix)
        post_dist = torchd.independent.Independent(post_dist, 1)
        prior_dist = torchd.independent.Independent(prior_dist, 1)

        metrics["post_entropy"] = post_dist.entropy().mean()
        metrics["prior_entropy"] = prior_dist.entropy().mean()

        rep_loss = kl_divergence(post_dist, prior_dist)
        rep_loss = torch.clip(rep_loss, min=self._training_params.free_nats)
        rep_loss = rep_loss.mean()
        rep_loss *= self._training_params.representation_loss_scale

        loss_dict["representation"] = rep_loss

        return loss_dict, metrics

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns a dictionary contaning the state_dict of each world
        model component. Used for saving the model.
        """
        state = {
            "rssm": self._rssm.state_dict(),
            "encoder": self._encoder.state_dict(),
            "decoder": self._decoder.state_dict(),
        }

        if self._reward_network is not None:
            state["reward_network"] = self._reward_network.state_dict()
        if self._reward_head is not None:
            state["reward_head"] = self._reward_head.state_dict()

        if self._continue_network is not None:
            state["continue_network"] = self._continue_network.state_dict()
        if self._continue_head is not None:
            state["continue_head"] = self._continue_head.state_dict()

        return state

    def load_state_dict(
        self,
        state: Dict[str, Dict[str, torch.Tensor]],
        strict: bool = True,
    ):
        """
        Loads a given state dictionary. Used when loading the model
        from a checkpoint.
        """
        self._rssm.load_state_dict(state["rssm"], strict=strict)
        self._encoder.load_state_dict(state["encoder"], strict=strict)
        self._decoder.load_state_dict(state["decoder"], strict=strict)

        if self._reward_network is not None and "reward_network" in state:
            self._reward_network.load_state_dict(state["reward_network"], strict=strict)
        if self._reward_head is not None and "reward_head" in state:
            self._reward_head.load_state_dict(state["reward_head"], strict=strict)

        if self._continue_network is not None and "continue_network" in state:
            self._continue_network.load_state_dict(
                state["continue_network"], strict=strict
            )
        if self._continue_head is not None and "continue_head" in state:
            self._continue_head.load_state_dict(state["continue_head"], strict=strict)

    def imagine_sequence(
        self,
        starting_deter: torch.Tensor,
        starting_stoch: torch.Tensor,
        length: int,
        policy: Optional[Callable] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagines a sequences from a given starting latent state.

        Args:
            starting_deter (torch.Tensor): starting deterministic
            state of shape (B, deter_dim).

            starting_stoch (torch.Tensor): starting stochastic state
            of shape (B, n_dists, n_classes)

            length (int): number of steps to imagine.

            policy (optional[callable]): optional policy to imagine
            actions with. If given, must be able to take in a latent
            state (torch Tensor) and return an action (torch tensor).

            actions (optinoal[torch.Tensor]): optional action to take
            at each imagined step. Must match the length given.

        Returns:
            Dict[str, torch.Tensor]: dictinoary of imagined latent
            components, each of shape (B, length + 1, ...). one
            extra for the length as it also returns the start state.
        """
        return self._rssm.imagine_sequence(
            starting_deter, starting_stoch, length, policy, actions
        )

    def encode_obs(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encodes observations into a latent tensor.

        Args:
            data (Dict[str, torch.Tensor]): data to encoded. Typically
            a replay buffer sample. Each value can either be of shape
            (B, T, ...) or (B, ...).

        Returns:
            torch.Tensor of encoded obs of shape (B, encoded_dim) or
            (B, T, encoded_dim)
        """
        return self._encoder.forward(data)

    def predict_reward(self, latent_states: torch.Tensor) -> torch.Tensor:
        """
        Returns the reward head's predicted reward for each latent state in the
        batch.

        Args:
            latent_states (torch.Tensor): latent states of shape (B,D) or (B, T, D)

        Returns:
            torch.Tensor: predicted reward of shape (B, 1) or (B, T, 1), in the
            original data space.
        """
        if len(latent_states.shape) == 3:
            B, T, D = latent_states.shape
        else:
            assert len(latent_states.shape) == 2, (
                f"Invalid number of dimensions of latent states: {latent_states.shape}"
            )
            B, D = latent_states.shape
            T = None
        if T is not None:
            latent_states = latent_states.reshape(B * T, D)
        reward_logits = self._reward_network(latent_states)
        reward_dist = self._reward_head(reward_logits)
        pred = reward_dist.predict()
        if T is not None:
            pred = pred.reshape(B, T, 1)
        return pred

    def predict_cont(
        self, latent_states: torch.Tensor, soft: bool = False
    ) -> torch.Tensor:
        """
        Returns the predicted probability that the trajectory continues for
        each latent state in the batch.

        Args:
            latent_states (torch.Tensor): latent states of shape (B, D) or (B, T, D)

        Returns:
            torch.Tensor: predicted continues of shape (B) or (B, T)
        """
        if len(latent_states.shape) == 3:
            B, T, D = latent_states.shape
        else:
            assert len(latent_states.shape) == 2, (
                f"Invalid number of dimensions of latent states: {latent_states.shape}"
            )
            B, D = latent_states.shape
            T = None
        if T is not None:
            latent_states = latent_states.reshape(B * T, D)
        continue_logits = self._continue_network(latent_states)
        continue_dist = self._continue_head(continue_logits)
        if soft:
            pred = continue_dist.mean()
        else:
            pred = continue_dist.pred()
        if T is not None:
            pred = pred.reshape(B, T)
        return pred

    def decode_images(self, deter: torch.Tensor, stoch: torch.Tensor) -> np.ndarray:
        """
        Decodes images and returns a numpy array of the decoded images.

        Args:
            deter (torch.Tensor): deterministic latent compopnent of shape
            (B, T, deter_dim)

            stoch (torch.Tensor): stochastic latent component of shape
            (B, T, n_dists, n_classes)

        Returns:
            np.ndarray of shape (B, T, H, W, C)
        """
        decoder_dict = self._decoder.forward(deter, stoch)
        decoder_dist = decoder_dict["image"]
        recon_images = decoder_dist.mean()
        recon_images = recon_images.detach().cpu().numpy().squeeze() * 255
        recon_images = np.rint(recon_images).astype(np.uint8)

        return recon_images

    def get_posterior(
        self,
        obs: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        prev_deter: torch.Tensor,
        prev_stoch: torch.Tensor,
        sample_latent: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the posterior latent state components of a given observation,
        previous action, and optional previous latent state.

        Args:
            obs: Dict[str, torch.Tensor]: dictionary of tensor observations
            for the current timestep.

            prev_action: torch.Tensor: action taken in the previous timestep.

            prev_deter (torch.Tensor): previous deterministic state.

            prev_stoch (torch.Tensor): previous stochastic state.

            sample_latent (optional, bool): whether to sample the stochasticv
            component. Defaults to True

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: determinsitic and posterior
            stochastic state.
        """
        encoded_obs = self._encoder.forward(obs)
        if len(prev_action.shape) == 1:
            prev_action = prev_action.unsqueeze(0)
        latent_components = self._rssm.obs_step(
            prev_action,
            encoded_obs,
            prev_deter,
            prev_stoch,
            sample_latent,
        )
        return latent_components["deter"], latent_components["post_sample"]

    def observe_sequence(
        self,
        prev_actions: torch.Tensor,
        encoded_obs: torch.Tensor,
        prev_deter: torch.Tensor,
        prev_stoch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the (posterior) latent state sequence for a sequence of encoded observations
        and actions. Assumes the first observation in the sequence is the starting state.

        Args:
            prev_actions (torch.Tensor) of shape (B, T, ...). Contains the previous action
            for each given observation in the sequence.

            encoded_obs (torch.Tensor) of shape (B, T, encoded_dim). Encoded observation
            sequence.

            prev_deter (torch.Tensor) of shape (B, deter_dim). Previous deterministic
            component for the step before the start step of this sequence.

            prev_stoch (torch.Tensor) of shape (B, n_dists, n_cats). Previous stochastic
            component of the step before the start step of this sequence.
        """
        return self._rssm.observe_sequence(
            prev_actions, encoded_obs, prev_deter, prev_stoch
        )

    def train(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[
        Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]
    ]:
        """
        Args:
            data [Dict[str, torch.Tensor]]: pre-processed replay
            buffer data, each tensor is of shape (B, T, ...)

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, torhc.Tenosr, Dict] dictionary of
            loss scalars for each components and computed posterior latent
            states by the world model, which are then used for the starting
            imagination states to train the actor-critic, and a dictionary
            of metircs used for logging purposes.

        """
        with torch.autocast(device_type="cuda"):
            encoded_obs = self._encoder.forward(data)
            B, T = encoded_obs.shape[0], encoded_obs.shape[1]
            if "encoder" not in self._grad_components:
                encoded_obs = encoded_obs.detach()

            latent_components = self._rssm.observe_sequence(
                data["prev_action"],
                encoded_obs,
                data["prev_deter"][:, 0],
                data["prev_stoch"][:, 0],
            )

            post_latent = combine_det_and_stoch(
                latent_components["deter"], latent_components["post_sample"]
            )

            decoder_deter = (
                latent_components["deter"]
                if "decoder" in self._grad_components
                else latent_components["deter"].detach()
            )
            decoder_stoch = (
                latent_components["post_sample"]
                if "decoder" in self._grad_components
                else latent_components["post_sample"].detach()
            )
            reconstructed_obs = self._decoder.forward(decoder_deter, decoder_stoch)

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

            loss, metrics = self._compute_loss(
                data, latent_components, reconstructed_obs, reward_dist, continue_dist
            )

        return (
            loss,
            latent_components["deter"],
            latent_components["post_sample"],
            metrics,
        )
