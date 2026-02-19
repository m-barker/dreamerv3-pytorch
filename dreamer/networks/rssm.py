from typing import Callable, Dict, Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared import RMSNormWrapper, BlockLinearLayer, truncated_normal_weight_init
from dreamer.distributions.distributions import OneHotDist
from dreamer.utils.utils import combine_det_and_stoch


@dataclass
class RSSMParams:
    deter_size: int  # size of deterministic latent component
    n_stoch_dists: int  # number of stochastic distributions
    n_stoch_cats: int  # number of categories in each distribution
    encoded_size: int  # encoded dim of the encoder
    hidden_size: int  # number of neurons per hidden layer
    act_func: str  # name of the activation function to use after each layer
    n_prior_layers: int
    n_post_layers: int
    n_deter_layers: int
    layer_norm: bool
    bias: bool
    unimix: float  # prop of logits that come from mixing with a uniform dist
    winit_scale: float  # mult for weight initialisation
    n_blocks: int  # n_blocks in block GRU
    action_dim: int  # total flattened action dim
    device: Optional[torch.device] = None
    compile: bool = False


class DeterministicModule(nn.Module):
    """Module for getting the deterministic component of Dreamer's Latent State"""

    def __init__(
        self,
        deter_size: int,
        stoch_size: int,
        action_size: int,
        hidden_size: int,
        n_layers: int,
        layer_norm: bool,
        bias: bool,
        n_blocks: int,
        winit_scale: float,
        act_func: str,
    ) -> None:
        """
        See RSSM for parameter descriptions
        """

        super().__init__()
        self._deter_size = deter_size
        self._stoch_size = stoch_size
        self._action_size = action_size
        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._layer_norm = layer_norm
        self._bias = bias
        self._n_blocks = n_blocks
        self._winit_scale = winit_scale
        self._act_func_name = act_func

        self._deter_deter_in = nn.Linear(
            self._deter_size, self._hidden_size, bias=self._bias
        )
        self._deter_act_in = nn.Linear(
            self._action_size, self._hidden_size, bias=self._bias
        )
        self._deter_stoch_in = nn.Linear(
            self._stoch_size, self._hidden_size, bias=self._bias
        )

        self._act_func = getattr(F, self._act_func_name.lower())

        if self._layer_norm:
            self._deter_deter_in_norm = RMSNormWrapper(self._hidden_size)
            self._deter_act_in_norm = RMSNormWrapper(self._hidden_size)
            self._deter_stoch_in_norm = RMSNormWrapper(self._hidden_size)

        self._deter_hidden_layers = nn.ModuleList(
            BlockLinearLayer(
                self._n_blocks
                * ((self._deter_size // self._n_blocks) + 3 * self._hidden_size),
                self._deter_size,
                self._n_blocks,
                self._bias,
                self._layer_norm,
                self._winit_scale,
                self._act_func_name,
            )
            for _ in range(self._n_layers)
        )

        self._deter_hidden_out = BlockLinearLayer(
            self._deter_size,
            3 * self._deter_size,
            self._n_blocks,
            self._bias,
            layer_norm=False,
            act_func=None,
        )

        self.apply(lambda m: truncated_normal_weight_init(m, self._winit_scale))

    def forward(
        self,
        prev_deter: torch.Tensor,
        prev_stoch: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Takes in the previous deterministic and stochastic latent state and action, and outputs the current
        timestep's deterministic state by passing through a block diagonal GRU.

        Args:
            prev_deter (torch.Tensor): of shape (B, self._deter_size)

            prev_stoch (torch.Tensor): of shape (B, self._stoch_size)

            prev_action (torch.Tensor): of shape (B, self._action_size)
        """
        # Normalise continuous actions
        prev_action = (
            prev_action
            / torch.maximum(torch.ones_like(prev_action), prev_action.abs()).detach()
        )

        batch_size = prev_deter.shape[0]

        prev_deter_embed = self._deter_deter_in(prev_deter)
        prev_stoch_embed = self._deter_stoch_in(prev_stoch)
        prev_action_embed = self._deter_act_in(prev_action)

        if self._layer_norm:
            prev_deter_embed = self._deter_deter_in_norm(prev_deter_embed)
            prev_stoch_embed = self._deter_stoch_in_norm(prev_stoch_embed)
            prev_action_embed = self._deter_act_in_norm(prev_action_embed)
        prev_deter_embed = self._act_func(prev_deter_embed)
        prev_stoch_embed = self._act_func(prev_stoch_embed)
        prev_action_embed = self._act_func(prev_action_embed)

        # Shape (B, 3 * self._hidden_size)
        context = torch.concat(
            [prev_deter_embed, prev_stoch_embed, prev_action_embed], dim=-1
        )
        # Shape (B, 1, 3*self._hidden_size)
        context = context.reshape(batch_size, 1, context.shape[-1])
        context = context.repeat(1, self._n_blocks, 1)

        # Split deterministic state into blocks
        # (B, N, deter_size // N)
        deter_blocks = prev_deter.view(
            batch_size, self._n_blocks, self._deter_size // self._n_blocks
        )

        # Concatenate local + global inputs per block
        # (B, N, (deter_block + 3 * hidden))
        block_input = torch.cat([deter_blocks, context], dim=-1)

        # Flatten blocks into feature dimension for BlockLinearLayer
        # (B, N * ((deter_block) + 3 * hidden))
        block_input = block_input.view(batch_size, -1)

        # Pass through stacked block-linear layers
        x = block_input
        for layer in self._deter_hidden_layers:
            x = layer(x)

        # Final projection to GRU gates
        # (B, 3 * deter_size)
        gates = self._deter_hidden_out(x)

        reset, candidate, update = torch.chunk(gates, 3, dim=-1)

        reset = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        update = torch.sigmoid(update - 1)

        # GRU-style update
        deter = update * candidate + (1.0 - update) * prev_deter

        return deter


class RSSM:
    """Recurrent State Space Model (RSSM)"""

    def __init__(
        self,
        deter_size: int,
        n_stoch_dists: int,
        n_stoch_cats: int,
        encoded_size: int,
        hidden_size: int,
        act_func: str,
        n_prior_layers: int,
        n_post_layers: int,
        n_deter_layers: int,
        layer_norm: bool,
        bias: bool,
        unimix: float,
        winit_scale: float,
        n_blocks: int,
        action_dim: int,
        device: Optional[torch.device] = None,
        compile: bool = False,
    ) -> None:
        """
        Args:
            deter_size (int): dimensionality of the deterministic component of the
            latent state.

            n_stoch_dists (int): the number of onehot categorical distributions that
            form the stochastic component of the latent state.

            n_stoch_cats (int): the number of categories in each of the categorical
            distributions of the stochastic latent state component.

            encoded_size (int): the dimensionality of the vector output by the encoder.

            hidden_size (int): the number of neurons in each hidden layer of the RSSM.

            act_func (str): the activation function that is applied after each linear
            layer.

            n_prior_layers (int): the number of linear layers that map the deterministic
            component to the prior stochastic dist.

            n_post_layers (int): the number of stochastic layers that map the deterministic
            component plus the encoder output, to the posterior stochastic dist.

            layer_norm (bool): whether to apply layer normalisation after every layer.

            bias (bool): whether every layer should have a bias or not.

            unimix (float): percentage of the stochastic component that compes from a
            uniform distribution. Used to make sure no stochastic part has non-zero
            probability.

            winit_scale (float): scalar amount to multiply the weight initialisation by.

            n_blocks (int): the number of blocks in the block linear GRU component.

            action_dim (int): number of dimensions across (flattened) actions.

            device (torch.device, optional): device to put tensors on. If None,
            defaults to CPU. Defaults to None.

            compile (bool, optional): whether to compile the module. Defaults to False
        """
        self._deter_size = deter_size
        self._n_stoch_dists = n_stoch_dists
        self._n_stoch_cats = n_stoch_cats
        self._encoded_size = encoded_size
        self._hidden_size = hidden_size
        self._act_func_name = act_func
        self._n_prior_layers = n_prior_layers
        self._n_post_layers = n_post_layers
        self._n_deter_layers = n_deter_layers
        self._layer_norm = layer_norm
        self._bias = bias
        self._unimix = unimix
        self._winit_scale = winit_scale
        self._n_blocks = n_blocks
        self._action_dim = action_dim

        self._stoch_dim = self._n_stoch_dists * self._n_stoch_cats
        self._latent_dim = self._deter_size + self._stoch_dim

        self._act_func = getattr(nn, self._act_func_name)
        if device is None:
            device = torch.device("cpu")
        self._device = device

        self._block_gru = DeterministicModule(
            self._deter_size,
            self._stoch_dim,
            self._action_dim,
            self._hidden_size,
            self._n_deter_layers,
            self._layer_norm,
            self._bias,
            self._n_blocks,
            self._winit_scale,
            self._act_func_name,
        ).to(device)
        if compile:
            self._block_gru = torch.compile(self._block_gru)

        self._prior_logit_network = (
            self._build_prior_network().apply(truncated_normal_weight_init).to(device)
        )
        if compile:
            self._prior_logit_network = torch.compile(self._prior_logit_network)
        self._post_logit_network = (
            self._build_post_network().apply(truncated_normal_weight_init).to(device)
        )
        if compile:
            self._post_logit_network = torch.compile(self._post_logit_network)

    def parameters(self) -> List[nn.Parameter]:
        params = []
        params += self._block_gru.parameters()
        params += self._prior_logit_network.parameters()
        params += self._post_logit_network.parameters()
        return params

    def _build_prior_network(self) -> nn.Sequential:
        """
        Constructs the network for taking in the current deterministic state and
        outputting the prior distribution's logits.
        """
        layers = []
        for layer in range(self._n_prior_layers):
            in_dim = self._deter_size if layer == 0 else self._hidden_size
            layers.append(nn.Linear(in_dim, self._hidden_size, self._bias))
            if self._layer_norm:
                layers.append(RMSNormWrapper(self._hidden_size))
            layers.append(self._act_func())

        # Logits for stochastic distribution
        layers.append(nn.Linear(self._hidden_size, self._stoch_dim))

        return nn.Sequential(*layers)

    def _build_post_network(self) -> nn.Sequential:
        """
        Constructs the network for taking in the current deterministic state and
        observation embedding and outputting the posterior distribution's logits.
        """
        layers = []
        for layer in range(self._n_post_layers):
            in_dim = (
                (self._deter_size + self._encoded_size)
                if layer == 0
                else self._hidden_size
            )
            layers.append(nn.Linear(in_dim, self._hidden_size, self._bias))
            if self._layer_norm:
                layers.append(RMSNormWrapper(self._hidden_size))
            layers.append(self._act_func())

        # Logits for stochastic distribution
        layers.append(nn.Linear(self._hidden_size, self._stoch_dim))

        return nn.Sequential(*layers)

    def _get_deterministic_latent(
        self, prev_deter: torch.Tensor, prev_stoch: torch.Tensor, prev_act: torch.Tensor
    ) -> torch.Tensor:
        """
        Gets the current deterministic state h_t = f(h(t-1), z(t-1), a(t-1))

        Args:
            prev_deter (torch.Tensor) of shape (B, self._deter_size)

            prev_stoch (torch.Tensor) of shape (B, self._stoch_dim)

            prev_act (torch.Tensor) of shape (B, self._action_dim)

        Returns:
           torch.Tensor of shape (B, self._deter_size)
        """
        return self._block_gru(prev_deter, prev_stoch, prev_act)

    def _get_prior_dist(self, deter: torch.Tensor) -> OneHotDist:
        """
        Takes in the current deterministic state h_t and outputs the
        prior categorical distribution.

        Args:
            deter (torch.Tensor) of shape (B, self._deter_size)

        Returns:
            OneHotDist - prior distribution
        """
        prior_logits = self._prior_logit_network(deter)
        prior_logits = prior_logits.reshape(
            (prior_logits.shape[0], self._n_stoch_dists, self._n_stoch_cats)
        )
        prior_dist = OneHotDist(logits=prior_logits, unimix_ratio=self._unimix)
        return prior_dist

    def _get_post_dist(self, deter: torch.Tensor, embed: torch.Tensor) -> OneHotDist:
        """
        Takes in the current deterministic state h_t and observation embedding x_t
        and outputs the posterior categorical distribution.

        Args:
            deter (torch.Tensor) of shape (B, self._deter_size)

            embed (torch.Tensor) of shape (B, self._encoded_size)

        Returns:
           OneHotDist - prior distribution
        """
        state = torch.concatenate([deter, embed], dim=-1)
        post_logits = self._post_logit_network(state)
        post_logits = post_logits.reshape(
            (post_logits.shape[0], self._n_stoch_dists, self._n_stoch_cats)
        )
        post_dist = OneHotDist(logits=post_logits, unimix_ratio=self._unimix)
        return post_dist

    def _get_initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns tensors of zeros for the initial deterministic and stochastic
        state
        """
        init_deter = torch.zeros((batch_size, self._deter_size)).to(self._device)
        init_stoch = torch.zeros(
            (batch_size, self._n_stoch_dists, self._n_stoch_cats)
        ).to(self._device)

        return init_deter, init_stoch

    def _handle_is_first(
        self, data: torch.Tensor, is_first: torch.Tensor
    ) -> torch.Tensor:
        """
        Zeros any data where is_first == 1

        Args:
            data (torch.Tensor) of shape (B, D) or (B, D, D)

            is_first (torch.Tensor) of shape (B, 1)
        """
        # For stochastic latent, add extra dim to is_first
        # to make broadcasting possible
        if len(data.shape) == 3:
            is_first = is_first.unsqueeze(-1)
            if len(is_first.shape) == 2:
                is_first = is_first.unsqueeze(-1)
        # (B, ) -> (B, 1)
        if len(is_first.shape) == 1:
            is_first = is_first.unsqueeze(-1)

        return data * (1.0 - is_first)

    def observe_sequence(
        self,
        prev_actions: torch.Tensor,
        encoded_obs: torch.Tensor,
        prev_deter: torch.Tensor,
        prev_stoch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            prev_actions (torch.Tensor): actions taken at the previous timestep of
            shape (B, T, self._action_size)

            encoded_obs (torch.Tensor): encoder output for observations at each timestep
            of shape (B, T, self._encoded_size)

            prev_deter (torch.Tensor) of shape (B, deter_dim). Previous deterministic
            component for the step before the start step of this sequence.

            prev_stoch (torch.Tensor) of shape (B, n_dists, n_cats). Previous stochastic
            component of the step before the start step of this sequence.

        """
        B, T, _ = prev_actions.shape

        # Initialize previous hidden states
        prev_deter, prev_post = prev_deter, prev_stoch

        # Prepare storage for outputs
        seq_outputs = {
            "deter": [],
            "prior_logits": [],
            "post_logits": [],
            "prior_sample": [],
            "post_sample": [],
        }

        # Iterate over time dimension
        for t in range(T):
            prev_action_t = prev_actions[:, t]
            encoded_obs_t = encoded_obs[:, t]

            out = self.obs_step(
                prev_action=prev_action_t,
                encoded_obs=encoded_obs_t,
                prev_deter=prev_deter,
                prev_post=prev_post,
            )

            # Update carry for next timestep
            prev_deter, prev_post = out["deter"], out["post_sample"]

            # Append outputs
            for k in seq_outputs.keys():
                seq_outputs[k].append(out[k])

        # Stack outputs along time dimension
        seq_outputs = {k: torch.stack(v, dim=1) for k, v in seq_outputs.items()}

        return seq_outputs

    def obs_step(
        self,
        prev_action: torch.Tensor,
        encoded_obs: torch.Tensor,
        prev_deter: torch.Tensor,
        prev_post: torch.Tensor,
        sample_latent: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            prev_action (torch.Tensor): previous action of shape (B, self._action_size)

            encoded_obs (torch.Tensor): current encoded obs of shape (B, self._encoded_size)

        """

        # (B, n_dist, n_cats) -> (B, n_dist * n_cats)
        prev_post = prev_post.reshape(prev_post.shape[0], self._stoch_dim)

        deter = self._get_deterministic_latent(prev_deter, prev_post, prev_action)
        prior_dist = self._get_prior_dist(deter)
        post_dist = self._get_post_dist(deter, encoded_obs)

        if sample_latent:
            # Samples are of shape (B, n_dist, n_cats)
            prior_sample = prior_dist.sample()
            post_sample = post_dist.sample()
        else:
            prior_sample = prior_dist.mode
            post_sample = post_dist.mode

        prior_logits = prior_dist.logits
        post_logits = post_dist.logits

        return {
            "deter": deter,
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "prior_sample": prior_sample,
            "post_sample": post_sample,
        }

    def img_step(
        self,
        prev_action: torch.Tensor,
        prev_deter: torch.Tensor,
        prev_stoch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Single "imagination" step, which computes the deterministic state and the
        prior stochastic state.

        Args:
            prev_action (torch.Tensor): previous actions of shape (B, self._action_size)

            prev_deter (torch.Tensor): previous deter latent of shape (B, self._deter_size)

            prev_stoch (torch.Tensor): previous stoch latent of shape (B, self._n_dists, self._n_cats)
        """

        # (B, n_dist, n_cats) -> (B, n_dist * n_cats)
        prev_stoch = prev_stoch.reshape(prev_stoch.shape[0], self._stoch_dim)

        deter = self._get_deterministic_latent(prev_deter, prev_stoch, prev_action)
        prior_dist = self._get_prior_dist(deter)

        # Samples are of shape (B, n_dist, n_cats)
        prior_sample = prior_dist.sample()

        prior_logits = prior_dist.logits

        return {
            "deter": deter,
            "prior_logits": prior_logits,
            "prior_sample": prior_sample,
        }

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "block_gru": self._block_gru.state_dict(),
            "prior_logit_network": self._prior_logit_network.state_dict(),
            "post_logit_network": self._post_logit_network.state_dict(),
        }

    def load_state_dict(
        self,
        state: Dict[str, Dict[str, torch.Tensor]],
        strict: bool = True,
    ):
        self._block_gru.load_state_dict(state["block_gru"], strict=strict)
        self._prior_logit_network.load_state_dict(
            state["prior_logit_network"], strict=strict
        )
        self._post_logit_network.load_state_dict(
            state["post_logit_network"], strict=strict
        )

    def imagine_sequence(
        self,
        starting_deter: torch.Tensor,
        starting_stoch: torch.Tensor,
        length: int,
        policy: Optional[Callable] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagines a sequence of latent states from a given set of starting states. Uses either a callable
        policy that takes as input a latent state and outputs an action, or executes pre-defined
        actions in the form of a Tensor.

        Args:
            starting_deter (torch.Tensor) starting (batch of) latent states to imagine from. Of shape
            (B, D)

            starting_stoch (torch.Tensor) starting (batch of) stochastic state samples to imageine from.
            of shape (B, N, S)

            length (int): number of steps from current state to imagine.

            policy (Optional[Callable]) policy that is callable with a batch of latent states to output
            an action. Defaults to None. Either the policy or actions must be provided.

            actions (Optional[torch.Tensor]): fixed actions to imagine with at each timestep. Must be
            of shape (B, length, D)
        """

        prev_deter = starting_deter
        prev_stoch = starting_stoch

        seq_outputs = {
            "deter": [starting_deter],
            "prior_logits": [],
            "prior_sample": [starting_stoch],
            "action": [],
        }

        for t in range(length):
            if policy is not None:
                latent = combine_det_and_stoch(prev_deter, prev_stoch)
                prev_action = policy(latent.detach())
            elif actions is not None:
                assert actions.shape[1] == length, (
                    f"Invalid number of actions given: {actions.shape}"
                )
                prev_action = actions[:, t].detach()
            else:
                raise ValueError(
                    "Must provide either a callable policy or a tensor of actions"
                )
            out = self.img_step(prev_action, prev_deter, prev_stoch)

            prev_deter = out["deter"]
            prev_stoch = out["prior_sample"]
            out["action"] = prev_action

            for k in seq_outputs.keys():
                seq_outputs[k].append(out[k])

        # Stack outputs along time dimension
        seq_outputs = {k: torch.stack(v, dim=1) for k, v in seq_outputs.items()}

        return seq_outputs
