from typing import Optional, Tuple, List, Dict, Union

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributions.distributions import MSEDist, SymlogDist

from .shared import (
    MLP,
    MLPParams,
    RMSNormWrapper,
    truncated_normal_weight_init,
)


@dataclass
class DecoderCNNParams:
    starting_res: int
    latent_dim: int
    kernel_size: int
    starting_depth: int
    depth_mults: Tuple[int, ...]
    bias: bool
    norm: bool
    act_func: str
    final_sigmoid: bool
    image_shape: Tuple[int, int, int] = (64, 64, 3)


@dataclass
class DecoderParams:
    image_keys: List[str]
    image_shapes: List[Tuple[int, int, int]]
    cnn_params: DecoderCNNParams
    pixel_loss_agg: str = "sum"
    vector_keys: Optional[List[str]] = None
    vector_shapes: Optional[List[int]] = None
    mlp_params: Optional[MLPParams] = None
    symlog_vecs: bool = True


class Decoder(nn.Module):
    """
    Decoder class to take (sequences of) a batch of latent states
    and output reconstructed image observations, and optionally vector
    observations.
    """

    def __init__(
        self,
        image_keys: List[str],
        image_shapes: List[Tuple[int, int, int]],
        cnn_params: DecoderCNNParams,
        pixel_loss_agg: str = "sum",
        vector_keys: Optional[List[str]] = None,
        vector_shapes: Optional[List[int]] = None,
        mlp_params: Optional[MLPParams] = None,
        symlog_vecs: bool = True,
    ):
        super().__init__()
        self._image_keys, self._image_shapes = zip(
            *sorted(zip(image_keys, image_shapes))
        )
        # Assert that there is only one h,w element in the set of all h,w.
        # I.e., a constant h,w for all image shapes
        assert len({(h, w) for h, w, _ in self._image_shapes}) == 1, self._image_shapes

        assert pixel_loss_agg in ("sum", "mean"), (
            f"Invalid pixel loss aggregation method: {pixel_loss_agg}"
        )

        total_channels = sum(i[-1] for i in self._image_shapes)

        self._cnn_params = cnn_params
        self._cnn_params.image_shape = (
            self._image_shapes[0][0],
            self._image_shapes[0][1],
            total_channels,
        )
        self._total_channels = total_channels
        self._image_keys = list(self._image_keys)
        self._cnn_decoder = CNNDecoder(**asdict(self._cnn_params))
        self._pixel_loss_agg = pixel_loss_agg

        self._vector_keys = vector_keys
        if vector_keys is not None:
            assert vector_shapes is not None
            assert mlp_params is not None
            self._vector_keys, self._vector_shapes = zip(
                *sorted(zip(vector_keys, vector_shapes))
            )
            vector_dim = sum(i for i in self._vector_shapes)
            self._mlp_params = mlp_params
            self._mlp_params.out_dim = vector_dim
            self._vector_keys = list(self._vector_keys)
            self._mlp_decoder = MLP(**asdict(self._mlp_params))
        self._symlog_vecs = symlog_vecs

    def forward(
        self, latent_states: torch.Tensor
    ) -> Dict[str, Union[MSEDist, SymlogDist]]:
        """
        Args:
            latent_states (torch.Tensor) latent states of shape (B, T, D) or
            (B, D)

        Returns:
            Dict[str, torch.Tensor] dictionary of decoded observations, where
            the key is the name of the observation. Each tensor used to construct
            the distribution is either of shape (B, T, ...) or (B, ...) where ...
            is the dimensionality of the observation.
        """

        T = None
        if len(latent_states.shape) == 2:
            B, _ = latent_states.shape
        elif len(latent_states.shape) == 3:
            B, T, _ = latent_states.shape
        else:
            raise ValueError(f"Invalid latent state shape, of {latent_states.shape}")

        if T is not None:
            latent_states = latent_states.reshape((B * T, -1))

        decoded_imgs = self._cnn_decoder(latent_states)
        if T is not None:
            H, W, C = decoded_imgs.shape[1:]
            decoded_imgs = decoded_imgs.reshape((B, T, H, W, C))

        decoded_vecs = None
        if self._vector_keys is not None:
            decoded_vecs = self._mlp_decoder(latent_states)
            if T is not None:
                decoded_vecs = decoded_vecs.reshape((B, T, -1))

        channel_start_index = 0
        decoder_dict = {}
        for name, shape in zip(self._image_keys, self._image_shapes):
            n_channels = shape[-1]
            decoded_image = decoded_imgs[
                ..., channel_start_index : n_channels + channel_start_index
            ]
            dist = MSEDist(decoded_image, self._pixel_loss_agg)
            decoder_dict[name] = dist

            channel_start_index += n_channels

        if decoded_vecs is not None:
            vec_dim_start_index = 0
            assert self._vector_keys is not None
            for name, shape in zip(self._vector_keys, self._vector_shapes):
                n_dims = shape
                decoded_vec = decoded_vecs[
                    ..., vec_dim_start_index : vec_dim_start_index + n_dims
                ]
                if self._symlog_vecs:
                    dist = SymlogDist(decoded_vec)
                else:
                    dist = MSEDist(decoded_vec)
                decoder_dict[name] = dist
                vec_dim_start_index += n_dims

        return decoder_dict


class RepeatLayer(nn.Module):
    """
    nn.Module wrapper to double heigh/width of the current
    image by repeating the pixels.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Doubles the height and width dimensions by repeating
        the pixels, as done in official JAX Dreamer implementation

        Args:
            x (torch.Tensor): shape (B, C, H, W)
        """
        return F.interpolate(x, scale_factor=2, mode="nearest")


class CNNDecoder(nn.Module):
    """
    CNN Decoder class. To be consistent with the official
    Dreamer version, Striding is not used for upsampling,
    instead, pixels are repeated to double the H,W, and then
    a normal convolution is applied to that.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        starting_res: int,
        latent_dim: int,
        kernel_size: int,
        starting_depth: int,
        depth_mults: Tuple[int, ...],
        bias: bool,
        norm: bool,
        act_func: str,
        final_sigmoid: bool,
    ) -> None:
        """
        Args:
            image_shape (Tuple[int, int, int]): (H, W, C) of target decoded
            image

            starting_res (int): the starting resolution to upscale.

            latent_dim (int): the dimension of the latent state to decode

            kernel_size (int): kernel size of convolutions

            starting_depth (int): the starting depth of the CNN encoder
            convolution. Since we want to reverse the process.

            depth_mults (Tuple[float, ...]): the same depth mults
            given to the CNN encoder.

            bias (bool): whether to use a bias in network components.

            norm (bool): whether to do RMS Layer Norm after each layer.

            act_func (str): string name of activiation function. Must match
            a PyTorch function.

            final_sigmoid (bool): whether the final layer should be passed
            through a sigmoid function.

        """
        super().__init__()

        assert len(image_shape) == 3, f"Image must be 3D, provided shape: {image_shape}"
        assert image_shape[0] == image_shape[1], (
            f"Resolution must be square, provided shape: {image_shape}"
        )

        self._image_shape = image_shape
        self._starting_res = starting_res
        self._latent_dim = latent_dim
        self._kernel_size = kernel_size
        # Reverse this, as want to reverse the CNN encoder
        # Drop the last one, as we want the final CNN to have a
        # depth equal to the number of image channels.
        self._starting_depth = depth_mults[-1] * starting_depth
        self._init_depth = starting_depth
        self._depth_mults = reversed(list(depth_mults[:-1]))
        self._bias = bias
        self._norm = norm
        self._act_func = act_func
        self._final_sigmoid = final_sigmoid

        self._cnn_network = self._configure_cnn_network()
        self._linear_layer = nn.Linear(
            self._latent_dim,
            self._starting_depth * self._starting_res * self._starting_res,
            bias=bias,
        )
        self._cnn_network.apply(truncated_normal_weight_init)
        self._linear_layer.apply(truncated_normal_weight_init)

    def _configure_cnn_network(self) -> nn.Sequential:
        layers = []
        activation_function = getattr(nn, self._act_func)
        current_depth = self._starting_depth

        for i, depth_mult in enumerate(self._depth_mults):
            layers.append(RepeatLayer())
            layers.append(
                nn.Conv2d(
                    in_channels=current_depth,
                    out_channels=self._init_depth * depth_mult,
                    kernel_size=self._kernel_size,
                    stride=1,
                    bias=self._bias,
                    padding="same",  # Can use same since we enforce stride=1
                )
            )
            if self._norm:
                # (B, C, H, W) -> (B, H, W, C)
                layers.append(
                    RMSNormWrapper(
                        self._init_depth * depth_mult,
                        permute=[0, 2, 3, 1],
                    )
                )
            layers.append(activation_function())
            current_depth = self._init_depth * depth_mult

        # Final layer needs to have the same number of channels as the source image
        layers.append(RepeatLayer())
        layers.append(
            nn.Conv2d(
                in_channels=current_depth,
                out_channels=self._image_shape[-1],
                kernel_size=self._kernel_size,
                stride=1,
                bias=self._bias,
                padding="same",  # Can use same since we enforce stride=1
            )
        )

        return nn.Sequential(*layers)

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_state (torch.Tensor): shape (B,  D)
        """
        B, _ = latent_state.shape
        x = self._linear_layer(latent_state)
        x = x.reshape((B, self._starting_depth, self._starting_res, self._starting_res))
        decoded_images = self._cnn_network(x)

        if self._final_sigmoid:
            decoded_images = F.sigmoid(decoded_images)

        decoded_images = decoded_images.reshape(
            (B, self._image_shape[0], self._image_shape[1], self._image_shape[2])
        )

        return decoded_images
