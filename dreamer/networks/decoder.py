from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared import RMSNormWrapper


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
        x = x.repeat((1, 1, 2, 2))
        return x


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
        norm: str,
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

            norm (str): normalisation method to use. Currently can either
            be "none" or "rms"

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

        assert norm in ("none", "rms"), f"Unhandled norm method {norm} probided"

        self._image_shape = image_shape
        self._starting_res = starting_res
        self._latent_dim = latent_dim
        self._kernel_size = kernel_size
        # Reverse this, as want to reverse the CNN encoder
        # Drop the last one, as we want the final CNN to have a
        # depth equal to the number of image channels.
        self._depth_mults = reversed(list(depth_mults[:-1]))
        self._starting_depth = depth_mults[-1] * starting_depth
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

    def _configure_cnn_network(self) -> nn.Sequential:
        layers = []
        activation_function = getattr(nn, self._act_func)
        current_depth = self._starting_depth

        for i, depth_mult in enumerate(self._depth_mults):
            layers.append(RepeatLayer())
            layers.append(
                nn.Conv2d(
                    in_channels=current_depth,
                    out_channels=current_depth * depth_mult,
                    kernel_size=self._kernel_size,
                    stride=1,
                    bias=self._bias,
                    padding="same",  # Can use same since we enforce stride=1
                )
            )
            if self._norm != "none":
                if self._norm == "rms":
                    # (B, C, H, W) -> (B, H, W, C)
                    layers.append(
                        RMSNormWrapper(
                            current_depth * depth_mult,
                            permute=[0, 2, 3, 1],
                        )
                    )
            layers.append(activation_function())
            current_depth *= depth_mult

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
            latent_state (torch.Tensor): shape (B, T, D)
        """
        B, T, _ = latent_state.shape
        x = latent_state.reshape((B * T, -1))
        x = self._linear_layer(x)
        x = x.reshape(
            (B * T, self._starting_depth, self._starting_res, self._starting_res)
        )
        decoded_images = self._cnn_network(x)

        if self._final_sigmoid:
            decoded_images = F.sigmoid(decoded_images)

        decoded_images = decoded_images.reshape(
            (B, T, self._image_shape[0], self._image_shape[1], self._image_shape[2])
        )

        return decoded_images
