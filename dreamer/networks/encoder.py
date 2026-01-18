from typing import List, Tuple, Optional, Dict

from dataclasses import dataclass, asdict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared import MLP, RMSNormWrapper, MLPParams, truncated_normal_weight_init
from dreamer.distributions.dist_utils import symlog


@dataclass
class CNNParams:
    initial_depth: int
    kernel_size: int
    stride: int
    min_res: int
    bias: bool
    norm: bool
    act_func: str
    max_pool: bool
    depth_mult: Tuple[int, ...]
    max_pool_stride: Optional[int] = None
    max_pool_kernel: Optional[int] = None
    dilation: int = 1
    image_shape: Tuple[int, int, int] = (64, 64, 3)


@dataclass
class EncoderParams:
    image_keys: List[str]
    image_shapes: List[Tuple[int, int, int]]
    cnn_params: CNNParams
    vector_keys: Optional[List[str]] = None
    vector_shapes: Optional[List[int]] = None
    mlp_params: Optional[MLPParams] = None
    symlog_vecs: bool = True


class Encoder(nn.Module):
    """
    Encoder class to embed vector observations and image observations
    to a single embedding vector.
    """

    def __init__(
        self,
        image_keys: List[str],
        image_shapes: List[Tuple[int, int, int]],
        cnn_params: CNNParams,
        vector_keys: Optional[List[str]] = None,
        vector_shapes: Optional[List[int]] = None,
        mlp_params: Optional[MLPParams] = None,
        symlog_vecs: bool = True,
    ) -> None:
        super().__init__()

        self._image_keys, self._image_shapes = zip(
            *sorted(zip(image_keys, image_shapes))
        )
        # Assert that there is only one h,w element in the set of all h,w.
        # I.e., a constant h,w for all image shapes
        assert len({(h, w) for h, w, _ in self._image_shapes}) == 1, self._image_shapes

        total_channels = sum(i[-1] for i in self._image_shapes)

        self._cnn_params = cnn_params
        self._cnn_params.image_shape = (
            self._image_shapes[0][0],
            self._image_shapes[0][1],
            total_channels,
        )
        self._total_channels = total_channels
        self._image_keys = list(self._image_keys)
        self._cnn_encoder = CNNEncoder(**asdict(self._cnn_params))

        self._vector_keys = vector_keys
        self._symlog_vecs = symlog_vecs
        if vector_keys is not None:
            assert vector_shapes is not None
            assert mlp_params is not None
            self._vector_keys, self._vector_shapes = zip(
                *sorted(zip(vector_keys, vector_shapes))
            )
            vector_dim = sum(i for i in self._vector_shapes)
            self._mlp_params = mlp_params
            self._mlp_params.input_dim = vector_dim
            self._vector_keys = list(self._vector_keys)
            self._mlp_encoder = MLP(**asdict(self._mlp_params))

        self._encoded_dim = self._cnn_encoder.encoded_dim
        if vector_keys is not None:
            self._encoded_dim += self._mlp_params.out_dim

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encodes the data into a single vector.

        Args:
            data (Dict[str, torch.Tensor]): k:v correpsonds to
            data name string and corresponding data tensor.
            each data tensor is assumed to be either of shape
            (B, T, ...) or (B, ...).

        Returns:
            torch.Tensor of shape (B, E) or (B, T, E)
        """
        image_data = [data[k] for k in self._image_keys]
        vector_data = None
        if self._vector_keys is not None:
            vector_data = [data[k] for k in self._vector_keys]

        if len(image_data[0].shape) == 5:
            B, T, H, W, _ = image_data[0].shape
        else:
            T = None
            B, H, W, _ = image_data[0].shape

        # Concatenate over channel dimension
        image_data = torch.concatenate(image_data, dim=-1)
        if T is not None:
            image_data = image_data.reshape((B * T, H, W, self._total_channels))
        if vector_data is not None:
            expected_shape = 3 if T is not None else 2
            assert len(vector_data[0].shape) == expected_shape
            vector_data = torch.concatenate(vector_data, dim=-1)
            if T is not None:
                vector_data = vector_data.reshape((B * T, -1))
            if self._symlog_vecs:
                vector_data = symlog(vector_data)

        encoded_imgs = self._cnn_encoder(image_data)
        encoded_vecs = None

        if vector_data is not None:
            encoded_vecs = self._mlp_encoder(vector_data)

        if T is not None:
            encoded_imgs = encoded_imgs.reshape((B, T, -1))
            if encoded_vecs is not None:
                encoded_vecs = encoded_vecs.reshape((B, T, -1))
        embed = encoded_imgs
        if encoded_vecs is not None:
            embed = torch.concatenate([encoded_imgs, encoded_vecs], dim=-1)
        return embed


class PadModule(nn.Module):
    """
    Torch Module Wrapper for adding padding to an input
    """

    def __init__(
        self, pad_left: int, pad_right: int, pad_top: int, pad_bot: int
    ) -> None:
        super().__init__()
        self._pad_left = pad_left
        self._pad_right = pad_right
        self._pad_top = pad_top
        self._pad_bot = pad_bot

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Adds padding to the given input.

        Args:
            input_tensor (torch.Tensor of shape (..., H, W, C))
        """

        return F.pad(
            input_tensor,
            [self._pad_left, self._pad_right, self._pad_top, self._pad_bot],
        )


class CNNEncoder(nn.Module):
    """CNN Encoder network. Pytorch's 'same' padding doesn't work for stride > 1,
    so this class implements tensorflow's version to be consistent with the JAX Dreamer.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        initial_depth: int,
        kernel_size: int,
        stride: int,
        min_res: int,
        bias: bool,
        norm: bool,
        act_func: str,
        max_pool: bool,
        depth_mult: Tuple[int, ...],
        max_pool_stride: Optional[int] = None,
        max_pool_kernel: Optional[int] = None,
        dilation: int = 1,
    ) -> None:
        """

        Args:
            image_shape (Tuple[int, int, int]): (H, W, C)

            initial_depth (int): Depth of first convolution operation

            kernel_size (int): convolution kernel size, assumed to be a square kernel

            stride (int): convolution stride

            min_res (int): minimum h/w to shrink image to through convolutions

            bias (bool): whether to use a bias in the network

            norm (bool): whether to use rms normalisation.

            act_func (str): string name of activation function. Must match a PyTorch function

            max_pool (bool): whether to shrink dimensions using max pooling instead of
                             striding. If true, ignores striding.

            depth_mult (Tuple[int, ...]): the new depth at each layer of the network, fournd
                                     by multiplying initial_depth by depth_mult[i]

            dilation (int, defaults to 1): dilation of the convolution operation.
        """

        super().__init__()

        assert len(image_shape) == 3, f"Image must be 3D, provided shape: {image_shape}"
        assert image_shape[0] == image_shape[1], (
            f"Resolution must be square, provided shape: {image_shape}"
        )
        if max_pool:
            assert stride == 1, "Stride must be equal to 1 if doing max pooling"
            assert max_pool_stride is not None
            assert max_pool_kernel is not None

        self._image_shape = image_shape
        self._initial_depth = initial_depth
        self._kernel_size = kernel_size
        self._stride = stride
        self._min_res = min_res
        self._use_bias = bias
        self._norm = norm
        self._act_func_name = act_func
        self._use_max_pool = max_pool
        self._max_pool_stride = max_pool_stride
        self._max_pool_kernel = max_pool_kernel
        self._dilation = dilation

        self._encoded_res = None
        self._encoded_dim = None

        self._depth_mults = depth_mult

        self._n_conv_layers = self._calculate_n_conv_layers()

        assert len(self._depth_mults) == self._n_conv_layers

        self._network = self._configure_network()
        self._network.apply(truncated_normal_weight_init)

    def _configure_network(self) -> nn.Sequential:
        """
        Configures and returns the CNN Encoder network
        """

        activation_function = getattr(nn, self._act_func_name)

        current_in_ch = self._image_shape[-1]
        current_res = self._image_shape[0]

        layers = []
        for layer in range(self._n_conv_layers):
            pad_left, pad_right, pad_top, pad_bot = self._calculate_same_padding(
                (current_res, current_res)
            )
            layers.append(PadModule(pad_left, pad_right, pad_top, pad_bot))
            layers.append(
                nn.Conv2d(
                    current_in_ch,
                    self._initial_depth * self._depth_mults[layer],
                    self._kernel_size,
                    self._stride,
                    padding=0,  # We've already handled padding the JAX way
                    dilation=self._dilation,
                    bias=self._use_bias,
                )
            )
            if self._use_max_pool and self._max_pool_kernel and self._max_pool_stride:
                layers.append(
                    nn.MaxPool2d(
                        self._max_pool_kernel, self._max_pool_stride, padding=0
                    )
                )

            if self._norm:
                # (B, C, H, W) -> (B, H, W, C)
                layers.append(
                    RMSNormWrapper(
                        self._initial_depth * self._depth_mults[layer],
                        permute=[0, 2, 3, 1],
                    )
                )
            layers.append(activation_function())

            current_res = self._calculate_output_dim(current_res)
            current_in_ch = self._initial_depth * self._depth_mults[layer]
        return nn.Sequential(*layers)

    def _calculate_same_padding(
        self, input_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calculates the amount of padding to apply across height and width dimensions.

        A re-implementation of the "same" padding method, as PyTorch only defines it
        for stride=1, whereas JAX/TensorFlow define it that output_dim = ceil(input_dim/stride)

        Returns Tuple[int, int, int, int]: padding_left, padding_right, padding_top, padding_bottom
        """

        def calc_half_pad(
            input_dim: int, kernel: int, stride: int, dilation: int
        ) -> int:
            """
            Calculates a single side (h/w) padding.

            Returns: int required padding
            """
            total_pad = max(
                (math.ceil(input_dim / stride) - 1) * stride
                + (kernel - 1) * dilation
                + 1
                - input_dim,
                0,
            )
            return total_pad

        kernel = self._kernel_size
        stride = self._stride

        total_width_padding = calc_half_pad(
            input_size[0], kernel, stride, self._dilation
        )

        total_height_padding = calc_half_pad(
            input_size[1], kernel, stride, self._dilation
        )

        left_pad, right_pad = total_width_padding // 2, total_width_padding // 2
        if total_width_padding % 2 != 0:
            right_pad += 1

        top_pad, bot_pad = total_height_padding // 2, total_height_padding // 2
        if total_height_padding % 2 != 0:
            bot_pad += 1

        return (left_pad, right_pad, top_pad, bot_pad)

    def _calculate_output_dim(self, input_res: int) -> int:
        """
        Calculates the output resolution of applying one convolution,
        using the formula given https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Since we assume a square image, this only needs to be calculated once

        Args:
            input_res (int): resolution before applying convolution
        """

        if self._use_max_pool:
            assert self._max_pool_kernel is not None
            assert self._max_pool_stride is not None
            return math.floor(
                (input_res - self._max_pool_kernel) / self._max_pool_stride + 1
            )

        pad_left, pad_right, pad_top, pad_bot_ = self._calculate_same_padding(
            (input_res, input_res)
        )
        required_padding = pad_left + pad_right
        kernel = self._max_pool_kernel if self._use_max_pool else self._kernel_size
        stride = self._max_pool_stride if self._use_max_pool else self._stride

        return math.floor(
            (
                (input_res + required_padding - self._dilation * (kernel - 1) - 1)
                / stride
            )
            + 1
        )

    def _calculate_n_conv_layers(self) -> int:
        """
        Calculates the number of convolution layers required
        to go from the initial input resolution to the specified
        minimum resolution.

        This should always match the same padding tensorflow implementation
        that yields:

        out_height = ceil(float(in_height) / float(strides[1]))
        out_width  = ceil(float(in_width) / float(strides[2]))

        https://tensorflow2.readthedocs.io/en/latest/tensorflow/g3doc/api_docs/python/nn.html

        """

        n_layers = 0
        current_res = self._image_shape[0]
        while current_res >= self._min_res:
            new_res = self._calculate_output_dim(current_res)
            print(f"CURRENT RES: {current_res}, NEW RES: {new_res}")
            if new_res < self._min_res:
                break
            n_layers += 1
            current_res = new_res

        self._encoded_res = max(current_res, self._min_res)
        self._encoded_dim = (
            self._encoded_res
            * self._encoded_res
            * self._initial_depth
            * self._depth_mults[-1]
        )

        return n_layers

    @property
    def encoded_res(self) -> int:
        """
        Property that returns the encoded resolution
        after all passes of the CNN have been completed.
        """
        if self._encoded_res is None:
            raise AttributeError(
                "Encoded res should never be none as it is set in the constructor"
            )
        return self._encoded_res

    @property
    def encoded_dim(self) -> int:
        """
        Gets the total vector dimension of the encoder output
        assuming output will be flattened.
        """
        if self._encoded_dim is None:
            raise AttributeError(
                "Encoded dim should never be none as it is set in the constructor"
            )
        return self._encoded_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor) of shape (B, H, W, C)
        """
        # Normalise to range [-0.5, 0.5]
        images = images / 255 - 0.5

        B, H, W, C = images.shape

        # (B, H, W, C) -> (B*T, C, H, W)
        images = images.permute(0, 3, 1, 2)

        output = self._network(images)
        # (B*T, C, H, W) -> (B*T, -1)
        output = output.reshape(B, -1)
        return output
