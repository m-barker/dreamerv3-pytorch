from typing import Tuple

import math
import torch.nn as nn


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
        layer_norm: bool,
        act_func: str,
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

            layer_norm (bool): whether to use layer normalisation

            act_func (str): string name of activation function. Must match a PyTorch function

            dilation (int, defaults to 1): dilation of the convolution operation.
        """

        super().__init__()

        assert len(image_shape) == 3, f"Image must be 3D, provided shape: {image_shape}"
        assert image_shape[0] == image_shape[1], (
            f"Resolution must be square, provided shape: {image_shape}"
        )

        self._image_shape = image_shape
        self._initial_depth = initial_depth
        self._kernel_size = kernel_size
        self._stride = stride
        self._min_res = min_res
        self._use_bias = bias
        self._do_layer_norm = layer_norm
        self._act_func_name = act_func
        self._dilation = dilation

        self._encoded_res = None

        self._n_conv_layers = self._calculate_n_conv_layers()

    def _calulate_same_padding(
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

        total_width_padding = calc_half_pad(
            input_size[0], self._kernel_size, self._stride, self._dilation
        )

        total_height_padding = calc_half_pad(
            input_size[1], self._kernel_size, self._stride, self._dilation
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

        pad_left, pad_right, pad_top, pad_bot_ = self._calulate_same_padding(
            (input_res, input_res)
        )
        required_padding = pad_left + pad_right

        return math.floor(
            (
                (
                    input_res
                    + required_padding
                    - self._dilation * (self._kernel_size - 1)
                    - 1
                )
                / self._stride
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
            if new_res < self._min_res:
                break
            n_layers += 1
            current_res = new_res

        self._encoded_res = current_res

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
