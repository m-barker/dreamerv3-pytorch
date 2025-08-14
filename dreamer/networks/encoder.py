from typing import Tuple
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
    ) -> None:
        """

        Args:
            image_shape: (H, W, C)
            initial_depth: Depth of first convolution operation
            kernel_size: convolution kernel size
            stride: convolution stride
            min_res: minimum h/w to shrink image to through convolutions
            bias: whether to use a bias in the network
            layer_norm: whether to use layer normalisation
            act_func: string name of activation function. Must match a PyTorch function
        """

        super().__init__()

        assert len(image_shape) == 3, f"Image must be 3D, provided shape: {image_shape}"

        self._image_shape = image_shape
        self._initial_depth = initial_depth
        self._kernel_size = kernel_size
        self._stride = stride
        self._min_res = min_res
        self._use_bias = bias
        self._do_layer_norm = layer_norm
        self._act_func_name = act_func
