import torch
import torch.nn as nn


class WNConv2d(nn.Module):
    """Weight-normalized 2d convolution.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (int): Side length of each convolutional kernel.
        padding (int): Padding to add on edges of input.
        bias (bool): Use bias in the convolution operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True, init_zeros=False):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

        # init last layer w zeros to init a flow w identity mapping
        if init_zeros:
            torch.nn.init.zeros_(self.conv.weight)
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        x = self.conv(x)

        return x

