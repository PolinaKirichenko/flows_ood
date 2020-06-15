import torch
import torch.nn as nn
import torch.nn.functional as F

# from flow_ssl.resnet_realnvp.resnet_util import WNConv2d
from flow_ssl.convnet_coupling import Conv2d, Conv2dZeros


class HighwayLayer(nn.Module):
    """Highway layer with weight norm."""
    def __init__(self, in_channels, out_channels):
        super(HighwayLayer, self).__init__()
        self.conv_h = Conv2d(in_channels, out_channels, do_actnorm=False)
        self.conv_t = Conv2d(in_channels, out_channels, do_actnorm=False)

    def forward(self, x):
        gate = F.sigmoid(self.conv_t(x))
        out = self.conv_h(x) * gate + x * (1 - gate)
        return out


class HighwayNetwork(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_blocks):
        super(HighwayNetwork, self).__init__()
        self.in_conv = Conv2d(in_channels, mid_channels, do_actnorm=False)
        self.layers = nn.Sequential(*[HighwayLayer(mid_channels, mid_channels) for _ in range(num_blocks)])
        self.out_conv = Conv2dZeros(mid_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.layers(x)
        x = self.out_conv(x)
        return x
