import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_ssl.resnet_realnvp.resnet_util import WNConv2d


class ConvNetCoupling(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 num_blocks, kernel_size, padding, init_zeros=False):
        super(ConvNetCoupling, self).__init__()
        
        layers_list = [WNConv2d(in_channels, mid_channels, kernel_size, padding, bias=True)]
        layers_list += [WNConv2d(mid_channels, mid_channels, kernel_size, padding, bias=True)
                        for _ in range(num_blocks)]
        layers_list += [WNConv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True, init_zeros=init_zeros)]
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.layers(x)
        return x
