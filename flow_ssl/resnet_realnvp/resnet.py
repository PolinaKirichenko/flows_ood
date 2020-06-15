import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_ssl.resnet_realnvp.resnet_util import WNConv2d


class ResidualBlock(nn.Module):
    """ResNet basic block with weight norm."""
    def __init__(self, in_channels, out_channels, use_batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.in_norm = nn.BatchNorm2d(in_channels)
        self.in_conv = WNConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        if use_batch_norm:
            self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_conv = WNConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        skip = x

        if self.use_batch_norm:
            x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        if self.use_batch_norm:
            x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        x = x + skip

        return x


class ResNet(nn.Module):
    """ResNet for scale and translate factors in Real NVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        out_channels (int): Number of channels in the output.
        num_blocks (int): Number of residual blocks in the network.
        kernel_size (int): Side length of each filter in convolutional layers.
        padding (int): Padding for convolutional layers.
        double_after_norm (bool): Double input after input BatchNorm.
    """
    def __init__(self, in_channels, mid_channels, out_channels,
                 num_blocks, kernel_size, padding, double_after_norm, init_zeros=False,
                 use_batch_norm=True, skip=True):
        super(ResNet, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.double_after_norm = double_after_norm
        self.skip = skip

        self.in_conv = WNConv2d(2*in_channels, mid_channels, kernel_size, padding, bias=True)
        self.blocks = nn.ModuleList([ResidualBlock(mid_channels, mid_channels, use_batch_norm)
                                     for _ in range(num_blocks)])
        self.out_conv = WNConv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True, init_zeros=init_zeros)

        if skip:
            self.in_skip = WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)
            self.skips = nn.ModuleList([WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)
                                    for _ in range(num_blocks)])
        else:
            self.skips = [None for _ in range(num_blocks)]

        if use_batch_norm:
            self.in_norm = nn.BatchNorm2d(in_channels)
            self.out_norm = nn.BatchNorm2d(mid_channels)

    def forward(self, x):
        if self.use_batch_norm:
            x = self.in_norm(x)
            if self.double_after_norm:
                x *= 2.

        x = torch.cat((x, -x), dim=1)

        x = F.relu(x)
        x = self.in_conv(x)
        if self.skip:
            x_skip = self.in_skip(x)

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            if self.skip:
                x_skip += skip(x)

        if self.skip:
            x = x_skip

        if self.use_batch_norm:
            x = self.out_norm(x)

        x = F.relu(x)
        x = self.out_conv(x)

        return x


class ResNetAE(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 num_blocks, kernel_size, padding, double_after_norm, latent_dim,
                 init_zeros=False, use_batch_norm=True, img_width=28):
        super(ResNetAE, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.double_after_norm = double_after_norm
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim
        self.mid_channels = mid_channels
        self.img_width = img_width

        self.in_conv = WNConv2d(2*in_channels, mid_channels, kernel_size, padding, bias=True)
        self.blocks = nn.ModuleList([ResidualBlock(mid_channels, mid_channels, use_batch_norm)
                                     for _ in range(num_blocks)])
        self.out_conv = WNConv2d(mid_channels, out_channels, kernel_size=1, padding=0,
                                 bias=True, init_zeros=init_zeros)

        # TODO: change to weight norm?
        self.bottleneck = nn.Sequential(nn.Linear(mid_channels*img_width*img_width, latent_dim),
                                        nn.Linear(latent_dim, mid_channels*img_width*img_width))

        if use_batch_norm:
            self.in_norm = nn.BatchNorm2d(in_channels)
            self.out_norm = nn.BatchNorm2d(mid_channels)

    def forward(self, x):
        if self.use_batch_norm:
            x = self.in_norm(x)
            if self.double_after_norm:
                x *= 2.

        x = torch.cat((x, -x), dim=1)

        x = F.relu(x)
        x = self.in_conv(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.num_blocks // 2 - 1:
                x = torch.flatten(x, start_dim=1)
                x = self.bottleneck(x)
                x = x.view(-1, self.mid_channels, self.img_width, self.img_width)

        if self.use_batch_norm:
            x = self.out_norm(x)

        x = F.relu(x)
        x = self.out_conv(x)

        return x
