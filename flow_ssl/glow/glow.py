import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_ssl.realnvp.coupling_layer import CouplingLayer
from flow_ssl.realnvp.coupling_layer import MaskChannelwise

from flow_ssl.invertible import iSequential
from flow_ssl.invertible.downsample import iLogits
from flow_ssl.invertible.downsample import keepChannels
from flow_ssl.invertible.downsample import SqueezeLayer
from flow_ssl.invertible.parts import addZslot
from flow_ssl.invertible.parts import FlatJoin
from flow_ssl.invertible.parts import passThrough
from flow_ssl.glow.glow_utils import InvertibleConv1x1, ActNorm2d


class GlowBase(nn.Module):
    
    def __init__(self):
        super(GlowBase, self).__init__()

    def forward(self,x):
        return self.body(x)

    def logdet(self):
        return self.body.logdet()

    def inverse(self,z):
        return self.body.inverse(z)

    @staticmethod
    def _glow_step(in_channels, mid_channels, actnorm_scale, st_type, num_layers):
        layers = [
                ActNorm2d(in_channels, actnorm_scale),
                InvertibleConv1x1(in_channels),
                CouplingLayer(in_channels, mid_channels, num_layers,
                    MaskChannelwise(reverse_mask=False), st_type=st_type),
        ]
        return layers


class Glow(GlowBase):

    def __init__(self, image_shape, mid_channels=64, num_scales=2, num_coupling_layers_per_scale=8,
            num_layers=3, actnorm_scale=1., multi_scale=True, st_type='glow_convnet'):
        super(Glow, self).__init__()

        layers = [addZslot(), passThrough(iLogits())]
        self.output_shapes = []

        C, H, W = image_shape

        for scale in range(num_scales):
            # Squeeze
            C, H, W = C * 4, H // 2, W // 2
            layers.append(passThrough(SqueezeLayer(downscale_factor=2)))
            self.output_shapes.append([-1, C, H, W])

            # Flow steps
            for _ in range(num_coupling_layers_per_scale):
                layers.append(
                    passThrough(*self._glow_step(
                        in_channels=C, mid_channels=mid_channels,
                        actnorm_scale=actnorm_scale, st_type=st_type,
                        num_layers=num_layers))
                    )
                self.output_shapes.append([-1, C, H, W])

            # Split and factor out
            if multi_scale:
                if scale < num_scales - 1:
                    layers.append(keepChannels(C//2))
                    self.output_shapes.append([-1, C//2, H, W])
                    C = C // 2

        layers.append(FlatJoin())
        self.body = iSequential(*layers)

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True
