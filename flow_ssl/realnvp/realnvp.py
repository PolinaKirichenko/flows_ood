import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_ssl.realnvp.coupling_layer import CouplingLayer
from flow_ssl.realnvp.coupling_layer import CouplingLayerTabular
from flow_ssl.realnvp.coupling_layer import MaskCheckerboard, MaskChannelwise, MaskTabular
from flow_ssl.realnvp.coupling_layer import MaskHorizontal, MaskVertical
from flow_ssl.realnvp.coupling_layer import MaskQuadrant, MaskSubQuadrant
from flow_ssl.realnvp.coupling_layer import MaskCenter

from flow_ssl.invertible import iSequential
from flow_ssl.invertible.downsample import iLogits
from flow_ssl.invertible.downsample import keepChannels
from flow_ssl.invertible.downsample import SqueezeLayer, iAvgPool2d
from flow_ssl.invertible.parts import addZslot
from flow_ssl.invertible.parts import FlatJoin
from flow_ssl.invertible.parts import passThrough
from flow_ssl.invertible.coupling_layers import iConv1x1
from flow_ssl.invertible import Swish, ActNorm1d, ActNorm2d


class RealNVPBase(nn.Module):

    def forward(self,x):
        return self.body(x)

    def logdet(self):
        return self.body.logdet()

    def inverse(self, z):
        return self.body.inverse(z)

    def nll(self,x,y=None,label_weight=1.):
        z = self(x)
        logdet = self.logdet()
        z = z.reshape((z.shape[0], -1))
        prior_ll = self.prior.log_prob(z, y,label_weight=label_weight)
        nll = -(prior_ll + logdet)
        return nll


class RealNVP(RealNVPBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_shape=(1, 28, 28), skip=True, latent_dim=100):
        super(RealNVP, self).__init__()

        layers = [addZslot(), passThrough(iLogits())]
        self.output_shapes = []
        _, _, img_width = img_shape

        for scale in range(num_scales):
            in_couplings = self._threecouplinglayers(in_channels, mid_channels, num_blocks, MaskCheckerboard,
                init_zeros, st_type, use_batch_norm, img_width, skip, latent_dim)
            layers.append(passThrough(*in_couplings))

            if scale == num_scales - 1:
                layers.append(passThrough(
                    CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True),
                        init_zeros, st_type, use_batch_norm, img_width, skip, latent_dim)))
            else:
                layers.append(passThrough(SqueezeLayer(2)))
                img_width = img_width // 2
                if st_type != 'autoencoder':  # in the autoencoder case we probably want the bottleneck size to be fixed?
                    mid_channels *= 2
                out_couplings = self._threecouplinglayers(4*in_channels, mid_channels, num_blocks,
                    MaskChannelwise, init_zeros, st_type, use_batch_norm, img_width, skip, latent_dim)
                layers.append(passThrough(*out_couplings))
                layers.append(keepChannels(2*in_channels))
                in_channels *= 2

        layers.append(FlatJoin())
        self.body = iSequential(*layers)

    @staticmethod
    def _threecouplinglayers(in_channels, mid_channels, num_blocks, mask_class, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_width=28, skip=True, latent_dim=100):
        layers = [
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm,
                    img_width=img_width, skip=skip, latent_dim=latent_dim),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=True),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm,
                    img_width=img_width, skip=skip, latent_dim=latent_dim),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm,
                    img_width=img_width, skip=skip, latent_dim=latent_dim)
        ]
        return layers


class RealNVPCycleMask(RealNVPBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_shape=(1, 28, 28), skip=True, latent_dim=None):
        super(RealNVPCycleMask, self).__init__()

        self.body = iSequential(
                addZslot(),
                passThrough(iLogits()),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                FlatJoin()
            )

class RealNVPSmall(RealNVPBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_shape=(1, 28, 28), skip=True, latent_dim=None):
        super(RealNVPSmall, self).__init__()

        self.body = iSequential(
                addZslot(),
                passThrough(iLogits()),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks,
                    MaskCheckerboard(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks,
                    MaskCheckerboard(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                FlatJoin()
            )


class RealNVP8Layers(RealNVPBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_shape=(1, 28, 28), skip=True, latent_dim=None):
        super(RealNVP8Layers, self).__init__()

        self.body = iSequential(
                addZslot(),
                passThrough(iLogits()),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                FlatJoin()
            )


class RealNVPMaskHorizontal(RealNVPBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_shape=(1, 28, 28), skip=True, latent_dim=None):
        super(RealNVPMaskHorizontal, self).__init__()

        self.body = iSequential(
                addZslot(),
                passThrough(iLogits()),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                FlatJoin()
            )


class RealNVPMaskHorizontal3Layers(RealNVPBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_shape=(1, 28, 28), skip=True, latent_dim=None):
        super(RealNVPMaskHorizontal3Layers, self).__init__()

        self.body = iSequential(
                addZslot(),
                passThrough(iLogits()),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                FlatJoin()
            )


class RealNVPTabular(RealNVPBase):

    def __init__(self, in_dim=2, num_coupling_layers=6, hidden_dim=256, 
                 num_layers=2, init_zeros=False, dropout=False):

        super(RealNVPTabular, self).__init__()
        
        self.body = iSequential(*[
                        CouplingLayerTabular(
                            in_dim, hidden_dim, num_layers, MaskTabular(reverse_mask=bool(i%2)), init_zeros=init_zeros, dropout=dropout)
                        for i in range(num_coupling_layers)
                    ])


#class RealNVPCenterCycleCenter(RealNVPBase):
#
#    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
#            st_type='resnet', use_batch_norm=True):
#        super(RealNVPCenterCycleCenter, self).__init__()
#
#        self.body = iSequential(
#                addZslot(),
#                passThrough(iLogits()),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCenter(), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCenter(reverse=True), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                FlatJoin()
#            )
#
#
#class RealNVPSubCycleMask(RealNVPBase):
#
#    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, 
#                 num_blocks=8, init_zeros=False, st_type='resnet', 
#                 use_batch_norm=True):
#        super(RealNVPSubCycleMask, self).__init__()
#
#        self.body = iSequential(
#                addZslot(),
#                passThrough(iLogits()),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskSubQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskSubQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskSubQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskSubQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                FlatJoin()
#            )
#
#
#class RealNVPNewMask(RealNVPBase):
#
#    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
#            st_type='resnet', use_batch_norm=True):
#        super(RealNVPNewMask, self).__init__()
#
#        self.body = iSequential(
#                addZslot(),
#                passThrough(iLogits()),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskVertical(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskVertical(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskVertical(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskVertical(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                FlatJoin()
#            )
#
#
#class RealNVPNewMask2(RealNVPBase):
#
#    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
#            st_type='resnet', use_batch_norm=True):
#        super(RealNVPNewMask2, self).__init__()
#
#        self.body = iSequential(
#                addZslot(),
#                passThrough(iLogits()),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskVertical(reverse_mask=False),   init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskVertical(reverse_mask=True),    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskVertical(reverse_mask=False),   init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskVertical(reverse_mask=True),    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                FlatJoin()
#            )
#
#
#class RealNVPMegaSubCycleMask(RealNVPBase):
#
#    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, 
#                 num_blocks=8, init_zeros=False, st_type='resnet', 
#                 use_batch_norm=True):
#        super(RealNVPMegaSubCycleMask, self).__init__()
#
#        self.body = iSequential(
#                addZslot(),
#                passThrough(iLogits()),
#                passThrough(CouplingLayer(in_channels, mid_channels//4, num_blocks, MaskSubQuadrant(input_quadrant=0, output_quadrant=1, factor=4), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels//4, num_blocks, MaskSubQuadrant(input_quadrant=1, output_quadrant=2, factor=4), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels//4, num_blocks, MaskSubQuadrant(input_quadrant=2, output_quadrant=3, factor=4), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels//4, num_blocks, MaskSubQuadrant(input_quadrant=3, output_quadrant=0, factor=4), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels//2, num_blocks, MaskSubQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels//2, num_blocks, MaskSubQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels//2, num_blocks, MaskSubQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels//2, num_blocks, MaskSubQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                FlatJoin()
#            )
#
#
#class RealNVPThreeCycleMask(RealNVPBase):
#
#    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
#            st_type='resnet', use_batch_norm=True):
#        super(RealNVPThreeCycleMask, self).__init__()
#
#        self.body = iSequential(
#                addZslot(),
#                passThrough(iLogits()),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                FlatJoin()
#            )
#
#
#class RealNVPSmoothedCycleMask(RealNVPBase):
#
#    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
#            st_type='resnet', use_batch_norm=True):
#        super(RealNVPSmoothedCycleMask, self).__init__()
#
#        self.body = iSequential(
#                addZslot(),
#                passThrough(iLogits()),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True))),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                FlatJoin()
#            )
#
#
#class RealNVPDeepCycleMask(RealNVPBase):
#
#    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
#            st_type='resnet', use_batch_norm=True):
#        super(RealNVPDeepCycleMask, self).__init__()
#
#        self.body = iSequential(
#                addZslot(),
#                passThrough(iLogits()),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                FlatJoin()
#            )
#
#
#class RealNVPMNIST(RealNVPBase):
#    def __init__(self, in_channels=1, mid_channels=64, num_blocks=4):
#        super(RealNVPMNIST, self).__init__()
#        
#        self.body = iSequential(
#                addZslot(), 
#                passThrough(iLogits()),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False))),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True))),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False))),
#                passThrough(SqueezeLayer(2)),
#                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=False))),
#                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=True))),
#                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=False))),
#                keepChannels(2*in_channels),                                                      
#                passThrough(CouplingLayer(2*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False))),
#                passThrough(CouplingLayer(2*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True))),
#                passThrough(CouplingLayer(2*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False))),
#                passThrough(SqueezeLayer(2)),
#                passThrough(CouplingLayer(8*in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=False))),
#                passThrough(CouplingLayer(8*in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=True))),
#                passThrough(CouplingLayer(8*in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=False))),
#                keepChannels(4*in_channels),
#                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False))),
#                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True))),
#                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False))),
#                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True))),
#                FlatJoin()
#            )
#
#
#class RealNVPOneCycleMask(RealNVPBase):
#
#    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
#            st_type='resnet', use_batch_norm=True):
#        super(RealNVPOneCycleMask, self).__init__()
#
#        self.body = iSequential(
#                addZslot(),
#                passThrough(iLogits()),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
#                FlatJoin()
#            )
#
#
#class RealNVPw1x1(RealNVP):
#    @staticmethod
#    def _threecouplinglayers(in_channels, mid_channels, num_blocks, mask_class):
#        layers = [
#                iConv1x1(in_channels),
#                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False)),
#                iConv1x1(in_channels),
#                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=True)),
#                iConv1x1(in_channels),
#                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False))
#        ]
#        return layers
#
#
#class RealNVPw1x1ActNorm(RealNVP):
#    @staticmethod
#    def _threecouplinglayers(in_channels, mid_channels, num_blocks, mask_class):
#        layers = [
#                ActNorm2d(in_channels),
#                iConv1x1(in_channels),
#                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False)),
#                ActNorm2d(in_channels),
#                iConv1x1(in_channels),
#                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=True)),
#                ActNorm2d(in_channels),
#                iConv1x1(in_channels),
#                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False))
#        ]
#        return layers
#
#
#class RealNVPwDS(RealNVP):
#
#    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
#        super().__init__()
#        
#        layers = [addZslot(), passThrough(iLogits())]
#
#        for scale in range(num_scales):
#            in_couplings = self._threecouplinglayers(in_channels, mid_channels, num_blocks, MaskCheckerboard)
#            layers.append(passThrough(*in_couplings))
#
#            if scale == num_scales - 1:
#                layers.append(passThrough(
#                    CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True))))
#            else:
#                layers.append(passThrough(iAvgPool2d()))
#                out_couplings = self._threecouplinglayers(4 * in_channels, 2 * mid_channels, num_blocks, MaskChannelwise)
#                layers.append(passThrough(*out_couplings))
#                layers.append(keepChannels(2 * in_channels))
#            
#            in_channels *= 2
#            mid_channels *= 2
#
#        layers.append(FlatJoin())
#        self.body = iSequential(*layers)
