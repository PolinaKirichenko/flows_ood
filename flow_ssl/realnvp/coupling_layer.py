import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import IntEnum
from flow_ssl.resnet_realnvp import ResNet, ResNetAE
from flow_ssl.highway import HighwayNetwork
from flow_ssl.convnet_coupling import ConvNetCoupling, AEOld, AE
from flow_ssl.convnet_coupling import get_glow_coupling_layer_convnet
from flow_ssl.realnvp.utils import checkerboard_mask
#from flow_ssl.invertible.downsample import squeeze, unsqueeze


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1
    TABULAR = 2
    HORIZONTAL = 3
    VERTICAL = 4
    Quadrant = 5
    SubQuadrant = 6
    Center = 7


class MaskChannelwise:
    def __init__(self, reverse_mask):
        self.type = MaskType.CHANNEL_WISE
        self.reverse_mask = reverse_mask

    def mask(self, x):
        if self.reverse_mask:
            x_id, x_change = x.chunk(2, dim=1)
        else:
            x_change, x_id = x.chunk(2, dim=1)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        if self.reverse_mask:
            return torch.cat((x_id, x_change), dim=1)
        else:
            return torch.cat((x_change, x_id), dim=1)
        
    def mask_st_output(self, s, t):
        return s, t


class MaskCheckerboard:
    def __init__(self, reverse_mask):
        self.type = MaskType.CHECKERBOARD
        self.reverse_mask = reverse_mask

    def mask(self, x):
        self.b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
        x_id = x * self.b
        x_change = x * (1 - self.b)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b + x_change * (1 - self.b)
    
    def mask_st_output(self, s, t):
        return s * (1 - self.b), t * (1 - self.b)


class MaskTabular:
    def __init__(self, reverse_mask):
        self.type = MaskType.TABULAR
        self.reverse_mask = reverse_mask

    def mask(self, x):
        dim = x.size(1)
        split = dim // 2
        self.b = torch.zeros((1, dim), dtype=torch.float).to(x.device)
        if self.reverse_mask:
            self.b[:, split:] = 1.
        else:
            self.b[:, :split] = 1.
        x_id = x * self.b
        x_change = x * (1 - self.b)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b + x_change * (1 - self.b)
    
    def mask_st_output(self, s, t):
        return s * (1 - self.b), t * (1 - self.b)


class MaskHorizontal:
    def __init__(self, reverse_mask):
        self.type = MaskType.HORIZONTAL
        self.reverse_mask = reverse_mask

    def mask(self, x):
        # x.shape = (bs, c, h, w)
        c, h, w = x.size(1), x.size(2), x.size(3)
        split = h // 2
        self.b = torch.zeros((1, c, h, w), dtype=torch.float).to(x.device)
        if self.reverse_mask:
            self.b[:, :, split:, :] = 1.
        else:
            self.b[:, :, :split, :] = 1.
        x_id = x * self.b
        x_change = x * (1 - self.b)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b + x_change * (1 - self.b)

    def mask_st_output(self, s, t):
        return s * (1 - self.b), t * (1 - self.b)


class MaskVertical:
    def __init__(self, reverse_mask):
        self.type = MaskType.VERTICAL
        self.reverse_mask = reverse_mask

    def mask(self, x):
        # x.shape = (bs, c, h, w)
        c, h, w = x.size(1), x.size(2), x.size(3)
        split = w // 2
        self.b = torch.zeros((1, c, h, w), dtype=torch.float).to(x.device)
        if self.reverse_mask:
            self.b[:, :, :, split:] = 1.
        else:
            self.b[:, :, :, :split] = 1.
        x_id = x * self.b
        x_change = x * (1 - self.b)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b + x_change * (1 - self.b)
    
    def mask_st_output(self, s, t):
        return s * (1 - self.b), t * (1 - self.b)


class MaskQuadrant:
    def __init__(self, input_quadrant, output_quadrant):
        self.type = MaskType.Quadrant
        self.input_quadrant = input_quadrant
        self.output_quadrant = output_quadrant

    @staticmethod
    def _get_quadrant_mask(quadrant, c, h, w):
        b = torch.zeros((1, c, h, w), dtype=torch.float)
        split_h = h // 2
        split_w = w // 2
        if quadrant == 0:
            b[:, :, :split_h, :split_w] = 1.
        elif quadrant == 1:
            b[:, :, :split_h, split_w:] = 1.
        elif quadrant == 2:
            b[:, :, split_h:, split_w:] = 1.
        elif quadrant == 3:
            b[:, :, split_h:, :split_w] = 1.
        else:
            raise ValueError("Incorrect mask quadrant")
        return b

    def mask(self, x):
        # x.shape = (bs, c, h, w)
        c, h, w = x.size(1), x.size(2), x.size(3)
        self.b_in = self._get_quadrant_mask(self.input_quadrant, c, h, w).to(x.device)
        self.b_out = self._get_quadrant_mask(self.output_quadrant, c, h, w).to(x.device)
        x_id = x * self.b_in
        x_change = x * (1 - self.b_in)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b_in + x_change * (1 - self.b_in)
    
    def mask_st_output(self, s, t):
        return s * self.b_out, t * self.b_out


#class MaskSubQuadrant:
#    def __init__(self, input_quadrant, output_quadrant):
#        self.type = MaskType.SubQuadrant
#        self.quad_mask = MaskQuadrant(input_quadrant, output_quadrant)
#
#    def mask(self, x):
#        # x.shape = (bs, c, h, w)
#        bs, c, h, w = x.shape
#        x_reshape = torch.zeros_like(x).reshape(bs*4, c, h//2, w//2)
#        h_split = h // 2
#        w_split = w // 2
#
#        x_reshape[:bs] = x[:, :, :h_split, :w_split]
#        x_reshape[bs:2*bs] = x[:, :, h_split:, :w_split]
#        x_reshape[2*bs:3*bs] = x[:, :, h_split:, w_split:]
#        x_reshape[3*bs:] = x[:, :, :h_split, w_split:]
#
#        x_id, x_change = self.quad_mask.mask(x_reshape)
#        return x_id, x_change
#
#    def unmask(self, x_id, x_change):
#        x_reshape = self.quad_mask.unmask(x_id, x_change)
#        bs, c, h, w = x_reshape.shape
#        new_bs = bs // 4
#        x = torch.zeros_like(x_reshape).reshape(new_bs, c, h*2, w*2)
#
#        x[:, :, :h, :w] = x_reshape[:new_bs]
#        x[:, :, h:, :w] = x_reshape[new_bs:2*new_bs] 
#        x[:, :, h:, w:] = x_reshape[2*new_bs:3*new_bs]
#        x[:, :, :h, w:] = x_reshape[3*new_bs:]
#
#        return x
#    
#    def mask_st_output(self, s, t):
#        return self.quad_mask.mask_st_output(s, t)
#
#    def reshape_logdet(self, logdet):
#        bs = logdet.shape[0] // 4
#        new_logdet = torch.zeros_like(logdet[:bs])
#        new_logdet = logdet[:bs] + logdet[bs:2*bs] + logdet[2*bs:3*bs] + logdet[3*bs:]
#        return new_logdet

class MaskSubQuadrant:
    def __init__(self, input_quadrant, output_quadrant, factor=2):
        self.type = MaskType.SubQuadrant
        self.quad_mask = MaskQuadrant(input_quadrant, output_quadrant)
        self.factor = factor

    def mask(self, x):
        # x.shape = (bs, c, h, w)
        bs, c, h, w = x.shape
        h_new = h // self.factor
        w_new = w // self.factor
        bs_new = bs * self.factor**2
        x_reshape = x.reshape((bs, c, self.factor, h_new, self.factor, w_new))
        x_reshape = x_reshape.permute(0, 2, 4, 1, 3, 5)
        x_reshape = x_reshape.reshape((bs_new, c, h_new, w_new))

        x_id, x_change = self.quad_mask.mask(x_reshape)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        x_reshape = self.quad_mask.unmask(x_id, x_change)
        bs_new, c, h_new, w_new = x_reshape.shape

        bs = bs_new // self.factor**2
        h = h_new * self.factor
        w = w_new * self.factor

        x_reshape = x_reshape.reshape((bs, self.factor, self.factor, c, h_new, w_new))
        x_reshape = x_reshape.permute(0, 3, 1, 4, 2, 5)
        x = x_reshape.reshape((bs, c, h, w))

        return x
    
    def mask_st_output(self, s, t):
        return self.quad_mask.mask_st_output(s, t)

    def reshape_logdet(self, logdet):
        bs_new = logdet.shape[0]
        bs = bs_new // self.factor**2
        logdet_reshape = logdet.reshape((bs, self.factor, self.factor))
        logdet = logdet_reshape.sum(dim=(1,2))
        return logdet


class MaskCenter:
    def __init__(self, margin=4, reverse=False):
        self.type = MaskType.Center
        self.reverse = True
        self.margin = margin

    def mask(self, x):
        # x.shape = (bs, c, h, w)
        bs, c, h, w = x.shape
        # Idea: can we speed everything up by not re-creating the mask every time?
        self.b = torch.zeros((1, c, h, w), dtype=torch.float).to(x.device)
        self.b[:, :, self.margin:-self.margin, self.margin:-self.margin] = 1.
        if not self.reverse:
            self.b = 1 - self.b
        x_id = x * self.b
        x_change = x * (1 - self.b)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b + x_change * (1 - self.b)

    def mask_st_output(self, s, t):
        return s * (1 - self.b), t * (1 - self.b)


class CouplingLayerBase(nn.Module):
    """Coupling layer base class in RealNVP.
    
    must define self.mask, self.st_net, self.rescale
    """

    def _get_st(self, x):
        #positional encoding
        #bs, c, h, w = x.shape
        #y_coords = torch.arange(h).float().cuda() / h
        #y_coords = y_coords[None, None, :, None].repeat((bs, 1, 1, w))
        #x_coords = torch.arange(w).float().cuda() / w
        #x_coords = x_coords[None, None, None, :].repeat((bs, 1, h, 1))
        #x = torch.cat([x, y_coords, x_coords], dim=1)

        x_id, x_change = self.mask.mask(x)
        st = self.st_net(x_id)
        #st = self.st_net(F.dropout(x_id, training=self.training, p=0.5))
        #st = self.st_net(F.dropout(x_id, training=True, p=0.9))
        s, t = st.chunk(2, dim=1)
        s = self.rescale(torch.tanh(s))

        #positional encoding
        #s = s[:, :-2]
        #t = t[:, :-2]
        #x_id = x_id[:, :-2]
        #x_change = x_change[:, :-2]
        return s, t, x_id, x_change

    def forward(self, x, sldj=None, reverse=True):
        s, t, x_id, x_change = self._get_st(x)
        s, t = self.mask.mask_st_output(s, t)
        #positional encoding
        #s = s[:, :-2]
        #t = t[:, :-2]

        exp_s = s.exp()
        if torch.isnan(exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = (x_change + t) * exp_s
        self._logdet = s.view(s.size(0), -1).sum(-1)
        if self.mask.type == MaskType.SubQuadrant:
            # DEBUG!!!!!!!
           self._logdet = self.mask.reshape_logdet(self._logdet) 
        x = self.mask.unmask(x_id, x_change)
        #positional encoding
        #x = x[:, :-2]
        return x

    def inverse(self, y):
        s, t, x_id, x_change = self._get_st(y)
        s, t = self.mask.mask_st_output(s, t)
        exp_s = s.exp()
        inv_exp_s = s.mul(-1).exp()
        if torch.isnan(inv_exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = x_change * inv_exp_s - t
        self._logdet = -s.view(s.size(0), -1).sum(-1)
        x = self.mask.unmask(x_id, x_change)

        #positional encoding
        #x = x[:, :-2]
        return x

    def logdet(self):
        return self._logdet


class CouplingLayer(CouplingLayerBase):
    """Coupling layer in RealNVP for image data.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask (MaskChannelWise or MaskChannelWise): mask.
    """

    def __init__(self, in_channels, mid_channels, num_blocks, mask, init_zeros=False, st_type='resnet',
            use_batch_norm=True, img_width=28, skip=True, latent_dim=100):
        super(CouplingLayer, self).__init__()

        self.mask = mask

        # Build scale and translate network
        if self.mask.type == MaskType.CHANNEL_WISE:
            in_channels //= 2

        if st_type == 'resnet':
            self.st_net = ResNet(in_channels, mid_channels, 2*in_channels,
                                 num_blocks=num_blocks, kernel_size=3, padding=1,
                                 double_after_norm=(self.mask.type == MaskType.CHECKERBOARD),
                                 init_zeros=init_zeros, use_batch_norm=use_batch_norm, skip=skip)
        if st_type == 'resnet_ae':
            self.st_net = ResNetAE(in_channels, mid_channels, 2*in_channels,
                                 num_blocks=num_blocks, kernel_size=3, padding=1,
                                 double_after_norm=(self.mask.type == MaskType.CHECKERBOARD),
                                 latent_dim=latent_dim, init_zeros=init_zeros,
                                 use_batch_norm=use_batch_norm, img_width=img_width)
        elif st_type == 'highway':
            self.st_net = HighwayNetwork(in_channels, mid_channels, 2*in_channels, num_blocks=num_blocks)
        elif st_type == 'convnet':
            self.st_net = ConvNetCoupling(in_channels, mid_channels, 2*in_channels,
                                 num_blocks=num_blocks, kernel_size=3, padding=1,
                                 init_zeros=init_zeros)
        elif st_type == 'glow_convnet':
            self.st_net = get_glow_coupling_layer_convnet(in_channels, mid_channels, 2*in_channels)
        elif st_type == 'autoencoder_old':
            self.st_net = AEOld(in_channels, latent_dim, 2*in_channels, init_zeros=init_zeros, img_width=img_width)
        elif st_type == 'autoencoder':
            self.st_net = AE(in_channels, latent_dim, 2*in_channels, init_zeros=init_zeros, img_width=img_width)

        # Learnable scale for s
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.

    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x


class CouplingLayerTabular(CouplingLayerBase):

    def __init__(self, in_dim, mid_dim, num_layers, mask, init_zeros=False, dropout=False):
        
        super(CouplingLayerTabular, self).__init__()
        self.mask = mask
        self.st_net = nn.Sequential(nn.Linear(in_dim, mid_dim),
                                    nn.ReLU(),
                                    nn.Dropout(.5) if dropout else nn.Sequential(),
                                    *self._inner_seq(num_layers, mid_dim),
                                    nn.Linear(mid_dim, in_dim*2))

        if init_zeros:
                # init w zeros to init a flow w identity mapping
                torch.nn.init.zeros_(self.st_net[-1].weight)
                torch.nn.init.zeros_(self.st_net[-1].bias)

        self.rescale = nn.utils.weight_norm(RescaleTabular(in_dim))

    @staticmethod
    def _inner_seq(num_layers, mid_dim):
        res = []
        for _ in range(num_layers):
            res.append(nn.Linear(mid_dim, mid_dim))
            res.append(nn.ReLU())
        return res


class RescaleTabular(nn.Module):
    def __init__(self, D):
        super(RescaleTabular, self).__init__()
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x):
        x = self.weight * x
        return x

