import torch
import torch.nn as nn
from torch.nn import functional as F


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        # initialize with random orhogonal matrix whose log det is 0
        w_init = torch.qr(torch.randn(*w_shape))[0]
        self.weight = nn.Parameter(torch.Tensor(w_init))
        self.w_shape = w_shape

    def get_weight_logdet(self, bs, h, w, inverse):
        logdet = (torch.slogdet(self.weight)[1] * h * w).expand(bs)
        if inverse:
            weight = torch.inverse(self.weight)
        else:
            weight = self.weight

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), logdet

    def forward(self, input):
        bs, c, h, w = input.shape
        weight, logdet = self.get_weight_logdet(bs, h, w, inverse=False)
        self._logdet = logdet
        z = F.conv2d(input, weight)
        return z

    def inverse(self, input):
        bs, c, h, w = input.shape
        weight, logdet = self.get_weight_logdet(bs, h, w, inverse=True)
        self._logdet = -logdet
        z = F.conv2d(input, weight)
        return z

    def logdet(self):
        return self._logdet


class ActNorm2d(nn.Module):
    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = - torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3],
                              keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def forward(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))

        bs, c, h, w = input.shape
        if not self.inited:
            self.initialize_parameters(input)

        input = input + self.bias
        input = input * torch.exp(self.logs)
        self._logdet = torch.sum(self.logs).expand(bs) * h * w
        return input

    def inverse(self, input):
        bs, c, h, w = input.shape
        if not self.inited:
            raise ValueError("Doing inverse path, but ActNorm not inited")

        input = input * torch.exp(-self.logs)
        input = input - self.bias
        self._logdet = -torch.sum(self.logs).expand(bs) * h * w
        return input

    def logdet(self):
        return self._logdet

