import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AEOld(nn.Module):

    def __init__(self, in_channels, latent_dim, out_channels, init_zeros=False, img_width=28):
        super(AEOld, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        hidden_dims = [32, 32, 64]
        kernel_sizes = [4, 4, 4]
        strides = [1, 2, 2]
        paddings = [2, 1, 1]
        self.output_shapes = [(in_channels, img_width, img_width)]

        # Build Encoder
        cin = in_channels
        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(cin, out_channels=hidden_dims[i],
                              kernel_size=kernel_sizes[i], stride=strides[i],
                              padding=paddings[i]),
                    nn.LeakyReLU()
                )
            )
            _, hin, _ = self.output_shapes[-1]
            hout = int(np.floor((hin + 2*paddings[i] - kernel_sizes[i]) / strides[i] + 1))
            self.output_shapes.append((hidden_dims[i], hout, hout))
            cin = hidden_dims[i]

        last_shape = int(np.prod(self.output_shapes[-1]))
        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(last_shape, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, last_shape)
        output_paddings_reversed = [0, 1, 0]
        for i in range(1, len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[-i], hidden_dims[-i-1],
                        kernel_size=kernel_sizes[-i], stride=strides[-i],
                        padding=paddings[-i], output_padding=output_paddings_reversed[-i]),
                    nn.LeakyReLU())
            )

        modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[0], out_channels,
                        kernel_size=kernel_sizes[0], stride=strides[0],
                        padding=paddings[0], output_padding=output_paddings_reversed[0]),
        ))

        # initialize the last layer with zeros
        if init_zeros: 
            modules[-1].weight.data.zero_()
            modules[-1].bias.data.zero_()

        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        z = self.fc_z(result)
        return z

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, *self.output_shapes[-1])
        result = self.decoder(result)
        return result

    def forward(self, input):
        z = self.encode(input)
        reconstruction = self.decode(z)
        return reconstruction



class AE(nn.Module):

    def __init__(
            self, in_channels, latent_dim, out_channels, init_zeros=False, img_width=28):
        super(AE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        hidden_dims = [32, 64, 64, 128]
        # taking into account squeezes and reshapes in flow model
        scale = img_width // 14
        kernel_sizes = [4, 4, 4, 4]
        strides = [1, 1, scale, scale]
        paddings = [0, 0, 0, 0]
        self.encoder_output_shapes = [(in_channels, img_width, img_width)]

        # Build Encoder
        cin = in_channels
        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(cin, out_channels=hidden_dims[i],
                              kernel_size=kernel_sizes[i], stride=strides[i],
                              padding=paddings[i]),
                    nn.LeakyReLU()
                )
            )
            _, hin, _ = self.encoder_output_shapes[-1]
            hout = int(np.floor((hin + 2*paddings[i] - kernel_sizes[i]) / strides[i] + 1))
            self.encoder_output_shapes.append((hidden_dims[i], hout, hout))
            cin = hidden_dims[i]

        last_shape = int(np.prod(self.encoder_output_shapes[-1]))
        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(last_shape, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_output_shapes = [self.encoder_output_shapes[-1],]
        self.decoder_input = nn.Linear(latent_dim, last_shape)
        for i in range(1, len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[-i], hidden_dims[-i-1],
                        kernel_size=kernel_sizes[-i], stride=strides[-i],
                        padding=paddings[-i]),
                    nn.LeakyReLU())
            )
            _, hin, _ = self.decoder_output_shapes[-1]
            hout = (hin - 1) * strides[-i] - 2 * paddings[-i] + kernel_sizes[-i]
            self.decoder_output_shapes.append((hidden_dims[-i-1], hout, hout))

        # w/o LeakyReLU
        modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[0], out_channels,
                        kernel_size=kernel_sizes[0], stride=strides[0],
                        padding=paddings[0]),
        ))
        _, hin, _ = self.decoder_output_shapes[-1]
        hout = (hin - 1) * strides[0] - 2 * paddings[0] + kernel_sizes[0]
        self.decoder_output_shapes.append((in_channels, hout, hout))

        # initialize the last layer with zeros
        if init_zeros:
            modules[-1].weight.data.zero_()
            modules[-1].bias.data.zero_()

        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        z = self.fc_z(result)
        return z

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, *self.encoder_output_shapes[-1])
        result = self.decoder(result)
        return result

    def forward(self, input):
        z = self.encode(input)
        return  self.decode(z)

