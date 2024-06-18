# Variational AutoEncoder using CNN
# https://github.com/sksq96/pytorch-vae

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)
"""
    def __init__(self, image_channels=3, , 1024], z_dim=32, batch_norm=True):
        super(VAE, self).__init__()
        self.batch_norm = batch_norm

        # Build Encoder
        self.encoder = nn.Sequential()
        in_channels = image_channels
        for h_dim in hidden_dims[:-1]:
            self.encoder.append(nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1))
            if self.batch_norm:
                self.encoder.append(nn.BatchNorm2d(h_dim))
            #self.encoder.append(nn.ReLU())
            self.encoder.append(nn.LeakyReLU())
            in_channels = h_dim
        self.encoder.append(Flatten())

        hdim = hidden_dims[-1]
        self.fc1 = nn.Linear(hdim, z_dim)
        self.fc2 = nn.Linear(hdim, z_dim)
        self.fc3 = nn.Linear(z_dim, hdim)
        hidden_dims = hidden_dims[:-2] + [hdim]
"""


def get_convtranspose2d_finalsize(kernels: list, paddings: list) -> int:
    if len(kernels) == 1:
        return kernels[0]
    return kernels[-1] + (paddings[-1]*(get_convtranspose2d_finalsize(kernels[:-1], paddings[:-1])-1))



class VAE(nn.Module):
    def __init__(
            self,
            image_channels=3,
            hidden_dims = [32, 64, 128, 256],
            h_dim=1024,
            z_dim=32,
            batch_norm=True,
            kernel_sizes_encoder = [4,4,4,4],
            strides_encoder = [2,2,2,2],
            kernel_sizes_decoder = [5,5,6,6],
            strides_decoder = [2,2,2,2],
        ):
        super(VAE, self).__init__()
        self.batch_norm = batch_norm

        # Build encoder
        self.encoder = nn.Sequential()
        in_channels = image_channels
        for hdim, ks, stride in zip(hidden_dims, kernel_sizes_encoder, strides_encoder):
            self.encoder.append(nn.Conv2d(in_channels, hdim, kernel_size=ks, stride=stride))
            if self.batch_norm:
                self.encoder.append(nn.BatchNorm2d(hdim))
            #self.encoder.append(nn.ReLU())
            self.encoder.append(nn.LeakyReLU())
            in_channels = hdim
        self.encoder.append(Flatten())

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        # Build Decoder
        hidden_dims[-1] = h_dim
        self.decoder = nn.Sequential()
        self.decoder.append(UnFlatten())
        for hdim, hdimpre, ks, stride in zip(hidden_dims[::-1], hidden_dims[-2::-1], kernel_sizes_decoder[:-1], strides_decoder[:-1]):
            self.decoder.append(nn.ConvTranspose2d(hdim, hdimpre, kernel_size=ks, stride=stride))
            if self.batch_norm:
                self.decoder.append(nn.BatchNorm2d(hdimpre))
            #self.decoder.append(nn.ReLU())
            self.decoder.append(nn.LeakyReLU())
            in_channels = hdimpre
        self.decoder.append(nn.ConvTranspose2d(in_channels, image_channels, kernel_size=kernel_sizes_decoder[-1], stride=strides_decoder[-1]))
        self.decoder.append(nn.Sigmoid())


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


def vae_loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    #BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD
