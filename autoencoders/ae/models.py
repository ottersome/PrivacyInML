from typing import List

import torch
from torch import nn


# Imports
class ConvVAE(nn.Module):
    def __init__(self, image_dim: List[int], input_dim, latent_dim):
        super(ConvVAE, self).__init__()
        # Convolution Stages
        strides = torch.LongTensor([2, 2])
        # CHECK: I am reducing size appropriately
        last_img_size = (torch.LongTensor(image_dim) // torch.prod(strides)).tolist()
        self.conv_seq = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=strides.tolist()[0], padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=strides.tolist()[1], padding=1),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 32, input_dim),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.for_mus = nn.Linear(64, latent_dim)
        self.for_vars = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.ReLU(),
            nn.Linear(128, 32 * 32 * 32),
        )
        self.deconv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            16, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, batch_imgs):
        conv_out = self.conv_seq(batch_imgs)
        encoded = self.encoder(conv_out)
        mus, log_vars = (self.for_mus(encoded), self.for_vars(encoded))
        latent = self.reparameterize(mus, log_vars)
        # Deconvolve
        deconv1 = self.deconv1(latent)
        deconv2 = self.deconv2(deconv1)

        return torch.sigmoid(deconv2)

    def reparameterize(self, mu, log_vars):
        std = torch.exp(0.5 * log_vars)
        eps = torch.rand_like(std)
        return mu + eps * std
