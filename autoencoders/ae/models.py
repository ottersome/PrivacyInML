import sys
from pathlib import Path
from typing import List

pml_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(pml_path))
from logging import DEBUG

import torch
from pml.utils import setup_logger  # type:ignore
from torch import nn


# Imports
class ConvVAE(nn.Module):
    def __init__(self, image_dim: List[int], input_dim, latent_dim):
        """
        params
        ------
            image_dim: H, W
        """
        super(ConvVAE, self).__init__()
        # Convolution Stages
        strides = torch.LongTensor([2, 2])
        self.logger = setup_logger(
            __class__.__name__, DEBUG, log_path=Path(__file__).resolve().parent
        )
        # CHECK: I am reducing size appropriately
        self.og_size = image_dim
        self.last_img_size = (
            torch.floor(
                (
                    (torch.LongTensor(image_dim) + (2 * 1) - (3 - 1) - 1)
                    / torch.prod(strides)
                    + 1
                )
            )
            .to(torch.long)
            .tolist()
        )
        self.conv_seq = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=strides.tolist()[0], padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=strides.tolist()[1], padding=1),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(32 * self.last_img_size[0] * self.last_img_size[1], input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
        )
        self.for_mus = nn.Linear(64, latent_dim)
        self.for_vars = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.ReLU(),
            nn.Linear(128, 32 * self.last_img_size[0] * self.last_img_size[1]),
        )
        self.deconv1 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=3,
            stride=strides.tolist()[1],
            padding=1,
            output_padding=0,
        )
        self.deconv2 = nn.ConvTranspose2d(
            16,
            3,
            kernel_size=3,
            stride=strides.tolist()[0],
            padding=1,
            output_padding=1,
        )

    def forward(self, batch_imgs):
        # Convolve
        conv_out = self.conv_seq(batch_imgs).view(batch_imgs.shape[0], -1)

        # Encode
        encoded = self.encoder(conv_out)

        # Reparameterize
        mus, log_vars = (self.for_mus(encoded), self.for_vars(encoded))
        latent = self.reparameterize(mus, log_vars)
        # Deconvolve

        decoded = self.decoder(latent).view(
            batch_imgs.shape[0], 32, self.last_img_size[0], self.last_img_size[1]
        )
        deconv1 = self.deconv1(decoded)
        deconv2 = self.deconv2(deconv1)

        return torch.sigmoid(deconv2).view(
            batch_imgs.shape[0], 3, self.og_size[0], self.og_size[1]
        )

    def reparameterize(self, mu, log_vars):
        std = torch.exp(0.5 * log_vars)
        eps = torch.rand_like(std)
        return mu + eps * std
