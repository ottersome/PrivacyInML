from typing import List, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return x + self.main(x)


class StarGenerator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(
            nn.Conv2d(
                3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False
            )
        )
        layers.append(
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        )
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True)
            )
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(
                nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True)
            )
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        )
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        T = self.sequence(x)
        return T + x  # Residual Function


class UpConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, output_padding=1),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(),
        ]
        self.sequence = nn.Sequential(*self.layers)

    def forward(self, x):
        # seq = self.sequence(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Generator(nn.Module):
    def __init__(self, amnt_attrs: int):
        super(Generator, self).__init__()
        self.amnt_attrs = amnt_attrs
        self.layers: List[nn.Module] = []
        self.layers += [
            nn.Conv2d(3 + amnt_attrs, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(),
        ]

        self.layers += [ResBlock() for i in range(6)]
        self.layers.append(UpConv())
        # Final Convolution and Tanh for image
        self.layers.append(
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )  # CHECK:Strides and the like
        self.layers_seq = nn.Sequential(*self.layers)

    def forward(self, imgs: torch.Tensor, labels: torch.Tensor):
        """
        Arguments
        ~~~~~~~~~
            imgs: (batch_size x channel x height x width)
            labels: one hot encoded (batch_size x amnt_attrs)
        """
        channeled_labels = labels.view(labels.size(0), labels.size(1), 1, 1).repeat(
            1, 1, imgs.size(2), imgs.size(3)
        )
        ccated_input = torch.cat((imgs, channeled_labels), dim=1)

        final_rep = self.layers_seq(ccated_input)
        # For debugging
        # x = ccated_input
        # for layer in self.layers:
        #     if isinstance(layer, UpConv):
        #         print("UpConv")
        #     x = layer(x)
        normed_out = F.tanh(final_rep)
        return normed_out


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(
            curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(
            h
        )  # Designed to output a sacalar, which acts as logit of attribute
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class VGGFace(nn.Module):
    def __init__(self):
        super(VGGFace, self).__init__()

        self.features = nn.ModuleDict(
            OrderedDict(
                {
                    # === Block 1 ===
                    "conv_1_1": nn.Conv2d(
                        in_channels=3, out_channels=64, kernel_size=3, padding=1
                    ),
                    "relu_1_1": nn.ReLU(inplace=True),
                    "conv_1_2": nn.Conv2d(
                        in_channels=64, out_channels=64, kernel_size=3, padding=1
                    ),
                    "relu_1_2": nn.ReLU(inplace=True),
                    "maxp_1_2": nn.MaxPool2d(kernel_size=2, stride=2),
                    # === Block 2 ===
                    "conv_2_1": nn.Conv2d(
                        in_channels=64, out_channels=128, kernel_size=3, padding=1
                    ),
                    "relu_2_1": nn.ReLU(inplace=True),
                    "conv_2_2": nn.Conv2d(
                        in_channels=128, out_channels=128, kernel_size=3, padding=1
                    ),
                    "relu_2_2": nn.ReLU(inplace=True),
                    "maxp_2_2": nn.MaxPool2d(kernel_size=2, stride=2),
                    # === Block 3 ===
                    "conv_3_1": nn.Conv2d(
                        in_channels=128, out_channels=256, kernel_size=3, padding=1
                    ),
                    "relu_3_1": nn.ReLU(inplace=True),
                    "conv_3_2": nn.Conv2d(
                        in_channels=256, out_channels=256, kernel_size=3, padding=1
                    ),
                    "relu_3_2": nn.ReLU(inplace=True),
                    "conv_3_3": nn.Conv2d(
                        in_channels=256, out_channels=256, kernel_size=3, padding=1
                    ),
                    "relu_3_3": nn.ReLU(inplace=True),
                    "maxp_3_3": nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                    # === Block 4 ===
                    "conv_4_1": nn.Conv2d(
                        in_channels=256, out_channels=512, kernel_size=3, padding=1
                    ),
                    "relu_4_1": nn.ReLU(inplace=True),
                    "conv_4_2": nn.Conv2d(
                        in_channels=512, out_channels=512, kernel_size=3, padding=1
                    ),
                    "relu_4_2": nn.ReLU(inplace=True),
                    "conv_4_3": nn.Conv2d(
                        in_channels=512, out_channels=512, kernel_size=3, padding=1
                    ),
                    "relu_4_3": nn.ReLU(inplace=True),
                    "maxp_4_3": nn.MaxPool2d(kernel_size=2, stride=2),
                    # === Block 5 ===
                    "conv_5_1": nn.Conv2d(
                        in_channels=512, out_channels=512, kernel_size=3, padding=1
                    ),
                    "relu_5_1": nn.ReLU(inplace=True),
                    "conv_5_2": nn.Conv2d(
                        in_channels=512, out_channels=512, kernel_size=3, padding=1
                    ),
                    "relu_5_2": nn.ReLU(inplace=True),
                    "conv_5_3": nn.Conv2d(
                        in_channels=512, out_channels=512, kernel_size=3, padding=1
                    ),
                    "relu_5_3": nn.ReLU(inplace=True),
                    "maxp_5_3": nn.MaxPool2d(kernel_size=2, stride=2),
                }
            )
        )

        self.fc = nn.ModuleDict(
            OrderedDict(
                {
                    "fc6": nn.Linear(in_features=512 * 7 * 7, out_features=4096),
                    "fc6-relu": nn.ReLU(inplace=True),
                    "fc6-dropout": nn.Dropout(p=0.5),
                    "fc7": nn.Linear(in_features=4096, out_features=4096),
                    "fc7-relu": nn.ReLU(inplace=True),
                    "fc7-dropout": nn.Dropout(p=0.5),
                    "fc8": nn.Linear(in_features=4096, out_features=2622),
                }
            )
        )

    def forward(self, x):
        # Forward through feature layers
        for k, layer in self.features.items():
            x = layer(x)

        # Flatten convolution outputs
        x = x.view(x.size(0), -1)

        # Forward through FC layers
        for k, layer in self.fc.items():
            x = layer(x)

        return x
