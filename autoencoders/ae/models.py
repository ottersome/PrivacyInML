import sys
from pathlib import Path
from typing import Dict, List

pml_path = Path(__file__).resolve().parent.parent.parent
from math import floor

sys.path.insert(0, str(pml_path))
from logging import DEBUG

import torch
from pml.utils import setup_logger  # type: ignore
from torch import nn


def calculate_output_size(layers, input_size):
    cheight, cwidth = input_size
    dimensions = []
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            # Extract parameters
            kernel_size = (
                layer.kernel_size
                if isinstance(layer.kernel_size, tuple)
                else (layer.kernel_size, layer.kernel_size)
            )
            stride = (
                layer.stride
                if isinstance(layer.stride, tuple)
                else (layer.stride, layer.stride)
            )
            padding = (
                layer.padding
                if isinstance(layer.padding, tuple)
                else (layer.padding, layer.padding)
            )
            dilation = (
                layer.dilation
                if isinstance(layer.dilation, tuple)
                else (layer.dilation, layer.dilation)
            )
            # Calculate output size
            new_height = floor(
                (cheight + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
                // stride[0]
                + 1
            )
            new_width = floor(
                (cwidth + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
                // stride[1]
                + 1
            )
            dimensions.append({"height": new_height, "width": new_width})
            cheight = new_height
            cwidth = new_width

        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
            # Extract parameters
            kernel_size = (
                layer.kernel_size
                if isinstance(layer.kernel_size, tuple)
                else (layer.kernel_size, layer.kernel_size)
            )
            stride = layer.stride if layer.stride is not None else kernel_size
            stride = stride if isinstance(stride, tuple) else (stride, stride)
            padding = (
                layer.padding
                if isinstance(layer.padding, tuple)
                else (layer.padding, layer.padding)
            )
            dilation = layer.dilation if layer.dilation is not None else 0
            dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
            # Calculate output size
            new_height = floor(
                (cheight + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)  # type: ignore
                // stride[0]
                + 1
            )
            new_width = floor(
                (cwidth + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)  # type: ignore
                // stride[1]
                + 1
            )
            dimensions.append({"height": new_height, "width": new_width})
            cheight = new_height
            cwidth = new_width

        # Add other layer types and calculations as needed
    return dimensions


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
        self.logger = setup_logger(
            __class__.__name__, DEBUG, log_path=Path(__file__).resolve().parent
        )
        # CHECK: I am reducing size appropriately
        self.og_size = image_dim

        self.conv_seq = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Add Pooling Layers
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.sizes = calculate_output_size(self.conv_seq, self.og_size)

        final_feat_size = 32 * self.sizes[-1]["height"] * self.sizes[-1]["width"]
        self.encoder = nn.Sequential(
            nn.Linear(final_feat_size, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
        )

        self.for_mus = nn.Linear(64, latent_dim)
        self.for_vars = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, final_feat_size),
        )
        self.deconv_seq = nn.Sequential(
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.ConvTranspose2d(
                16,
                3,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),
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
            batch_imgs.shape[0], 32, self.sizes[-1]["height"], self.sizes[-1]["width"]
        )
        deconved = self.deconv_seq(decoded)

        return torch.sigmoid(deconved).view(
            batch_imgs.shape[0], 3, self.og_size[0], self.og_size[1]
        )

    def reparameterize(self, mu, log_vars):
        std = torch.exp(0.5 * log_vars)
        eps = torch.rand_like(std)
        return mu + eps * std


class DoubleTroubleDownScale(nn.Module):
    def __init__(self, inchan: int, outchan: int):
        super(DoubleTroubleDownScale, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=inchan, out_channels=outchan, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=outchan, out_channels=outchan, kernel_size=3),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        seq = self.sequence(x)
        return seq, self.max_pool(seq)


class DoubleTroubleUpScale(nn.Module):
    def __init__(self, cin: int, cout: int):
        super(DoubleTroubleUpScale, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(
                in_channels=cin,
                out_channels=cout,
                kernel_size=3,
            ),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.Conv2d(in_channels=cout, out_channels=cout, kernel_size=3),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(
                in_channels=cout,
                out_channels=cout // 2,
                kernel_size=2,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x, skip_boi):
        # Crop and concatenate
        # CHECK: proper dimension order here "H,W,C"
        cropped_boi = center_crop_tensor(skip_boi, x)  # TODO: write cropping here
        contatenated_boi = torch.cat(
            (x, cropped_boi), 1
        )  # TODO: figure out which is the dimension for channels

        return self.sequence(contatenated_boi)


class UNet(nn.Module):
    def __init__(self, specs: Dict, og_size: Dict):
        super(UNet, self).__init__()

        # Down Slope
        amnt_down = len(specs["downslopes"])
        self.downslope = nn.Sequential()
        for i, spec in enumerate(specs["downslopes"]):
            inchan = (
                specs["downslopes"][i - 1]["channel_out"]
                if i != 0
                else og_size["channels"]
            )  # FIX: a bit too hard coded this channel
            outchan = spec["channel_out"]
            self.downslope.add_module(
                f"DTDown{i}", DoubleTroubleDownScale(inchan, outchan)
            )

        self.tunnel = nn.Sequential(  # FIX: A bit too hard coded here
            nn.Conv2d(
                specs["downslopes"][-1]["channel_out"],
                specs["tunnel_convs"][0]["channel_out"],
                kernel_size=3,
            ),
            nn.Conv2d(
                specs["tunnel_convs"][0]["channel_out"],
                specs["tunnel_convs"][1]["channel_out"],
                kernel_size=3,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(
                specs["tunnel_convs"][1]["channel_out"],
                specs["tunnel_convs"][2]["channel_out"],
                kernel_size=2,
            ),
        )

        # Up Slope
        self.upslope = nn.Sequential()
        for i, spec in enumerate(specs["upslopes"]):
            cin = (
                specs["tunnel_convs"][2]["channel_out"] * 2
                if i == 0
                else specs["upslopes"][i - 1]["channel_out"]
            )
            cout = spec["channel_out"]
            self.upslope.add_module(f"DTUp{i}", DoubleTroubleUpScale(cin, cout))

        self.fit_to_og_size = nn.Sequential(
            nn.Upsample((og_size["height"], og_size["width"]), mode="bilinear"),
            nn.Conv2d(
                in_channels=specs["upslopes"][-1]["channel_out"] // 2,
                out_channels=og_size["channels"],
                padding=1,
                kernel_size=3,
            ),
        )

    def forward(self, x):
        skips = []
        cur_val = x
        # Down
        for down in self.downslope:
            skip, downsamp = down(cur_val)
            skips.append(skip)
            cur_val = downsamp

        # Tunnel
        tunnel_out = self.tunnel(cur_val)

        # Up
        cur_val = tunnel_out
        for up in self.upslope:
            y = up(cur_val, skips.pop())
            cur_val = y

        # TODO: final 1x1 conv
        final_logits = self.fit_to_og_size(cur_val)

        return final_logits


class SimpleAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleAutoEncoder, self).__init__()
        self.autoencoder = nn.Sequential(
            # Encoder
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Decoder
            nn.Conv2d(12, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        self.protocombiner = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=1), nn.Sigmoid()
        )

    def forward(self, imgs):
        # x = torch.cat([imgs, same_proto], dim=1)
        x = imgs
        y = self.autoencoder(x)

        # rec_same = torch.cat([x, same_proto], dim=1)
        # rec_oppo = torch.cat([x, oppo_proto], dim=1)

        return self.protocombiner(
            y
        )  # self.protocombiner(rec_same), self.protocombiner(rec_oppo)


def center_crop_tensor(input_tensor, target_size):
    _, _, target_height, target_width = target_size.shape
    # Calculate the starting and ending indices for the crop
    height_start = (input_tensor.size(2) - target_height) // 2
    width_start = (input_tensor.size(3) - target_width) // 2
    height_end = height_start + target_height
    width_end = width_start + target_width
    # Crop the tensor
    # CHECK: Height and Width are in the correct place
    cropped_tensor = input_tensor[:, :, height_start:height_end, width_start:width_end]
    return cropped_tensor
