"""
VGG-Style Model
Will follow architecture layed out by Melis
"""
from typing import List

import torch
from torch import nn

"""
Two convolution blocks consiting of cl and mp
three cb consits of two cl and mp
two fl
"""
# TODO: Might be too man pooling layers. For 200x200 images I might end up with 3x3
configuration = ["cl", "mp", "cl", "cl", "mp", "cl", "cl", "mp"]


class VGGModel(nn.Module):
    first_linears_size = 4096
    conv_width = 64
    conv_width_inc_factor = 2  # Should only increase if followed by pooling layer

    def __init__(self, conv_layers: int = 10):
        """
        Args ----
            conv_layers (int): Amount of Convolution Layers as Specified by Simonyan
            num_classes (int): Number of classifications for final layer.
        """
        super().__init__()
        # CHECK: I might need to have CxHxW instead of HxWxC
        self.final_pool_size = 3
        self.channels = self.conv_width
        # Initialize values
        self.init_stack = torch.nn.Sequential(*self._get_sequence_of_cnns(conv_layers))
        self.bridge = torch.nn.Sequential(
            torch.nn.AdaptiveMaxPool2d((3, 3)),  # CHECK: might not be enough
            torch.nn.Flatten(),
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(
                self.channels * self.final_pool_size**2, self.first_linears_size
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(self.first_linears_size, self.first_linears_size),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.init_stack(x)
        x = self.bridge(x)
        x = self.fc_layers(x)
        return x

    def _get_sequence_of_cnns(self, conv_layers: int) -> List[torch.nn.Module]:
        layers = []
        prev_channels = 3
        for conf in configuration:
            if conf == "cl":
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=prev_channels,
                        out_channels=self.conv_width,
                        kernel_size=3,
                        padding=1,
                    )
                )
                prev_channels = self.conv_width
                layers.append(torch.nn.ReLU())
            elif conf == "mp":
                layers.append(
                    torch.nn.MaxPool2d(
                        kernel_size=2,
                        # stride=2,
                    )
                )
        return layers


# Create a Wrapper that adds a Regression Output
class VGGRegressionModel(VGGModel):
    def __init__(self, conv_layers: int = 10):
        super().__init__(conv_layers)
        self.regression = torch.nn.Linear(self.first_linears_size, 1)

    def forward(self, x):
        x = self.init_stack(x)
        x = self.bridge(x)
        x = self.fc_layers(x)
        x = self.regression(x)
        return x


# Now one for classification
class VGGClassificationModel(VGGModel):
    def __init__(self, conv_layers: int = 10):
        super().__init__(conv_layers)
        self.classification = torch.nn.Linear(self.first_linears_size, num_classes)

    def forward(self, x):
        x = self.init_stack(x)
        x = self.bridge(x)
        x = self.fc_layers(x)
        x = self.classification(x)
        return x
