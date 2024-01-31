import torch
from torch import nn


class DescriptorLoss(nn.Module):
    """
    This task will evaluate Reconstruction
    """

    def __init__(self, description_generator: nn.Module):
        super().__init__()
        self.dg = description_generator
        self.criterion = nn.MSELoss()

    def forward(self, img, trans_img):
        img_desc = self.dg(img)
        trans_img_desc = self.dg(trans_img)

        loss = nn.MSELoss(img_desc, trans_img_desc)
        return loss
