import torch
import torch.nn.functional as F
from torch import nn


class ImageClassifier(nn.Module):
    def __init__(self, class_dim):
        self.super(ImageClassifier, self).__init__()
