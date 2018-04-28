import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class BilinearUp(nn.Module):
    """Caffe style bilinear upsampling.

    Currently everything's hardcoded and only supports upsampling factor of 2.
    """
    def __init__(self, in_channels, out_channels):
        super(BilinearUp, self).__init__()
        assert in_channels==out_channels
        self.groups = in_channels
        self.init_weights()

    def forward(self, x):
        return F.conv_transpose3d(x, self.weight,
            stride=(1,2,2), padding=(0,1,1), groups=self.groups
        )

    def init_weights(self):
        weight = torch.Tensor(self.groups, 1, 1, 4, 4)
        width = weight.size(-1)
        hight = weight.size(-2)
        assert width==hight
        f = float(math.ceil(width / 2.0))
        c = float(width - 1) / (2.0 * f)
        for w in range(width):
            for h in range(hight):
                weight[...,h,w] = (1 - abs(w/f - c)) * (1 - abs(h/f - c))
        self.register_buffer('weight', weight)
