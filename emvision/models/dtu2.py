import torch
import torch.nn as nn

from functools import reduce

from . import utils
from . import layers


__all__ = ['DTU2']


DEFAULT_WIDTH = [12,72,432,2592] #feature maps
DEFAULT_KSZS = [(1,3,3),(1,3,3),(3,3,3),(3,3,3)] #kernel sizes
DEFAULT_FACTORS = [(1,3,3),(1,3,3),(3,3,3)] #downsampling/upsampling


class DTU2(nn.Module):
    def __init__(self, width=DEFAULT_WIDTH,
                 kszs=DEFAULT_KSZS,
                 factors=DEFAULT_FACTORS):
        super(DTU2, self).__init__()
        assert len(width) > 1
        assert len(width) == len(kszs) == len(factors) + 1, "mismatched args"
        self.depth = len(width)
        self.factors = factors

        self.iconv = ReLUConv(width[0], width[0], kszs[0])

        #d below corresponds to the mip at which a module performs convolutions
        self.dconvs = nn.ModuleList()
        for d in range(1,self.depth):
            kszs_d = [kszs[d]] * 2
            self.dconvs.append(DownModule(factors[d-1], width[d-1],width[d], kszs_d))

        #proceeding from last to first in order to preserve meaning of depth
        # (this avoids making computing crops a conceptual nightmare)
        self.uconvs = nn.ModuleList()
        self.uconvs.append(UpModule(factors[0], width[1],width[0], [kszs[0]]))
        for d in range(1,self.depth-1):
            kszs_d = [kszs[d]] * 2
            self.uconvs.append(UpModule(factors[d], width[d+1],width[d], kszs_d))

        self.crops = self.compute_crops()
        self.init_weights()

    def forward(self, x):
        x = self.iconv(x)

        skip = list()
        for dconv in self.dconvs:
            skip.append(x)
            x = dconv(x)

        for d in reversed(range(len(self.uconvs))):
            crop = self.crops[d]
            x = self.uconvs[d](x, skip.pop(), crop)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def compute_crops(self):
        crops = [(0,0,0)] * (self.depth-1)
        # crop for bridge
        crops[-1] = utils.mul3(self.dconvs[-1].crop, self.factors[-1])

        for d in reversed(range(1,self.depth-1)):
            prev_crop = crops[d]
            new_dcrop = self.dconvs[d-1].crop #offset by 1 for iconv
            new_ucrop = self.uconvs[d].crop
            new_crop = utils.sum3(new_ucrop, new_dcrop)

            crops[d-1] = utils.mul3(utils.sum3(prev_crop, new_crop),
                                    self.factors[d-1])

        return crops


class DownModule(nn.Module):
    def __init__(self, down_factor, in_channels, out_channels, kernel_sizes):
        super(DownModule, self).__init__()
        self.downsample = nn.MaxPool3d(down_factor)
        self.convmod = ConvModule(in_channels,out_channels, kernel_sizes)
        self.crop = self.convmod.crop

    def forward(self, x):
        x = self.downsample(x)
        return self.convmod(x)


class UpModule(nn.Module):
    def __init__(self, up_factor, in_channels, out_channels, kernel_sizes):
        super(UpModule, self).__init__()
        self.upsample = convt(in_channels, out_channels, up_factor)
        self.convmod = ConvModule(out_channels, out_channels, kernel_sizes)
        self.crop = self.convmod.crop

    def forward(self, x, skip, crop):
        x = self.upsample(x)
        x = utils.residual_sum(x, skip, crop, True)
        return self.convmod(x)


class ConvModule(nn.Sequential):
    def __init__(self, in_channels, out_channels, kszs, mode="valid"):
        super(ConvModule, self).__init__()
        self.add_module("0", ReLUConv(in_channels, out_channels, kszs[0]))
        for i in range(1,len(kszs)):
            self.add_module(str(i), ReLUConv(out_channels, out_channels, kszs[i]))
        self.crop = self.compute_crop(kszs, mode)

    def compute_crop(self, kszs, mode="valid"):
        if mode == "same":
            return (0,0,0)
        elif mode == "valid":
            pad_sizes = [utils.pad_size(ksz,"same") for ksz in kszs]
            return reduce(utils.sum3, pad_sizes)


class ReLUConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ReLUConv, self).__init__()
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", conv(in_channels, out_channels, kernel_size))



def conv(in_channels, out_channels, kernel_size=3, bias=True, mode="valid"):
    padding = utils.pad_size(kernel_size, 'valid')
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                     padding=padding, bias=bias)


def convt(in_channels, out_channels, kernel_size, stride=None, bias=True):
    stride = kernel_size if stride is None else stride
    return torch.nn.ConvTranspose3d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    bias=True)
