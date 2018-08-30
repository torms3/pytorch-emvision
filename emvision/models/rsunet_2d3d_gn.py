import torch
import torch.nn as nn

from . import utils
from . import layers


__all__ = ['rsunet_2d3d_gn']


width = [16,32,64,128,256,512]


def rsunet_2d3d_gn(width=width, group=16, depth2d=0, kernel2d=(1,3,3)):
    return RSUNet(width, group, depth2d, kernel2d)


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    padding = utils.pad_size(kernel_size, 'same')
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)


class GNReLU(nn.Sequential):
    def __init__(self, in_channels):
        super(GNReLU, self).__init__()
        self.add_module('norm', nn.GroupNorm(in_channels//G, in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))


class GNReLUConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(GNReLUConv, self).__init__()
        self.add_module('norm_relu', GNReLU(in_channels))
        self.add_module('conv', conv(in_channels, out_channels,
                                     kernel_size=kernel_size))


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = GNReLUConv(channels, channels, kernel_size=kernel_size)
        self.conv2 = GNReLUConv(channels, channels, kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.add_module('pre',  GNReLUConv(in_channels, out_channels,
                                            kernel_size=kernel_size))
        self.add_module('res',  ResBlock(out_channels,
                                            kernel_size=kernel_size))
        self.add_module('post', GNReLUConv(out_channels, out_channels,
                                            kernel_size=kernel_size))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=(1,2,2)):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=up, mode='trilinear'),
            layers.BilinearUp(in_channels, in_channels),
            conv(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x, skip):
        return self.up(x) + skip


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=(1,2,2), kernel_size=3):
        super(UpConvBlock, self).__init__()
        self.up = UpBlock(in_channels, out_channels, up=up)
        self.conv = ConvBlock(out_channels, out_channels,
                              kernel_size=kernel_size)

    def forward(self, x, skip):
        x = self.up(x, skip)
        return self.conv(x)


G = 16


class RSUNet(nn.Module):
    def __init__(self, width, group, depth2d, kernel2d):
        super(RSUNet, self).__init__()
        assert len(width) > 1
        assert len(width) >= depth2d
        depth = len(width) - 1

        global G
        G = group

        kernel_size = kernel2d if depth2d > 0 else 3
        self.iconv = ConvBlock(width[0], width[0], kernel_size=kernel_size)

        self.dconvs = nn.ModuleList()
        for d in range(depth):
            kernel_size = kernel2d if depth2d > d+1 else 3
            self.dconvs.append(nn.Sequential(nn.MaxPool3d((1,2,2)),
                                             ConvBlock(width[d], width[d+1],
                                                    kernel_size=kernel_size)))

        self.uconvs = nn.ModuleList()
        for d in reversed(range(depth)):
            kernel_size = kernel2d if depth2d > d else 3
            self.uconvs.append(UpConvBlock(width[d+1], width[d],
                                                    kernel_size=kernel_size))

        self.final = GNReLU(width[0])

        self.init_weights()

    def forward(self, x):
        x = self.iconv(x)

        skip = list()
        for dconv in self.dconvs:
            skip.append(x)
            x = dconv(x)

        for uconv in self.uconvs:
            x = uconv(x, skip.pop())

        return self.final(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
