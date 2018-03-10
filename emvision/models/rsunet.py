import torch
import torch.nn as nn

from . import utils


__all__ = ['RSUNet']


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    padding = utils.pad_size(kernel_size, 'same')
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.norm1 = nn.BatchNorm3d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(channels, channels)

        self.norm2 = nn.BatchNorm3d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv(channels, channels)

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += residual
        return out


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BNReLUConv, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', conv(in_channels, out_channels,
                                     kernel_size=kernel_size))


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.add_module('pre', BNReLUConv(in_channels, out_channels))
        self.add_module('res', ResBlock(out_channels))
        self.add_module('post', BNReLUConv(out_channels, out_channels))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=(1,2,2)):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=up, mode='trilinear')),
            conv(in_channels, out_channels, kernel_size=1))
        )

    def forward(self, x, skip):
        return self.up(x) + skip


width = [16,32,64,128,256,512]


class RSUNet(nn.Module):
    def __init__(self, width=width):
        super(RSUNet, self).__init__()
        assert len(width) > 1
        depth = len(width) - 1

        # InputBlock.
        self.iconv = ConvBlock(width[0], width[0])

        # Contracting/downsampling pathway.
        self.dconvs = nn.ModuleList()
        for d in range(depth):
            self.dconvs.append(nn.Sequential(nn.MaxPool3d((1,2,2)),
                                             ConvBlock(width[d], width[d+1])))

        # Expanding/upsampling pathway.
        self.uconvs = nn.ModuleList()
        for d in reversed(range(depth)):
            self.uconvs.append(nn.Sequential(UpBlock(width[d+1], width[d]),
                                             ConvBlock(width[d], width[d])))

        # Initialize weights.
        self.init_weights()

    def forward(self, x):
        # InputBlock.
        x = self.iconv(x)

        # Contracting/downsampling pathway.
        skip = list()
        for dconv in self.dconvs:
            skip.append(x)
            x = dconv(x)

        # Expanding/upsampling pathway.
        for uconv in self.uconvs:
            x = uconv(x, skip.pop())

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
