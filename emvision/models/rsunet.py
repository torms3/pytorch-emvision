import torch
import torch.nn as nn

from . import utils


__all__ = ['RSUNet']


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    padding = utils.pad_size(kernel_size, 'same')
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BNReLUConv, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', conv(in_channels, out_channels,
                                     kernel_size=kernel_size))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = BNReLUConv(channels, channels)
        self.conv2 = BNReLUConv(channels, channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.add_module('pre',  BNReLUConv(in_channels, out_channels))
        self.add_module('res',  ResBlock(out_channels))
        self.add_module('post', BNReLUConv(out_channels, out_channels))


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
        return F.conv_transpose3d(x, Variable(self.weight),
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


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=(1,2,2)):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=up, mode='trilinear'),
            BilinearUp(in_channels, out_channels),
            conv(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x, skip):
        return self.up(x) + skip


width = [16,32,64,128,256,512]


class RSUNet(nn.Module):
    def __init__(self, width=width):
        super(RSUNet, self).__init__()
        assert len(width) > 1
        depth = len(width) - 1

        self.in_channels = width[0]

        self.iconv = ConvBlock(width[0], width[0])

        self.dconvs = nn.ModuleList()
        for d in range(depth):
            self.dconvs.append(nn.Sequential(nn.MaxPool3d((1,2,2)),
                                             ConvBlock(width[d], width[d+1])))
        self.uconvs = nn.ModuleList()
        for d in reversed(range(depth)):
            self.uconvs.append(nn.Sequential(UpBlock(width[d+1], width[d]),
                                             ConvBlock(width[d], width[d])))

        self.out_channels = width[0]

        self.init_weights()

    def forward(self, x):
        x = self.iconv(x)

        skip = list()
        for dconv in self.dconvs:
            skip.append(x)
            x = dconv(x)

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
