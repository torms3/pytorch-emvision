import torch
import torch.nn as nn

from . import utils
from . import layers


__all__ = ['dynamic_rsunet']


width = [16,32,64,128,256,512]
nonlinearity = 'ReLU'
params = {}
g_unroll = 1


def set_unroll(n):
    global unroll
    unroll = n
    assert unroll > 0


def set_nonlinearity(act, **act_params):
    global nonlinearity
    assert act in ['LeakyReLU','PReLU','ELU','ReLU']
    nonlinearity = act

    global params
    params = {}
    params.update(act_params)
    # Use in-place module if available.
    if nonlinearity in ['LeakyReLU','ReLU','ELU']:
        params['inplace'] = True


def dynamic_rsunet(width=width, unroll=1, act='ReLU', **act_params):
    set_unroll(unroll)
    set_nonlinearity(act, **act_params)
    return DynamicRSUNet(width)


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    padding = utils.pad_size(kernel_size, 'same')
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)


class BNAct(nn.Module):
    def __init__(self, in_channels):
        super(BNAct, self).__init__()
        global unroll
        self.unroll = unroll
        self.bn = nn.ModuleList([nn.BatchNorm3d(in_channels) for _ in range(self.unroll)])
        self.act = getattr(nn, nonlinearity)(**params)

    def forward(self, x, unroll=1):
        assert unroll > 0 and unroll <= self.unroll
        return self.act(self.bn[unroll-1](x))


class BNActConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BNActConv, self).__init__()
        self.norm_act = BNAct(in_channels)
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size))

     def forward(self, x, unroll=1):
         return self.conv(self.norm_act(x, unroll=unroll))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = BNActConv(channels, channels)
        self.conv2 = BNActConv(channels, channels)

    def forward(self, x, unroll=1):
        residual = x
        x = self.conv1(x, unroll=unroll)
        x = self.conv2(x, unroll=unroll)
        x += residual
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.add_module('pre',  BNActConv(in_channels, out_channels))
        self.add_module('res',  ResBlock(out_channels))
        self.add_module('post', BNActConv(out_channels, out_channels))

    def forward(self, x, unroll=1):
        x = self.pre(x, unroll=unroll)
        x = self.res(x, unroll=unroll)
        return self.post(x, unroll=unroll)


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=(1,2,2)):
        super(UpConvBlock, self).__init__()
        self.down = nn.MaxPool3d(down)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, unroll=1):
        return self.conv(self.down(x), unroll=unroll)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=(1,2,2)):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=up, mode='trilinear'),
            layers.BilinearUp(in_channels, in_channels, factor=up),
            conv(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x, skip):
        return self.up(x) + skip


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=(1,2,2)):
        super(UpConvBlock, self).__init__()
        self.up = UpBlock(in_channels, out_channels, up=up)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip, unroll=1):
        x = self.up(x, skip)
        return self.conv(x, unroll=unroll)


class DynamicRSUNet(nn.Module):
    def __init__(self, width):
        super(RSUNet, self).__init__()
        assert len(width) > 1
        depth = len(width) - 1

        self.iconv = ConvBlock(width[0], width[0])

        self.dconvs = nn.ModuleList()
        for d in range(depth):
            self.dconvs.append(DownConvBlock(width[d], width[d+1]))

        self.uconvs = nn.ModuleList()
        for d in reversed(range(depth)):
            self.uconvs.append(UpConvBlock(width[d+1], width[d]))

        self.final = BNAct(width[0])

        self.init_weights()

    def forward(self, x, unroll=1):
        x = self.iconv(x, unroll=unroll)

        skip = list()
        for dconv in self.dconvs:
            skip.append(x)
            x = dconv(x, unroll=unroll)

        for uconv in self.uconvs:
            x = uconv(x, skip.pop(), unroll=unroll)

        return self.final(x, unroll=unroll)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if nonlinearity == 'LeakyReLU':
                    nn.init.kaiming_normal_(
                        m.weight,
                        nonlinearity='leaky_relu',
                        a=params['negative_slope'] if 'negative_slope' in params else 0.01
                    )
                elif nonlinearity == 'PReLU':
                    nn.init.kaiming_normal_(
                        m.weight,
                        nonlinearity='leaky_relu',
                        a=params['init'] if 'init' in params else 0.25
                    )
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
