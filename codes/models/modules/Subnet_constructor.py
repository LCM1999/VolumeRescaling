import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import codes.models.modules.module_util as mutil


def default_conv(channel_in, channel_out, kernel_size):
    return nn.Conv3d(channel_in, channel_out, kernel_size, padding=(kernel_size//2))


class BasicBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size):
        super(BasicBlock, self).__init__()
        self.inConv = nn.Sequential(
            nn.Conv3d(channel_in, channel_out, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.inConv(x)


class ResBlock(nn.Module):
    def __init__(
            self, n_feats=1, kernel_size=3, bias=True,
            bn=False, act=(nn.ReLU(True)), res_scale=0.1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv3d(in_channels=n_feats,
                               out_channels=n_feats,
                               kernel_size=kernel_size,
                               padding=1,
                               bias=bias))
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, withConvReLU=True):
        super(DownSampleBlock, self).__init__()
        if withConvReLU:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=1,
                          kernel_size=3,
                          padding=1,
                          stride=2),
                nn.Conv3d(in_channels=1,
                          out_channels=1,
                          kernel_size=3,
                          padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Conv3d(in_channels=in_channels,
                                  out_channels=1,
                                  kernel_size=3,
                                  padding=1,
                                  stride=2)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.Conv3d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class PixelShuffle3d(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = x.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class SubVoxelUpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale=2,):
        super(SubVoxelUpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=8 * in_channels,
                      kernel_size=3,
                      padding=1),
            PixelShuffle3d(scale),
            nn.Conv3d(in_channels=in_channels,
                      out_channels=1,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        else:
            return None

    return constructor
