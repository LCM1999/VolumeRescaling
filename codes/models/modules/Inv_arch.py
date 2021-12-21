import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from codes.models.modules.Subnet_constructor import default_conv, BasicBlock, ResBlock, DownSampleBlock, SubVoxelUpsampleBlock


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        # print('InvBlockExp,split_len1,split_len2', self.split_len1, self.split_len2)
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        # print('InvBlockExp,x1,x2', x1.shape, x2.shape)

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        #3D
        weight0 = torch.stack([self.haar_weights, self.haar_weights], axis=2)
        weight1 = torch.stack([self.haar_weights, -self.haar_weights], axis=2)
        self.haar_weights = torch.cat([weight0, weight1], axis=0)

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]
            self.last_jac = self.elements / 4 * np.log(1/16.)  #暂未改动

            out = F.conv3d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 8.0  #除以2还是8
            out = out.reshape([x.shape[0], self.channel_in, 8, x.shape[2] // 2, x.shape[3] // 2, x.shape[4] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 8, x.shape[2] // 2, x.shape[3] // 2, x.shape[4] // 2])
            # print('HaarDownsampling,out',out.shape)
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]
            self.last_jac = self.elements / 4 * np.log(16.)  #暂未改动

            out = x.reshape([x.shape[0], 8, self.channel_in, x.shape[2], x.shape[3], x.shape[4]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 8, x.shape[2], x.shape[3], x.shape[4]])
            return F.conv_transpose3d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac


class InvRescaleNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2):
        super(InvRescaleNet, self).__init__()

        operations = []

        current_channel = channel_in
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 8
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                # print('InvRescaleNet',rev,out.shape)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                # print('InvRescaleNet',rev, out.shape)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out


class InvertibleDownsamplingNet(nn.Module):
    def __init__(self, opt_net):
        super(InvertibleDownsamplingNet, self).__init__()

        # operations = []

        channel_in = opt_net.get('in_nc', 1)
        channel_out = opt_net.get('out_nc', 1)
        kernel_size = opt_net.get('k_size', 3)
        scale = opt_net.get('scale', 2)
        res_scale = opt_net.get('res_scale', 0.1)

        current_channel = channel_in

        n_feats = 32

        # for i in range(int(math.log(scale, 2))):
        encoder_head = [BasicBlock(channel_in=channel_in, channel_out=n_feats, kernel_size=kernel_size)]
        encoder_body = []
        for _ in range(opt_net['block_num']):
            encoder_body.append(
                ResBlock(n_feats=n_feats,
                         kernel_size=kernel_size,
                         res_scale=res_scale))
        encoder_body.append(default_conv(n_feats, n_feats, kernel_size))

        encoder_tail = [
            DownSampleBlock(in_channels=n_feats),
        ]

        self.encoder_head = nn.Sequential(*encoder_head)
        self.encoder_body = nn.Sequential(*encoder_body)
        self.encoder_tail = nn.Sequential(*encoder_tail)

        decoder_head = [BasicBlock(channel_in=channel_in, channel_out=n_feats, kernel_size=kernel_size)]
        decoder_body = []
        for _ in range(opt_net['block_num']):
            decoder_body.append(
                ResBlock(n_feats=n_feats,
                         kernel_size=kernel_size,
                         res_scale=res_scale))
        decoder_body.append(default_conv(n_feats, n_feats, kernel_size))

        decoder_tail = [SubVoxelUpsampleBlock(in_channels=n_feats, scale=scale)]

        self.decoder_head = nn.Sequential(*decoder_head)
        self.decoder_body = nn.Sequential(*decoder_body)
        self.decoder_tail = nn.Sequential(*decoder_tail)

    def forward(self, x):
        x = self.encoder_head(x)
        x = self.encoder_body(x)
        LR = self.encoder_tail(x)

        x = self.decoder_head(LR)
        x = self.decoder_body(x)
        HR = self.decoder_tail(x)

        return LR, HR

    def encode(self, x):
        x = self.encoder_head(x)
        x = self.encoder_body(x)
        LR = self.encoder_tail(x)

        return LR

    def decode(self, x):
        x = self.decoder_head(x)
        x = self.decoder_body(x)
        HR = self.decoder_tail(x)

        return HR

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

