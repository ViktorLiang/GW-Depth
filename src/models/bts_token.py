# Copyright (C) 2020 Guanglei Yang
#
# This file is a part of PGA
# add btsnet with attention .
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from unittest import skip
import torch
import torch.nn as nn
import math
from torch.nn.functional import unfold
from torch.nn import functional as F
import numpy as np

def bn_init_as_tf(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model
        m.eval()  # or zero mean and variance of one if the batch norm layer has no pretrained values
        m.affine = True
        m.requires_grad = True

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True,
                                                                   track_running_stats=True, eps=1.1e-5))

        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels,
                                                                              out_channels=out_channels * 2, bias=False,
                                                                              kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels * 2, momentum=0.01,
                                                                                   affine=True,
                                                                                   track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2,
                                                                              out_channels=out_channels, bias=False,
                                                                              kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation),
                                                                              dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)


class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                              padding=1)
        self.ratio = ratio

    def forward(self, x, new_size=None):
        if new_size is None:
            up_x = F.interpolate(x, scale_factor=self.ratio, mode='nearest')
        else:
            up_x = F.interpolate(x, size=new_size, mode='nearest')

        out = self.conv(up_x)
        out = self.elu(out)
        return out


class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()

        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module('final',
                                          torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
                                                                        kernel_size=1, stride=1, padding=0),
                                                              nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(
                                          nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2

    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)

        return net


class local_planar_guidance(nn.Module):
    def __init__(self, upratio):
        super(local_planar_guidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq, focal):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1 * u + n2 * v + n3)


class bts(nn.Module):
    def __init__(self, max_depth, token_channel, num_features=64):
        super(bts, self).__init__()
        # depth prediction
        self.max_depth = max_depth
        self.daspp_3 = atrous_conv(token_channel,                                          num_features // 2, 3, apply_bn_first=False)
        self.daspp_6 = atrous_conv(token_channel + num_features // 2,                      num_features // 4, 6)
        self.daspp_12 = atrous_conv(token_channel + num_features // 2 + num_features // 4, num_features // 4, 12)
        self.daspp_18 = atrous_conv(token_channel + num_features,                          num_features // 4, 18)
        self.daspp_24 = atrous_conv(token_channel + num_features + num_features // 4,      num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(
            nn.Conv2d(token_channel + num_features //2 + num_features, num_features, 3, 1, 1, bias=False),
            nn.ELU())
        self.upconv1 = upconv(num_features, num_features, ratio=2)
        self.bn1 = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        self.upconv2 = upconv(num_features, num_features)
        self.bn2 = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)

        # seg prediction
        self.saspp_3 = atrous_conv(token_channel,                                          num_features // 2, 3, apply_bn_first=False)
        self.saspp_6 = atrous_conv(token_channel + num_features // 2,                      num_features // 4, 6)
        self.saspp_12 = atrous_conv(token_channel + num_features // 2 + num_features // 4, num_features // 4, 12)
        self.saspp_18 = atrous_conv(token_channel + num_features,                          num_features // 4, 18)
        self.saspp_24 = atrous_conv(token_channel + num_features + num_features // 4,      num_features // 4, 24)
        self.saspp_conv = torch.nn.Sequential(
            nn.Conv2d(token_channel + num_features //2 + num_features, num_features, 3, 1, 1, bias=False),
            nn.ELU())
        self.upconv1_seg = upconv(num_features, num_features, ratio=2)
        self.bn1_seg = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        self.upconv2_seg = upconv(num_features, num_features)
        self.bn2_seg = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)

        self.get_depth = torch.nn.Sequential(nn.Conv2d(num_features, 1, 3, 1, 1, bias=False),
                                             nn.Sigmoid())
        self.get_seg = torch.nn.Sequential(nn.Conv2d(num_features, 2, 3, 1, 1, bias=False),
                                             nn.Sigmoid())

    def forward(self, depth_token, seg_token, input_size):
        layer_sizes = [(depth_token.shape[1], depth_token.shape[2])]
        layer_sizes.append(input_size)

        depth_features = torch.nn.ReLU()(depth_token)

        daspp_3 = self.daspp_3(depth_features)
        concat1 = torch.cat([depth_features, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat1)
        concat2 = torch.cat([concat1, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat2)
        concat3 = torch.cat([concat2, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat3)
        concat4 = torch.cat([concat3, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4)
        concat_daspp = torch.cat([depth_features, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat_daspp)
        upconv1 = self.bn1(self.upconv1(daspp_feat))  # H/2
        upconv2 = self.bn2(self.upconv2(upconv1, new_size=input_size))  # H/1
        depth_pred = self.get_depth(upconv2)

        seg_features = torch.nn.ReLU()(seg_token)
        saspp_3 = self.saspp_3(seg_features)
        concat_s1 = torch.cat([seg_features, saspp_3], dim=1)
        saspp_6 = self.saspp_6(concat_s1)
        concat_s2 = torch.cat([concat_s1, saspp_6], dim=1)
        saspp_12 = self.saspp_12(concat_s2)
        concat_s3 = torch.cat([concat_s2, saspp_12], dim=1)
        saspp_18 = self.saspp_18(concat_s3)
        concat_s4 = torch.cat([concat_s3, saspp_18], dim=1)
        saspp_24 = self.saspp_24(concat_s4)
        concat_saspp = torch.cat([seg_features, saspp_3, saspp_6, saspp_12, saspp_18, saspp_24], dim=1)
        saspp_feat = self.saspp_conv(concat_saspp)
        upconv_s1 = self.bn1_seg(self.upconv1_seg(saspp_feat))  # H/2
        upconv_s2 = self.bn2_seg(self.upconv2_seg(upconv_s1, new_size=input_size))  # H/1
        seg_pred = self.get_seg(upconv_s2)

        return None, None, None, None, depth_pred, seg_pred

def build_depth_decoder(args):
    return bts(args.max_depth, 64, num_features=64)
