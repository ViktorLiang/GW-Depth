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

from models.center_transformer import Mlp

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
    def __init__(self, max_depth, feat_out_channels, num_features=256):
        super(bts, self).__init__()
        self.max_depth = max_depth

        self.upconv5 = upconv(feat_out_channels[-1], num_features)
        self.bn5 = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)

        self.conv5 = torch.nn.Sequential(
            nn.Conv2d(num_features + feat_out_channels[-2], num_features, 3, 1, 1, bias=False),
            nn.ELU())
        self.upconv4 = upconv(num_features, num_features // 2)
        self.bn4 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(num_features // 2 + feat_out_channels[-3], num_features // 2, 3, 1, 1, bias=False),
            nn.ELU())
        self.bn4_2 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)

        self.daspp_3 = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6 = atrous_conv(num_features // 2 + num_features // 4 + feat_out_channels[1], num_features // 4, 6)
        self.daspp_12 = atrous_conv(num_features + feat_out_channels[1], num_features // 4, 12)
        self.daspp_18 = atrous_conv(num_features + num_features // 4 + feat_out_channels[1], num_features // 4, 18)
        self.daspp_24 = atrous_conv(num_features + num_features // 2 + feat_out_channels[1], num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(
            nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())

        self.upconv3 = upconv(num_features // 4, num_features // 4)
        self.bn3 = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(num_features // 4 + feat_out_channels[0], num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())
        
        # token fuse
        self.depth_token_fuse = Mlp(in_features=num_features // 2, out_features=num_features // 4)
        self.seg_token_fuse = Mlp(in_features=num_features // 2, out_features=num_features // 4)
        
        # depth pred
        self.upconv2_depth = upconv(num_features // 4, num_features // 8)
        self.bn2_depth = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2_depth = torch.nn.Sequential(
            nn.Conv2d(num_features // 8, num_features // 8, 3, 1, 1, bias=False),
            nn.ELU())
        self.upconv1_depth = upconv(num_features // 8, num_features // 16)
        self.reduc1x1_depth = reduction_1x1(num_features // 16, num_features // 32, self.max_depth, is_final=True)
        self.conv1_depth = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 1, num_features // 16, 3, 1, 1, bias=False),
                                         nn.ELU())
        self.get_depth = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
                                             nn.Sigmoid())
        
        # seg pred
        self.upconv2_seg = upconv(num_features // 4, num_features // 8)
        self.bn2_seg = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2_seg = torch.nn.Sequential(
            nn.Conv2d(num_features // 8, num_features // 8, 3, 1, 1, bias=False),
            nn.ELU())
        self.upconv1_seg = upconv(num_features // 8, num_features // 16)
        self.reduc1x1_seg = reduction_1x1(num_features // 16, num_features // 32, self.max_depth, is_final=True)
        self.conv1_seg = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 1, num_features // 16, 3, 1, 1, bias=False),
                                         nn.ELU())
        self.get_seg = nn.Conv2d(num_features // 16, 2, 3, 1, 1, bias=False)

    def forward(self, features, depth_token, seg_token, input_size):
        layer_sizes = [(f.shape[2], f.shape[3]) for f in features]
        layer_sizes.append(input_size)

        dense_features = torch.nn.ReLU()(features[0])
        upconv5 = self.upconv5(dense_features, new_size=layer_sizes[1])  # H/16
        upconv5 = self.bn5(upconv5)

        concat5 = torch.cat([upconv5, features[1]], dim=1)
        iconv5 = self.conv5(concat5)

        upconv4 = self.upconv4(iconv5, new_size=layer_sizes[2])  # H/8
        upconv4 = self.bn4(upconv4)
        concat4 = torch.cat([upconv4, features[2]], dim=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)

        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)

        upconv3 = self.upconv3(daspp_feat, new_size=layer_sizes[3])  # H/4
        upconv3 = self.bn3(upconv3)
        concat3 = torch.cat([upconv3, features[3]], dim=1)
        iconv3 = self.conv3(concat3)

        # depth prediction
        depth_mat = torch.cat((iconv3, depth_token), dim=1).permute(0, 2, 3, 1).contiguous()
        depth_mat = self.depth_token_fuse(depth_mat).permute(0, 3, 1, 2).contiguous()
        upconv2 = self.bn2_depth(self.upconv2_depth(depth_mat, new_size=(layer_sizes[3][0]*2, layer_sizes[3][1]*2)))  # H/2
        iconv2 = self.conv2_depth(upconv2)
        upconv1 = self.upconv1_depth(iconv2, new_size=layer_sizes[4])
        reduc1x1 = self.reduc1x1_depth(upconv1)
        iconv1 = self.conv1_depth(torch.cat([upconv1, reduc1x1], dim=1))
        depth_pred = self.max_depth * self.get_depth(iconv1)

        # seg prediction
        seg_mat = torch.cat((iconv3, seg_token), dim=1).permute(0, 2, 3, 1).contiguous()
        seg_mat = self.seg_token_fuse(seg_mat).permute(0, 3, 1, 2).contiguous()
        upconv2_s = self.bn2_seg(self.upconv2_seg(seg_mat, new_size=(layer_sizes[3][0]*2, layer_sizes[3][1]*2)))  # H/2
        iconv2_s = self.conv2_seg(upconv2_s)
        upconv1_s = self.upconv1_seg(iconv2_s, new_size=layer_sizes[4])
        reduc1x1_s = self.reduc1x1_seg(upconv1_s)
        iconv1_s = self.conv1_seg(torch.cat([upconv1_s, reduc1x1_s], dim=1))
        seg_pred = self.get_seg(iconv1_s)
        
        return None, None, None, depth_pred, seg_pred

class gseg(nn.Module):
    def __init__(self, max_depth, feat_out_channels, num_features=512):
        super(gseg, self).__init__()
        self.max_depth = max_depth

        self.upconv5 = upconv(feat_out_channels[4], num_features)
        self.bn5 = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)

        self.conv5 = torch.nn.Sequential(
            nn.Conv2d(num_features + feat_out_channels[3] + 256, num_features, 3, 1, 1, bias=False),
            nn.ELU())
        self.upconv4 = upconv(num_features, num_features // 2)
        self.bn4 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(num_features // 2 + feat_out_channels[2] + 256, num_features // 2, 3, 1, 1, bias=False),
            nn.ELU())
        self.bn4_2 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)

        self.daspp_3 = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6 = atrous_conv(num_features // 2 + num_features // 4 + feat_out_channels[2] + 256, num_features // 4, 6)
        # self.daspp_12 = atrous_conv(num_features + feat_out_channels[2] + 256, num_features // 4, 12)
        self.daspp_18 = atrous_conv(num_features + feat_out_channels[2] + 256, num_features // 4, 18)
        # self.daspp_24 = atrous_conv(num_features + num_features // 2 + feat_out_channels[2]+256, num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(
            nn.Conv2d(num_features + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())
        self.reduc8x8 = reduction_1x1(num_features // 4, num_features // 4, self.max_depth)
        self.lpg8x8 = local_planar_guidance(8)

        self.upconv3 = upconv(num_features // 4, num_features // 4)
        self.bn3 = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(num_features // 4 + feat_out_channels[1] + 128, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())
        self.reduc4x4 = reduction_1x1(num_features // 4, num_features // 8, self.max_depth)
        self.lpg4x4 = local_planar_guidance(4)

        self.upconv2 = upconv(num_features // 4, num_features // 8)
        self.bn2 = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(num_features // 8 + feat_out_channels[0] + 32, num_features // 8, 3, 1, 1, bias=False),
            nn.ELU())

        self.reduc2x2 = reduction_1x1(num_features // 8, num_features // 16, self.max_depth)
        self.lpg2x2 = local_planar_guidance(2)

        self.upconv1 = upconv(num_features // 8, num_features // 16)
        self.reduc1x1 = reduction_1x1(num_features // 16, num_features // 32, self.max_depth, is_final=True)
        self.conv1 = torch.nn.Sequential(nn.Conv2d(num_features // 16, num_features // 16, 3, 1, 1, bias=False),
                                         nn.ELU())
        # scale 0
        self.get_seg = torch.nn.Sequential(nn.Conv2d(num_features // 16, 2, 3, 1, 1, bias=False))

        # reflection hint
        #1/16
        self.sp_red1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            upconv(16, 32, ratio=0.5), #1/2
            )
        
        self.sp_red2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            upconv(64, 128, ratio=0.5),#1/4
            )
        self.sp_red3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            upconv(256, 256, ratio=0.5),#1/8
            )
        self.sp_red4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            upconv(256, 256, ratio=0.5),#1/16
            )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, features, ref_diff=None):
        if ref_diff is not None:
            ht1 = self.sp_red1(ref_diff)#1/2, 32
            ht2 = self.sp_red2(ht1)#1/4, 128
            ht3 = self.sp_red3(ht2)#1/8, 256
            ht4 = self.sp_red4(ht3)#1/16, 256

        skip0, skip1, skip2, skip3 = features[1], features[2], features[3], features[4]
        dense_features = torch.nn.ReLU()(features[5])
        upconv5 = self.upconv5(dense_features)  # H/16
        upconv5 = self.bn5(upconv5)

        concat5 = torch.cat([upconv5, skip3, ht4], dim=1)
        iconv5 = self.conv5(concat5)

        upconv4 = self.upconv4(iconv5)  # H/8
        upconv4 = self.bn4(upconv4)
        concat4 = torch.cat([upconv4, skip2, ht3], dim=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)

        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        # daspp_12 = self.daspp_12(concat4_3)
        # concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_3)
        # concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        # daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_18], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)
    
        # reduc8x8 = self.reduc8x8(daspp_feat)
        # plane_normal_8x8 = reduc8x8[:, :3, :, :]
        # plane_normal_8x8 = F.normalize(plane_normal_8x8, 2, 1)
        # plane_dist_8x8 = reduc8x8[:, 3, :, :]
        # plane_eq_8x8 = torch.cat([plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1)
        # depth_8x8 = self.lpg8x8(plane_eq_8x8, focal)
        # depth_8x8_scaled = depth_8x8.unsqueeze(1) / self.params.max_depth
        # depth_8x8_scaled_ds = F.interpolate(depth_8x8_scaled, scale_factor=0.25, mode='nearest')

        upconv3 = self.upconv3(daspp_feat)  # H/4
        upconv3 = self.bn3(upconv3)
        # concat3 = torch.cat([upconv3, skip1, depth_8x8_scaled_ds], dim=1)
        concat3 = torch.cat([upconv3, skip1, ht2], dim=1)
        iconv3 = self.conv3(concat3)
        #iconv3 torch.Size([1, 128, 104, 128])

        # reduc4x4 = self.reduc4x4(iconv3)
        # plane_normal_4x4 = reduc4x4[:, :3, :, :]
        # plane_normal_4x4 = F.normalize(plane_normal_4x4, 2, 1)
        # plane_dist_4x4 = reduc4x4[:, 3, :, :]
        # plane_eq_4x4 = torch.cat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
        # depth_4x4 = self.lpg4x4(plane_eq_4x4, focal)
        # depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.params.max_depth
        # depth_4x4_scaled_ds = F.interpolate(depth_4x4_scaled, scale_factor=0.5, mode='nearest')

        upconv2 = self.upconv2(iconv3)  # H/2
        upconv2 = self.bn2(upconv2)
        # concat2 = torch.cat([upconv2, skip0, depth_4x4_scaled_ds], dim=1)
        concat2 = torch.cat([upconv2, skip0, ht1], dim=1)
        #iconv2 torch.Size([1, 64, 208, 256])
        iconv2 = self.conv2(concat2)

        # reduc2x2 = self.reduc2x2(iconv2)
        # plane_normal_2x2 = reduc2x2[:, :3, :, :]
        # plane_normal_2x2 = F.normalize(plane_normal_2x2, 2, 1)
        # plane_dist_2x2 = reduc2x2[:, 3, :, :]
        # plane_eq_2x2 = torch.cat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
        # depth_2x2 = self.lpg2x2(plane_eq_2x2, focal)
        # depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.params.max_depth

        upconv1 = self.upconv1(iconv2)
        # reduc1x1 = self.reduc1x1(upconv1)
        # concat1 = torch.cat([upconv1, reduc1x1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], dim=1)

        iconv1 = self.conv1(upconv1)
        final_seg = self.get_seg(iconv1)

        return final_seg

def build_depth_decoder(args):
    return bts(args.max_depth, [64, 128, 256, 512], num_features=256)

def build_segmentation_decoder(args):
    gseg(args.max_depth, [64, 256, 512, 1024, 2048], num_features=512)
