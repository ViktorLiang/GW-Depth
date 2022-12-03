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


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class bts(nn.Module):
    def __init__(self, max_depth, feat_out_channels, num_features=256, args=None):
        super(bts, self).__init__()
        self.max_depth = max_depth

        # token fuse
        self.depth_token_fuse = Mlp(in_features=num_features + 1 + args.class_token_dim, out_features=args.class_token_dim)
        self.seg_token_fuse = Mlp(in_features=num_features + args.class_token_dim, out_features=args.class_token_dim)
        
        # depth pred
        self.upconv1_depth = upconv(args.class_token_dim, args.class_token_dim)
        self.bn1_depth = nn.BatchNorm2d(args.class_token_dim, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv1_depth = torch.nn.Sequential(
            nn.Conv2d(args.class_token_dim, args.class_token_dim, 3, 1, 1, bias=False),
            nn.ELU())

        self.upconv2_depth = upconv(args.class_token_dim, args.class_token_dim // 2)
        self.conv2_depth = torch.nn.Sequential(nn.Conv2d(args.class_token_dim // 2, args.class_token_dim // 2, 3, 1, 1, bias=False),
                                         nn.ELU())
        self.get_depth = torch.nn.Sequential(nn.Conv2d(args.class_token_dim // 2, 1, 3, 1, 1, bias=False),
                                             nn.Sigmoid())
        
        # seg pred
        self.upconv1_seg = upconv(args.class_token_dim, args.class_token_dim)
        self.bn1_seg = nn.BatchNorm2d(args.class_token_dim, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv1_seg = torch.nn.Sequential(
            nn.Conv2d(args.class_token_dim, args.class_token_dim, 3, 1, 1, bias=False),
            nn.ELU())

        self.upconv2_seg = upconv(args.class_token_dim, args.class_token_dim // 2)
        self.conv2_seg = torch.nn.Sequential(nn.Conv2d(args.class_token_dim // 2, args.class_token_dim // 2, 3, 1, 1, bias=False),
                                         nn.ELU())
        self.get_seg = nn.Conv2d(args.class_token_dim // 2, 2, 3, 1, 1, bias=False)

    def forward(self, decoder_top_feat, decoder_depth_pred, depth_token, seg_token, input_size):
        # depth prediction
        dec_fuse = torch.cat([decoder_top_feat, decoder_depth_pred, depth_token], dim=1)
        B, _, H, W = dec_fuse.shape
        detph_feats = self.depth_token_fuse(dec_fuse.flatten(2).permute(0, 2, 1))
        detph_feats = detph_feats.permute(0, 2, 1).reshape(B, -1, H, W)
        upconv1 = self.bn1_depth(self.upconv1_depth(detph_feats))  # H/2
        conv1d = self.conv1_depth(upconv1)
        upconv2 = self.upconv2_depth(conv1d, new_size=input_size)
        conv2d = self.conv2_depth(upconv2)
        depth_pred = self.max_depth * self.get_depth(conv2d)

        # seg prediction
        seg_dec_fuse = torch.cat([decoder_top_feat, seg_token], dim=1)
        B, _, H, W = seg_dec_fuse.shape
        seg_feats = self.seg_token_fuse(seg_dec_fuse.flatten(2).permute(0, 2, 1))
        seg_feats = seg_feats.permute(0, 2, 1).reshape(B, -1, H, W)
        upconv1_seg = self.bn1_seg(self.upconv1_seg(seg_feats))  # H/2
        conv1s = self.conv1_seg(upconv1_seg)
        upconv2_seg = self.upconv2_seg(conv1s, new_size=input_size)
        conv2s = self.conv2_seg(upconv2_seg)
        seg_pred = self.get_seg(conv2s)
        return depth_pred, seg_pred

def build_depth_decoder(args):
    return bts(args.max_depth, [64, 128, 256, 512], num_features=64, args=args)