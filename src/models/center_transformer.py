# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------
# import sys
# sys.path.append('/home/ly/workspace/git/segm/line_segm/letr-depth-2/src')

from curses import erasechar
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import init

import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv.utils import get_logger
import logging
from models.position_encoding import PositionEmbeddingSine
from util.misc import NestedTensor
from models.sne_model import  SNE

def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmseg".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """

    logger = get_logger(name='mmseg', log_file=log_file, log_level=log_level)

    return logger

def get_cam_param():
    camParam = torch.tensor([[6.360779e+02, 0.000000e+00, 6.348217e+02],
                            [0.000000e+00, 6.352265e+02, 3.570233e+02],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=torch.float32)
    return camParam

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

class Olp(nn.Module):
    """ one layer perceptron."""

    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or 1
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class ConvA(nn.Module):
    """ one layer convolution and activation."""

    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or 1
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., num_ref=30, args=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # diffusion params
        # self.scale = dim ** -0.5
        self.diff_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.diff_logsigma = nn.Parameter(torch.zeros(1, 1, dim))

        self.border_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.border_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.diff_logsigma)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        #reference attention passing
        self.num_ref = num_ref # reference line contains three points
        self.ref_qk = nn.Linear(dim, dim * 2, bias=True)
        self.ref_attn_diffusion = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=3, padding=1)
        # self.ref_attn_norm = nn.LayerNorm(self.window_size*self.window_size*num_ref, )
        # self.gru = nn.GRUCell(self.window_size*self.window_size*num_ref, self.window_size*self.window_size*num_ref)

    def forward(self, x, mask=None, x_ref=None, **kwargs):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        ref_qk = self.ref_qk(x_ref).reshape(x_ref.shape[0], x_ref.shape[1], 2, C).permute(2, 0, 1, 3)
        ref_q, ref_v = ref_qk[0], ref_qk[1]

        rB, n_rf, rC = ref_q.shape
        n_win = B_ // rB
        mu = self.diff_mu.expand(rB, n_rf, -1)
        sigma = self.diff_logsigma.exp().expand(rB, n_rf, -1)
        ref_q = mu + sigma * ref_q
        ref_q = ref_q.reshape(rB, n_rf, self.num_heads, rC // self.num_heads).permute(0, 2, 1, 3) # (batchsize, num_head, num_ref, C//num_head)
        ref_k = torch.cat([ref_q[i:i+1].expand(n_win, -1, -1, -1) for i in range(rB)], dim=0)

        ref_v = ref_v.reshape(rB, n_rf, self.num_heads, rC // self.num_heads).permute(0, 2, 1, 3)
        ref_v = torch.cat([ref_v[i:i+1].expand(n_win, -1, -1, -1) for i in range(rB)], dim=0)

        ref_attn = (q @ ref_k.transpose(-2, -1)) # (num_windows*B, num_head, N, num_ref)
        ref_attn_r = ref_attn.view(rB, n_win, self.num_heads, N, n_rf).permute(0, 2, 1, 3, 4).reshape(rB,
            self.num_heads, n_win * N, n_rf).contiguous() # (B, num_head, num_windows*N, num_ref)
        for i in range(3):
            ref_attn_update = self.ref_attn_diffusion(ref_attn_r)
            ref_attn_update = F.gelu(F.layer_norm(ref_attn_update, [n_win*N, n_rf]))
            ref_attn_r = ref_attn_r + ref_attn_update
        
        ref_attn = ref_attn_r.reshape(rB, self.num_heads, n_win, N, n_rf).permute(0, 2, 1, 3, 4).reshape(
            rB*n_win, self.num_heads, N, n_rf) # (num_windows*B, num_head, N, num_ref)
        
        ref_attn = self.softmax(ref_attn)
        q_new = ref_attn @ ref_v

        q_new = q_new * self.scale
        attn = (q_new @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kwargs['depth_token'], kwargs['seg_token']

class WindowClassAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., num_ref=30, args=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # diffusion params
        # self.scale = dim ** -0.5
        self.diff_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.diff_logsigma = nn.Parameter(torch.zeros(1, 1, dim))

        self.border_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.border_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.diff_logsigma)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # t_dim = dim + args.class_token_dim + args.class_token_dim
        # self.qkv = nn.Linear(t_dim, t_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        #reference attention passing
        self.num_ref = num_ref # reference line contains three points
        self.ref_qk = nn.Linear(dim, dim * 2, bias=True)
        self.ref_attn_diffusion = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=3, padding=1)

        # class toekn fuse
        self.cls_dth_q = nn.Linear(args.class_token_dim, args.class_token_dim)
        self.cls_seg_q = nn.Linear(args.class_token_dim, args.class_token_dim)
        self.global_k = nn.Linear(dim + args.class_token_dim * 2, dim + args.class_token_dim * 2)
        self.global_v = nn.Linear(dim + args.class_token_dim * 2, dim + args.class_token_dim * 2)
        self.proj_dth = nn.Linear(args.class_token_dim, args.class_token_dim)
        self.proj_dth_drop = nn.Dropout(proj_drop)
        self.proj_seg = nn.Linear(args.class_token_dim, args.class_token_dim)
        self.proj_seg_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None, x_ref=None, depth_token=None, seg_token=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        # ref_qk = self.ref_qk(x_ref).reshape(x_ref.shape[0], x_ref.shape[1], 2, C).permute(2, 0, 1, 3)
        # ref_q, ref_v = ref_qk[0], ref_qk[1]

        # rB, n_rf, rC = ref_q.shape
        # n_win = B_ // rB
        # mu = self.diff_mu.expand(rB, n_rf, -1)
        # sigma = self.diff_logsigma.exp().expand(rB, n_rf, -1)
        # ref_q = mu + sigma * ref_q
        # ref_q = ref_q.reshape(rB, n_rf, self.num_heads, rC // self.num_heads).permute(0, 2, 1, 3) # (batchsize, num_head, num_ref, C//num_head)
        # ref_k = torch.cat([ref_q[i:i+1].expand(n_win, -1, -1, -1) for i in range(rB)], dim=0)

        # ref_v = ref_v.reshape(rB, n_rf, self.num_heads, rC // self.num_heads).permute(0, 2, 1, 3)
        # ref_v = torch.cat([ref_v[i:i+1].expand(n_win, -1, -1, -1) for i in range(rB)], dim=0)

        # ref_attn = (q @ ref_k.transpose(-2, -1)) # (num_windows*B, num_head, N, num_ref)
        # ref_attn_r = ref_attn.view(rB, n_win, self.num_heads, N, n_rf).permute(0, 2, 1, 3, 4).reshape(rB,
        #     self.num_heads, n_win * N, n_rf).contiguous() # (B, num_head, num_windows*N, num_ref)
        # for i in range(3):
        #     ref_attn_update = self.ref_attn_diffusion(ref_attn_r)
        #     ref_attn_update = F.gelu(F.layer_norm(ref_attn_update, [n_win*N, n_rf]))
        #     ref_attn_r = ref_attn_r + ref_attn_update
        
        # ref_attn = ref_attn_r.reshape(rB, self.num_heads, n_win, N, n_rf).permute(0, 2, 1, 3, 4).reshape(
        #     rB*n_win, self.num_heads, N, n_rf) # (num_windows*B, num_head, N, num_ref)
        
        # ref_attn = self.softmax(ref_attn)
        # q_new = ref_attn @ ref_v

        q_new = q
        q_new = q_new * self.scale
        attn = (q_new @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)

        # class token
        depth_q = self.cls_dth_q(depth_token).reshape(B_, N, self.num_heads, depth_token.shape[-1] // self.num_heads).permute(0, 2, 1, 3).contiguous()
        seg_q = self.cls_seg_q(seg_token).reshape(B_, N, self.num_heads, seg_token.shape[-1] // self.num_heads).permute(0, 2, 1, 3).contiguous()
        t_x = torch.cat([x, depth_token,  seg_token], dim=-1)
        tC = t_x.shape[-1]
        t_k = self.global_k(t_x).reshape(B_, N, self.num_heads, tC // self.num_heads).permute(0, 2, 1, 3).contiguous()
        t_v = self.global_v(t_x).reshape(B_, N, self.num_heads, tC // self.num_heads).permute(0, 2, 1, 3).contiguous()

        depth_q = depth_q *self.scale
        depth_attn = depth_q.transpose(-2, -1).contiguous() @ t_k
        depth_attn = self.attn_drop(self.softmax(depth_attn))
        depth_token = (depth_attn @ t_v.transpose(-2, -1)).reshape(B_, -1, N).permute(0, 2, 1).contiguous()
        depth_token = self.proj_drop(self.proj_dth(depth_token))

        seg_q = seg_q *self.scale
        seg_attn = seg_q.transpose(-2, -1) @ t_k
        seg_attn = self.attn_drop(self.softmax(seg_attn))
        seg_token = (seg_attn @ t_v.transpose(-2, -1)).reshape(B_, -1, N).permute(0, 2, 1).contiguous()
        seg_token = self.proj_drop(self.proj_dth(seg_token))

        return x, depth_token, seg_token

class TokenFuse(nn.Module):
    def __init__(self, dim, args=None):
        super().__init__()
        self.depth_proj = Mlp(in_features=dim, hidden_features=dim * 4, out_features=dim)
        self.depth_pred = torch.nn.Sequential(nn.Conv2d(dim, 1, 3, 1, 1, bias=False),
                                             nn.Sigmoid())
        self.sne = SNE()
        self.args = args
    
    def forward(self, depth_token, seg_token, sample_coors, pos_emb, with_pos=True):
        #token [B, H, W, C], sample_coors [B, num_line, 3, 2]
        print('depth_token, seg_token, sample_coors, pos_emb,', depth_token.shape, seg_token.shape, sample_coors.shape, pos_emb.shape)
        depth_feature = self.depth_proj(depth_token)
        depth_pred = self.depth_pred(depth_feature.permute(0, 3, 1, 2)) * self.args.max_depth
        samp_depth = F.grid_sample(depth_pred, sample_coors, mode='nearest') #(B, 1, num_line, 3)
        cam_param = get_cam_param()
        cam_param = cam_param.cuda(samp_depth.device)
        samp_norm_list = []
        for i in range(len(samp_depth)):
            samp_norm_list.append(self.sne(samp_depth[i].squeeze(0), cam_param))
        samp_norm = torch.stack(samp_norm_list)#(B, 3, num_line, 3)

        samp_seg_token = F.grid_sample(seg_token.permute(0, 3, 1, 2), sample_coors, mode='nearest') #(B, C, num_line, 3)

        

class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_attn=WindowAttention, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_ref=100, class_pred=False, args=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = window_attn(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_ref=num_ref, args=args)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.class_pred = class_pred
        if class_pred:
            self.token_fuse = TokenFuse(dim=args.class_token_dim, args=args)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix, ref_coors, ref_pos=None, depth_token=None, seg_token=None):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
            
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
            ref_roll_coors = torch.zeros_like(ref_coors)
            ref_roll_coors[:, :, :, 0] = ref_coors[:, :, :, 0] - self.shift_size / Wp
            ref_roll_coors[:, :, :, 1] = ref_coors[:, :, :, 1] - self.shift_size / Hp
            ref_roll_coors = torch.clamp(ref_roll_coors, min=0)
            if ref_pos is not None:
                shifted_ref_pos = torch.roll(ref_pos, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x
            attn_mask = None
            ref_roll_coors = ref_coors
            shifted_ref_pos = ref_pos

        x_ref = F.grid_sample(shifted_x.permute(0, 3, 1, 2), ref_roll_coors, mode='nearest') # (B, C, line_num, 3)
        if ref_pos is not None:
            x_ref_pos = F.grid_sample(shifted_ref_pos, ref_roll_coors, mode='nearest') # (B, C, line_num, 3)
            x_ref = x_ref + x_ref_pos
        x_ref = x_ref.reshape(B, C, -1).permute(0, 2, 1)

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        if depth_token is not None and seg_token is not None:
            tC = depth_token.shape[2]
            depth_token = depth_token.view(B, H, W, tC)
            seg_token = seg_token.view(B, H, W, tC)
            depth_token = F.pad(depth_token, (0, 0, pad_l, pad_r, pad_t, pad_b))
            seg_token = F.pad(seg_token, (0, 0, pad_l, pad_r, pad_t, pad_b))
            if self.shift_size > 0:
                depth_token = torch.roll(depth_token, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                seg_token = torch.roll(seg_token, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            depth_token_windows = window_partition(depth_token, self.window_size)  # nW*B, window_size, window_size, C
            depth_token = depth_token_windows.view(-1, self.window_size * self.window_size, tC)  # nW*B, window_size*window_size, C
            seg_token_windows = window_partition(seg_token, self.window_size)  # nW*B, window_size, window_size, C
            seg_token = seg_token_windows.view(-1, self.window_size * self.window_size, tC)  # nW*B, window_size*window_size, C
        # W-MSA/SW-MSA
        attn_windows, depth_token,  seg_token = self.attn(x_windows, mask=attn_mask, x_ref=x_ref, depth_token=depth_token, seg_token=seg_token)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
                    

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if depth_token is not None and seg_token is not None:
            # merge class token windows
            tC = depth_token.shape[-1]

            depth_token = depth_token.view(-1, self.window_size, self.window_size, tC)
            depth_token = window_reverse(depth_token, self.window_size, Hp, Wp)  # B H' W' C
            seg_token = seg_token.view(-1, self.window_size, self.window_size, tC)
            seg_token = window_reverse(seg_token, self.window_size, Hp, Wp)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                depth_token = torch.roll(depth_token, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                seg_token = torch.roll(seg_token, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            if pad_r > 0 or pad_b > 0:
                depth_token = depth_token[:, :H, :W, :].contiguous()
                seg_token = seg_token[:, :H, :W, :].contiguous()
            
            if self.class_pred:
                aa = self.token_fuse(depth_token, seg_token, ref_coors, ref_pos, with_pos=True)

            depth_token = depth_token.view(B, H * W, tC)       
            seg_token = seg_token.view(B, H * W, tC)         
        

        return x, depth_token,  seg_token


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_attn=WindowAttention,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False, 
                 num_ref=100,
                 class_pred=False,
                 args=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_attn=window_attn,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, 
                num_ref=num_ref,
                class_pred=class_pred if i == depth - 1 else False,
                args=args)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W, ref_coors, ref_pos, depth_token=None, seg_token=None):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x, depth_token, seg_token = blk(x, attn_mask, ref_coors=ref_coors, ref_pos=ref_pos, depth_token=depth_token, seg_token=seg_token)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww, depth_token, seg_token
        else:
            return x, H, W, x, H, W, depth_token, seg_token


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class ReferTransformer(nn.Module):
    def __init__(self, args, feat_out_channels):
        super().__init__()
        self.depth_token = nn.Parameter(torch.zeros(1, 1, args.class_token_dim))
        self.seg_token = nn.Parameter(torch.zeros(1, 1, args.class_token_dim))

        self.pos_enc = PositionEmbeddingSine(num_pos_feats=args.dense_trans_dim // 2)
        self.dense_transformer = BasicLayer(dim=args.dense_trans_dim, depth=args.dense_trans_layers[0], num_heads=args.dense_trans_heads, \
            window_attn=WindowAttention, window_size=7, mlp_ratio=2, num_ref=args.num_ref * 3, class_pred=False, args=args)
        #1/16
        self.pos_cls1 = PositionEmbeddingSine(num_pos_feats=args.dense_trans_dim // 4)
        self.proj_class1 = nn.Linear(args.dense_trans_dim, args.dense_trans_dim // 2)
        self.proj_backbn1  = ConvA(feat_out_channels[-2], out_features=args.dense_trans_dim // 2)
        self.class_transformer1 = BasicLayer(dim=args.dense_trans_dim // 2, depth=args.class_trans_layers[0], num_heads=args.dense_trans_heads, \
            window_attn=WindowClassAttention, window_size=7, mlp_ratio=2, num_ref=args.num_ref * 3, class_pred=False, args=args)
        
        #1/8
        self.pos_cls2 = PositionEmbeddingSine(num_pos_feats=args.dense_trans_dim // 8)
        self.proj_class2 = nn.Linear(args.dense_trans_dim // 2, args.dense_trans_dim // 4)
        self.proj_backbn2  = ConvA(feat_out_channels[-3], out_features=args.dense_trans_dim // 4)
        self.class_transformer2 = BasicLayer(dim=args.dense_trans_dim // 4, depth=args.class_trans_layers[1], num_heads=args.dense_trans_heads, \
            window_attn=WindowClassAttention, window_size=7, mlp_ratio=2, num_ref=args.num_ref * 3, class_pred=False, args=args)

        #1/4
        self.pos_cls3 = PositionEmbeddingSine(num_pos_feats=args.dense_trans_dim // 16)
        self.proj_class3 = nn.Linear(args.dense_trans_dim // 4, args.dense_trans_dim // 8)
        self.proj_backbn3  = ConvA(feat_out_channels[-4], out_features=args.dense_trans_dim // 8)
        self.class_transformer3 = BasicLayer(dim=args.dense_trans_dim // 8, depth=args.class_trans_layers[2], num_heads=args.dense_trans_heads, \
            window_attn=WindowClassAttention, window_size=7, mlp_ratio=2, num_ref=args.num_ref * 3, class_pred=False, args=args)

        self.num_ref = args.num_ref
        self.args = args
        trunc_normal_(self.depth_token, std=.02)
        trunc_normal_(self.seg_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, nested_top_mat, features, sample_points, sample_points_scores, layers_size=None, with_pos=True):
        top_mat, mask = nested_top_mat.decompose()
        B, C, H, W = top_mat.shape
        t_values, t_ids = torch.topk(sample_points_scores[:, :, 0], self.num_ref, dim=-1)
        choosen_points = torch.stack([sample_points[i][t_ids[i]] for i in range(B)])
        choosen_points = choosen_points.reshape(B, self.num_ref, -1, 2)
        # norm to (-1, 1)
        choosen_points = choosen_points * 2 - 1.0
        
        pos_emb = self.pos_enc(nested_top_mat) if with_pos else None
        d_enc_out, H, W, x, Wh, Ww, _, _ = self.dense_transformer(top_mat.flatten(2).permute(0, 2, 1), H, W, choosen_points, pos_emb)

        # d_enc_out (B, N, C)
        dense_out = d_enc_out.permute(0, 2, 1).reshape(-1, C, H, W)
        H_cls1, W_cls1 = layers_size[0]
        dense_out_up = F.interpolate(dense_out, size=(H_cls1, W_cls1), mode='nearest')
        cls_enc_inp1 = self.proj_class1(dense_out_up.flatten(2).permute(0, 2, 1))
        backbn_mat1, mask1 = features[self.args.layer1_num - 1].decompose()
        cls_enc_inp1 = cls_enc_inp1 + self.proj_backbn1(backbn_mat1).flatten(2).permute(0, 2, 1)
        pos_emb_cls1 = self.pos_cls1(NestedTensor(dense_out_up, mask1))
        depth_token = self.depth_token.expand(B, H_cls1 * W_cls1, -1)
        seg_token = self.seg_token.expand(B, H_cls1 * W_cls1, -1)
        cls_enc_out1, H1, W1, x1, Wh1, Ww1, depth_token, seg_token  = self.class_transformer1(cls_enc_inp1, H_cls1, W_cls1, choosen_points, pos_emb_cls1, 
            depth_token=depth_token, seg_token=seg_token)

        C1 = cls_enc_out1.shape[-1]
        # cls_enc_out (B, N, C1)
        cls_enc_out1 = cls_enc_out1.permute(0, 2, 1).reshape(-1, C1, H1, W1)
        H_cls2, W_cls2 = layers_size[1]
        cls_enc_out1_up = F.interpolate(cls_enc_out1, size=(H_cls2, W_cls2), mode='nearest')
        cls_enc_inp2 = self.proj_class2(cls_enc_out1_up.flatten(2).permute(0, 2, 1))
        backbn_mat2, mask2 = features[self.args.layer1_num - 2].decompose()
        cls_enc_inp2 = cls_enc_inp2 + self.proj_backbn2(backbn_mat2).flatten(2).permute(0, 2, 1)
        pos_emb_cls2 = self.pos_cls2(NestedTensor(cls_enc_out1_up, mask2))
        depth_token = self.depth_token.expand(B, H_cls2 * W_cls2, -1)
        seg_token = self.seg_token.expand(B, H_cls2 * W_cls2, -1)
        cls_enc_out2, H2, W2, x2, Wh2, Ww2, depth_token, seg_token  = self.class_transformer2(cls_enc_inp2, H_cls2, W_cls2, choosen_points, pos_emb_cls2, 
            depth_token=depth_token, seg_token=seg_token)

        C2 = cls_enc_out2.shape[-1]
        # cls_enc_out2 (B, N, C2)
        cls_enc_out2 = cls_enc_out2.permute(0, 2, 1).reshape(-1, C2, H2, W2)
        H_cls3, W_cls3 = layers_size[2]
        cls_enc_out2_up = F.interpolate(cls_enc_out2, size=(H_cls3, W_cls3), mode='nearest')
        cls_enc_inp3 = self.proj_class3(cls_enc_out2_up.flatten(2).permute(0, 2, 1))
        backbn_mat3, mask3 = features[self.args.layer1_num - 3].decompose()
        cls_enc_inp3 = cls_enc_inp3 + self.proj_backbn3(backbn_mat3).flatten(2).permute(0, 2, 1)
        pos_emb_cls3 = self.pos_cls3(NestedTensor(cls_enc_out2_up, mask3))
        depth_token = self.depth_token.expand(B, H_cls3 * W_cls3, -1)
        seg_token = self.seg_token.expand(B, H_cls3 * W_cls3, -1)
        cls_enc_out3, H3, W3, x3, Wh3, Ww3, depth_token, seg_token  = self.class_transformer3(cls_enc_inp3, H_cls3, W_cls3, choosen_points, pos_emb_cls3, 
            depth_token=depth_token, seg_token=seg_token)

        C3 = cls_enc_out3.shape[-1]
        dense_out_list = [dense_out, #1/32
                        cls_enc_out1, #1/16
                        cls_enc_out2, #1/8
                        cls_enc_out3.permute(0, 2, 1).reshape(-1, C3, H3, W3),] #1/4
        depth_token = depth_token.permute(0, 2, 1).reshape(-1, C3, H3, W3) #1/4
        seg_token = seg_token.permute(0, 2, 1).reshape(-1, C3, H3, W3) #1/4
        return dense_out_list, depth_token, seg_token

def ref_trans_test():

    from args import get_args_parser
    parser = get_args_parser()
    args = parser.parse_args()

    Trans = build_dense_transformer(args)
    mat = torch.randn(2, 512, 24, 26)
    mask = torch.randint(0, 2, (2, 24, 26))
    NT = NestedTensor(mat, mask)
    scores = torch.rand(2, 100)
    ref_coords = torch.rand(2, 100, 6)
    SR = ReferTransformer(args)
    B, C, H, W = mat.shape
    # ref_x [B, C, top_num, points_num]
    ref_x, choosen_points = SR(ref_coords, scores, mat, top=args.num_ref, nst=NT, with_pos=True)
    ref_lines = ref_x[:,:,:,:2].reshape(B, C, -1).permute(0, 2, 1) # (B, feature_num, C)
    ref_centers = ref_x[:,:,:,2:].reshape(B, C, -1).permute(0, 2, 1)
    
    line_coords = choosen_points[:, :, :2].reshape(B, -1, 2) # (B, top_num, points_num, 2)
    center_coords = choosen_points[:, :, 2:].reshape(B, -1, 2)
    ref_lines = torch.cat([ref_lines, line_coords], dim=-1)
    ref_centers = torch.cat([ref_centers, center_coords], dim=-1)

    _,C, h, w = mat.shape
    d_enc_out, H, W, x, Wh, Ww = Trans(mat.flatten(2).permute(0, 2, 1), h, w, ref_centers, ref_lines, choosen_points)
    print('d_enc_out', d_enc_out.shape)

def build_dense_transformer(args):
    # b = BasicLayer(dim=args.dense_trans_dim, depth=args.dense_trans_layers[0], num_heads=args.dense_trans_heads, \
    #         window_size=7, mlp_ratio=2, num_ref=args.num_ref * 3)
    RT = ReferTransformer(args, [256, 512, 1024, 512])
    return RT

if __name__ == '__main__':
    ref_trans_test()