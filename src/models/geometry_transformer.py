# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------
# import sys
# sys.path.append('/home/ly/workspace/git/segm/line_segm/letr-depth-2/src')
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import init

import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

#from mmcv.utils import get_logger
import logging
from models.position_encoding import PositionEmbeddingSine
from src import args
from util.misc import NestedTensor

from src.models.geometry_utils import NonLocalPlannarGuidance, Global2PointGraph, PointGuidedTokenFuse
from src.models.points.points_sample import OffsetGeneration
# from src.models.geometry_utils import TokenRelationWithBackbone
from src.util.commons import show_sampled_points, show_smapled_lines
from src.models.swin_transformer import WindowAttention as OriginWindowAttention
from src.models.points.points_sample import sample_along_seg, sample_mid_seg

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

class MlpNorm(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=None, norm_layer=None, drop=0.):
        super().__init__()
        self.with_act = True if act_layer else False
        self.with_norm = True if norm_layer else False

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        if act_layer:
            self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if norm_layer:
            self.norm = norm_layer(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        if self.with_act:
            x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if self.with_norm:
            x = self.norm(x)
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


class PointAttention(nn.Module):
    # def __init__(self, dim, num_heads, qk_scale=None, qkv_bias=True  attn_drop=0., proj_drop=0., args=None):
    def __init__(self, dim, num_heads, qk_scale=None, attn_drop=0., proj_drop=0., args=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.ref_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x_ref):
        rB, n_rf, rC = x_ref.shape
        x_ref_shourtcut = x_ref
        ref_qkv = self.ref_qkv(x_ref).reshape(x_ref.shape[0], x_ref.shape[1], 3, rC).permute(2, 0, 1, 3)
        ref_q, ref_k, ref_v = ref_qkv[0], ref_qkv[1], ref_qkv[2]
        ref_q = ref_q.reshape(rB, n_rf, self.num_heads, rC // self.num_heads).permute(0, 2, 1, 3) # (batchsize, num_head, num_ref, C//num_head)
        ref_k = ref_k.reshape(rB, n_rf, self.num_heads, rC // self.num_heads).permute(0, 2, 1, 3) # (batchsize, num_head, num_ref, C//num_head)
        ref_v = ref_v.reshape(rB, n_rf, self.num_heads, rC // self.num_heads).permute(0, 2, 1, 3) # (batchsize, num_head, num_ref, C//num_head)
        ref_q = ref_q * self.scale
        ref_attn = ref_q @ ref_k.permute(0, 1, 3, 2)
        ref_attn = self.softmax(ref_attn)
        ref_x = ref_attn @ ref_v
        ref_x = ref_x.permute(0, 2, 1, 3).reshape(rB, -1, rC)
        x_ref = self.proj_drop(self.proj(ref_x)) + x_ref_shourtcut
        return x_ref


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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., size_ratio=0.25, group_attention=False, args=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.args = args

        # diffusion params
        # self.scale = dim ** -0.5
        self.diff_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.diff_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
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
        self.size_ratio = size_ratio # reference line contains th
        # if args.with_point_attention:
        #     self.point_attention = PointAttention(dim, num_heads, attn_drop=0.1, proj_drop=0.1,)
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
        
        # if self.args.with_point_attention:
        #     x_ref = self.point_attention(x_ref)

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

        q = q * self.scale
        ref_attn = (q @ ref_k.transpose(-2, -1)) # (num_windows*B, num_head, N, num_ref)
        ref_attn_r = ref_attn.view(rB, n_win, self.num_heads, N, n_rf).permute(0, 2, 1, 3, 4).reshape(rB,
            self.num_heads, n_win * N, n_rf).contiguous() # (B, num_head, num_windows*N, num_ref)
        for _ in range(3):
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

class PointTokenAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., args=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or args.class_token_dim ** -0.5
        self.global_proj = nn.Linear(dim, args.class_token_dim)
        # self.global_token_proj = nn.Linear(window_size**2 + args.num_ref * 2, window_size**2)
        self.global_token_proj = nn.Linear(args.class_token_dim, args.class_token_dim)

        self.cls_pnt_q = nn.Linear(args.class_token_dim, args.class_token_dim)
        self.global_k = nn.Linear(args.class_token_dim, args.class_token_dim)
        self.global_v = nn.Linear(args.class_token_dim, args.class_token_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_token = nn.Linear(args.class_token_dim, args.class_token_dim)
        self.proj_token_drop = nn.Dropout(proj_drop)

    # point_token (B, num_pnt, cls_token_dim)
    def forward(self, x, point_token=None):
        B, H, W, C = x.shape
        B, num_pnt, cls_dim = point_token.shape
        x_g = self.global_proj(x.flatten(start_dim=1, end_dim=2))
        # (B, num_pnt, cls_dim) -> (B, num_pnt, num_head, head_dim) -> (B, num_head, num_pnt, head_dim)
        pnt_q = self.cls_pnt_q(point_token).reshape(B, num_pnt, self.num_heads, cls_dim // self.num_heads).permute(0, 2, 1, 3).contiguous()

        t_x = torch.cat([x_g, point_token], dim=1)
        t_x = self.global_token_proj(t_x)
        tC = t_x.shape[-1]
        N = H * W + num_pnt
        t_k = self.global_k(t_x).reshape(B, N, self.num_heads, tC // self.num_heads).permute(0, 2, 1, 3).contiguous()
        t_v = self.global_v(t_x).reshape(B, N, self.num_heads, tC // self.num_heads).permute(0, 2, 1, 3).contiguous()
        
        pnt_q = pnt_q * self.scale
        pnt_attn = pnt_q @ t_k.transpose(-2, -1).contiguous()
        pnt_attn = self.attn_drop(self.softmax(pnt_attn))
        pnt_token = (pnt_attn @ t_v).permute(0, 2, 1, 3).reshape(B, num_pnt, -1).contiguous()
        pnt_token = self.proj_token_drop(self.proj_token(pnt_token))

        return pnt_token

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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., size_ratio=0.25, group_attention=False, args=None):

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

        self.group_attention = group_attention
        if group_attention:
            #reference attention passing
            self.size_ratio = size_ratio # reference line contains three points
            self.ref_qk = nn.Linear(dim, dim * 2, bias=True)
            self.ref_attn_diffusion = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=3, padding=1)
            self.q_norm = nn.LayerNorm(dim)
            self.q_proj = nn.Linear(dim, dim)

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
        
        if self.group_attention:
            q_shortcut = q
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

            q = q * self.scale
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
            q_new = q_new.permute(0, 2, 1, 3).flatten(2)
            q_shortcut = q_shortcut.permute(0, 2, 1, 3).flatten(2)
            q_new = self.q_proj(self.q_norm(q_new+q_shortcut)) + q_shortcut
            q_new = q_new.reshape(-1, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
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
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, size_ratio=0.25, token_fuse=False, num_points=None, with_point_token=False, group_attention=False, args=None):
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
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, group_attention=group_attention, args=args)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # class token
        self.token_fuse = token_fuse
        if window_attn == WindowClassAttention:
            self.norm_seg1 = norm_layer(args.class_token_dim)
            self.norm_depth1 = norm_layer(args.class_token_dim) 
            self.drop_path_class = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            mlp_dense_hidden_dim = int(args.class_token_dim * mlp_ratio)
            self.mlp_seg = Mlp(in_features=args.class_token_dim, hidden_features=mlp_dense_hidden_dim, act_layer=act_layer, drop=drop)
            self.norm_seg2 = norm_layer(args.class_token_dim)
            self.mlp_depth = Mlp(in_features=args.class_token_dim, hidden_features=mlp_dense_hidden_dim, act_layer=act_layer, drop=drop)
            self.norm_depth2 = norm_layer(args.class_token_dim)
            if token_fuse:
                # self.token_relation = TokenFuse(size_ratio=size_ratio, args=args)
                self.token_relation = PointGuidedTokenFuse(self.dim, args=args)
        

        if args.with_line_depth and with_point_token:
            # __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., size_ratio=0.25, args=None):
            self.pointTokenAttn = PointTokenAttention(dim, num_heads, qkv_bias=True, args=args)

        self.H = None
        self.W = None
        self.args = args

    def forward(self, x, mask_matrix, ref_coors, ref_pos=None, depth_token=None, seg_token=None, point_token=None, token_pos=None):
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
        if depth_token is not None and seg_token is not None:
            depth_token_shortcut = depth_token
            seg_token_shortcut = seg_token
            depth_token = self.norm_depth1(depth_token)
            seg_token = self.norm_seg1(seg_token)

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
            if ref_coors is not None:
                # roll refered coords (bounds between -1~1)
                ref_roll_coors = torch.zeros_like(ref_coors)
                ref_roll_coors[:, :, :, 0] = ref_coors[:, :, :, 0] - ((self.shift_size / (Wp - 1)) * 2)
                ref_roll_coors[:, :, :, 1] = ref_coors[:, :, :, 1] - ((self.shift_size / (Hp - 1)) * 2)
                # mod the coords that cross the bound
                ref_roll_coors[ref_roll_coors < -1] = -1 - (1+ref_roll_coors[ref_roll_coors < -1])
                if ref_pos is not None:
                    shifted_ref_pos = torch.roll(ref_pos, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x
            attn_mask = None
            ref_roll_coors = ref_coors
            shifted_ref_pos = ref_pos

        if ref_coors is not None:
            x_ref = F.grid_sample(shifted_x.permute(0, 3, 1, 2), ref_roll_coors, mode='nearest') # (B, C, line_num, 3)
            if ref_pos is not None:
                x_ref_pos = F.grid_sample(shifted_ref_pos, ref_roll_coors, mode='nearest') # (B, C, line_num, 3)
                x_ref = x_ref + x_ref_pos
                if point_token is not None:
                    pnt_token_pos = x_ref_pos.flatten(2)[:, :self.args.class_token_dim].permute(0, 2, 1)
                    point_token = point_token + pnt_token_pos
            x_ref = x_ref.reshape(B, C, -1).permute(0, 2, 1)
        else:
            x_ref = None

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
        if not isinstance(self.attn, OriginWindowAttention): 
            attn_windows, depth_token, seg_token = self.attn(x_windows, mask=attn_mask, x_ref=x_ref, depth_token=depth_token, seg_token=seg_token)  # nW*B, window_size*window_size, C
        else:
            attn_windows, _, _ = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        
        

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
    
        if self.args.with_line_depth and point_token is not None:
            point_token = self.pointTokenAttn(x, point_token)

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
            
            depth_token = depth_token_shortcut.reshape(B, H, W, -1) + self.drop_path_class(depth_token)
            depth_token = depth_token + self.drop_path_class(self.mlp_depth(self.norm_depth2(depth_token)))
            seg_token = seg_token_shortcut.reshape(B, H, W, -1) + self.drop_path_class(seg_token)
            seg_token = seg_token + self.drop_path_class(self.mlp_seg(self.norm_seg2(seg_token)))
            
            depth_token = depth_token.view(B, H * W, tC)       
            seg_token = seg_token.view(B, H * W, tC)
            if self.token_fuse and self.args.with_line:
                r_seg_token = seg_token.view(B, H, W, tC)
                r_depth_token = depth_token.view(B, H, W, tC)
                depth_token = self.token_relation(x, r_seg_token.permute(0,3,1,2), r_depth_token.permute(0,3,1,2), 
                        ref_roll_coors, token_pos, with_pos=True)
                depth_token = depth_token.permute(0, 2, 3, 1).view(B, H * W, tC)

        return x, depth_token, seg_token, point_token


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
        size_ratio (float): down size ratio for this layer.
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
                 size_ratio=0.25,
                 group_attention_blocks=None,
                 pre_point_double=False,
                 num_points=None,
                 class_pred=False,
                 pre_class_pred=False,
                 with_point_token=False,
                 args=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.args = args

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
                size_ratio=size_ratio,
                group_attention=group_attention_blocks[i] if group_attention_blocks is not None else False,
                num_points=num_points,
                with_point_token=with_point_token,
                args=args)
            for i in range(depth)])

        self.pre_point_double = False
        if pre_point_double:
            self.point_double = OffsetGeneration(dim, args.num_ref, args=args)
            self.pre_point_double = True

        self.pre_class_pred = pre_class_pred
        if pre_class_pred:
            self.pre_depth_pred = nn.Sequential(nn.Linear(dim+args.class_token_dim, args.class_token_dim),
                                            nn.Linear(args.class_token_dim, 1), 
                                            nn.Sigmoid())

        self.class_pred = class_pred
        if class_pred:
            self.nonlocal_pred = NonLocalPlannarGuidance(backbone_dim=dim, num_points=num_points, pre_pred=pre_class_pred, args=args)
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W, ref_coors, ref_pos, depth_token=None, seg_token=None, point_token=None, token_pos=None, depth_pred=None):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        if self.pre_point_double:
            ref_coors = self.point_double(x, seg_token, depth_token, ref_coors, token_pos, size=(H, W))

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
                x, depth_token, seg_token, point_token = blk(x, attn_mask, ref_coors=ref_coors, ref_pos=ref_pos, 
                                depth_token=depth_token, seg_token=seg_token, point_token=point_token, token_pos=token_pos)
        if self.class_pred:
            if self.pre_class_pred:
                depth_pred = self.pre_depth_pred(torch.cat([x, depth_token], dim=2))
                depth_pred = depth_pred.reshape(depth_pred.shape[0], H, W, -1).permute(0, 3, 1, 2)

            r_x = x.view(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2)
            r_seg_token = seg_token.view(-1, H, W, seg_token.shape[-1]).permute(0, 3, 1, 2)
            r_depth_token = depth_token.view(-1, H, W, depth_token.shape[-1]).permute(0, 3, 1, 2)

            depth_pred, refer_global_attn = self.nonlocal_pred(r_x, r_seg_token, r_depth_token, ref_coors, token_pos, depth_pred=depth_pred, with_pos=True)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww, depth_token, seg_token, point_token, ref_coors, depth_pred
        else:
            return x, H, W, x, H, W, depth_token, seg_token, point_token, ref_coors, depth_pred


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
        if args.with_line_depth:
            self.point_depth_token = nn.Parameter(torch.zeros(1, args.num_ref * 2, args.class_token_dim))
            self.init_token = nn.Parameter(torch.zeros(1, 32, 32, args.class_token_dim))
        else:
            self.depth_token = nn.Parameter(torch.zeros(1, 1, args.class_token_dim))
        self.seg_token = nn.Parameter(torch.zeros(1, 1, args.class_token_dim))

        self.pos_enc = PositionEmbeddingSine(num_pos_feats=args.dense_trans_dim // 2)

        if args.with_line:
            self.dense_transformer = BasicLayer(dim=args.dense_trans_dim, depth=args.dense_trans_layers[0], num_heads=args.dense_trans_heads, \
                window_attn=WindowAttention, window_size=7, mlp_ratio=2, size_ratio=1/32, with_point_token=False, args=args)
        else:
            self.dense_transformer = BasicLayer(dim=args.dense_trans_dim, depth=args.dense_trans_layers[0], num_heads=args.dense_trans_heads, \
                window_attn=OriginWindowAttention, window_size=7, mlp_ratio=2, size_ratio=1/32, args=args)

        line_pnt_num = 3 if args.with_dense_center else 2
        #1/16
        if args.with_line_depth:
            self.gpg1 = Global2PointGraph(1, args.num_ref * 2, args)
        self.pos_cls1 = PositionEmbeddingSine(num_pos_feats=args.dense_trans_dim // 4)
        self.pos_cls1_token = PositionEmbeddingSine(num_pos_feats=args.class_token_dim // 2)
        self.proj_class1 = nn.Linear(args.dense_trans_dim, args.dense_trans_dim // 2)
        self.proj_backbn1  = ConvA(feat_out_channels[-2], out_features=args.dense_trans_dim // 2)
        num_points = args.num_ref * line_pnt_num if not args.points_double_layers[0] else args.num_ref * line_pnt_num * 2
        self.class_transformer1 = BasicLayer(dim=args.dense_trans_dim // 2, depth=args.class_trans_layers[0], num_heads=args.dense_trans_heads, \
            window_attn=WindowClassAttention, window_size=7, mlp_ratio=2, size_ratio=1/16, 
            pre_point_double=args.points_double_layers[0], 
            num_points=num_points, 
            pre_class_pred=True,
            group_attention_blocks=args.group_attention_layers[0],
            # class_pred=args.depth_pred_layers[0],
            args=args)
        # if not args.with_line:
        if not args.depth_pred_layers[0]:
            self.depth_pred16 = nn.Sequential(nn.Linear((args.dense_trans_dim // 2)+args.class_token_dim, args.class_token_dim),
                                            nn.Linear(args.class_token_dim, 1), nn.Sigmoid())
        else:
            # backbone_dim=128, num_points=50, pre_pred=False, args=None):
            self.depth_pred16_pg = NonLocalPlannarGuidance(backbone_dim=args.dense_trans_dim // 2, num_points=num_points, pre_pred=True, num_levels=2, args=args)

        #1/8
        if args.with_line_depth:
            self.gpg2 = Global2PointGraph(2, args.num_ref * 2, args)
        else:
            # self.old_depth_token_proj8 = ConvA(args.class_token_dim, args.class_token_dim)
            # self.old_seg_token_proj8 = ConvA(args.class_token_dim, args.class_token_dim)
            self.old_depth_token_proj8 = MlpNorm(args.class_token_dim, hidden_features=args.class_token_dim * 2, norm_layer=nn.LayerNorm)
            self.old_seg_token_proj8 = MlpNorm(args.class_token_dim, hidden_features=args.class_token_dim * 2, norm_layer=nn.LayerNorm)
        self.pos_cls2 = PositionEmbeddingSine(num_pos_feats=args.dense_trans_dim // 8)
        self.pos_cls2_token = PositionEmbeddingSine(num_pos_feats=args.class_token_dim // 2)
        self.proj_class2 = nn.Linear(args.dense_trans_dim // 2, args.dense_trans_dim // 4)
        self.proj_backbn2  = ConvA(feat_out_channels[-3], out_features=args.dense_trans_dim // 4)
        num_points = num_points * 2 if args.points_double_layers[1] else num_points
        self.class_transformer2 = BasicLayer(dim=args.dense_trans_dim // 4, depth=args.class_trans_layers[1], num_heads=args.dense_trans_heads, \
            window_attn=WindowClassAttention, window_size=7, mlp_ratio=2, size_ratio=1/8, 
            pre_point_double=args.points_double_layers[1],
            num_points=num_points,
            group_attention_blocks=args.group_attention_layers[1],
            # class_pred=args.depth_pred_layers[1],
            args=args)
        # if not args.with_line:
        if not args.depth_pred_layers[1]:
            self.depth_pred8 = nn.Sequential(nn.Linear((args.dense_trans_dim // 4)+args.class_token_dim, args.class_token_dim),
                                            nn.Linear(args.class_token_dim, 1), nn.Sigmoid())
        else:
             self.depth_pred8_pg = NonLocalPlannarGuidance(backbone_dim=args.dense_trans_dim // 4, num_points=num_points, num_levels=3, args=args)

        #1/4
        if args.with_line_depth:
            self.gpg3 = Global2PointGraph(4, args.num_ref * 2, args)
        else:
            # self.old_depth_token_proj4 = ConvA(args.class_token_dim, args.class_token_dim)
            # self.old_seg_token_proj4 = ConvA(args.class_token_dim, args.class_token_dim)
            self.old_depth_token_proj4 = MlpNorm(args.class_token_dim, hidden_features=args.class_token_dim * 2, norm_layer=nn.LayerNorm)
            self.old_seg_token_proj4 = MlpNorm(args.class_token_dim, hidden_features=args.class_token_dim * 2, norm_layer=nn.LayerNorm)

        num_plane3 = args.num_ref * (2 ** (int(args.points_double_layers[0]) + int(args.points_double_layers[1]) + int(args.points_double_layers[1])))
        self.pos_cls3 = PositionEmbeddingSine(num_pos_feats=args.dense_trans_dim // 16)
        self.pos_cls3_token = PositionEmbeddingSine(num_pos_feats=args.class_token_dim // 2)
        self.proj_class3 = nn.Linear(args.dense_trans_dim // 4, args.dense_trans_dim // 8)
        self.proj_backbn3  = ConvA(feat_out_channels[-4], out_features=args.dense_trans_dim // 8)
        num_points = num_points * 2 if args.points_double_layers[2] else num_points
        self.class_transformer3 = BasicLayer(dim=args.dense_trans_dim // 8, depth=args.class_trans_layers[2], num_heads=args.dense_trans_heads, \
            window_attn=WindowClassAttention, window_size=7, mlp_ratio=2, size_ratio=1/4, 
            pre_point_double=args.points_double_layers[2], 
            num_points=num_points, 
            group_attention_blocks=args.group_attention_layers[2],
            # class_pred=args.depth_pred_layers[2],
            args=args)

        # self.depth_proj16 = nn.Linear((args.dense_trans_dim // 2)+args.class_token_dim, args.class_token_dim)
        # self.depth_proj8 = nn.Linear((args.dense_trans_dim // 4)+args.class_token_dim, args.class_token_dim)
        # self.depth_porj4 = nn.Linear((args.dense_trans_dim // 8)+args.class_token_dim, args.class_token_dim)
        # self.depth_upsample16to8 = ConvA(args.class_token_dim, args.class_token_dim)
        # self.depth_upsample8to4 = ConvA(args.class_token_dim, args.class_token_dim)
        # self.depth_pred16 = nn.Sequential(nn.Linear(args.class_token_dim, 1), nn.Sigmoid())
        # self.depth_pred8 = nn.Sequential(nn.Linear(args.class_token_dim, 1), nn.Sigmoid())

        # if not args.with_line:
        if not args.depth_pred_layers[2]:
            self.depth_pred4 = nn.Sequential(nn.Linear((args.dense_trans_dim // 8)+args.class_token_dim, args.class_token_dim),
                                            nn.Linear(args.class_token_dim, 1), nn.Sigmoid())
        else:
            self.depth_pred4_pg = NonLocalPlannarGuidance(backbone_dim=args.dense_trans_dim // 8, num_points=num_points, args=args)

        self.num_ref = args.num_ref
        self.args = args
        trunc_normal_(self.seg_token, std=.02)
        if args.with_line_depth:
            trunc_normal_(self.point_depth_token, std=.02)
            trunc_normal_(self.init_token, std=.02)
        else:
            trunc_normal_(self.depth_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
    
    def forward(self, nested_top_mat, features, sample_points, sample_points_scores, layers_size=None, with_pos=True, 
                reflc_points=None, reflc_mat=None, input_size=None, input_images=None):
        top_mat, mask = nested_top_mat.decompose()
        B, C, H, W = top_mat.shape
        # sample_points_scores torch.Size([1, 100, 2])
        # sample_points torch.Size([1, 100, 6])

        # lines = sample_points[:, :, :4]
        # centers = sample_points[:, :, 4:]
        # choosen_points_bycenter = sample_by_centers(centers, lines, sample_points_scores, input_size[0], input_size[1],
        #     shortest_ratio=self.args.shortest_ratio, num_clusters=self.args.num_clusters, top_num=6, sample_line_num=self.args.num_ref)
        # choosen_points_bycenter = choosen_points_bycenter.reshape(B, -1, 2, 2)
        # choosen_points_bycenter = choosen_points_bycenter * 2 - 1.0

        if self.args.with_line and sample_points_scores is not None:
            t_values, t_ids = torch.topk(sample_points_scores[:, :, 0], self.num_ref, dim=-1)
            choosen_points = torch.stack([sample_points[i][t_ids[i]] for i in range(B)])
            choosen_points = choosen_points.reshape(B, self.num_ref, -1, 2)
            # if reflc_points is not None:
            #     rhints = reflc_points.type(choosen_points.type()).cuda(sample_points.device)
            #     choosen_points = torch.cat([choosen_points.reshape(B, -1, 2), rhints], dim=1)
            #     choosen_points = choosen_points.unsqueeze(2)

            # norm to (-1, 1)
            choosen_points_origin = choosen_points * 2 - 1.0
        else:
            choosen_points_origin = None
        if self.args.with_line and not self.args.with_dense_center:
            choosen_points_origin = choosen_points_origin[:, :, :2]
        
        pos_emb = self.pos_enc(nested_top_mat) if with_pos else None  
        if self.args.with_line_depth:
            point_token = self.point_depth_token.expand(B, -1, -1)
        else:
            point_token = None
        d_enc_out, H, W, x, Wh, Ww, _, _, point_token, choosen_points, _ = self.dense_transformer(top_mat.flatten(2).permute(0, 2, 1), H, W, choosen_points_origin, pos_emb, point_token=point_token)
        # show_smapled_lines((choosen_points.detach().cpu() + 1)/2, input_images.tensors.cpu(), with_center=True, title='sampled'+str(input_size), stop_show=True)

        # d_enc_out (B, N, C)
        dense_out = d_enc_out.permute(0, 2, 1).reshape(-1, C, H, W)
        H_cls1, W_cls1 = layers_size[0]
        dense_out_up = F.interpolate(dense_out, size=(H_cls1, W_cls1), mode='nearest')
        cls_enc_inp1 = self.proj_class1(dense_out_up.flatten(2).permute(0, 2, 1))
        backbn_mat1, mask1 = features[self.args.layer1_num - 1].decompose()
        cls_enc_inp1 = cls_enc_inp1 + self.proj_backbn1(backbn_mat1).flatten(2).permute(0, 2, 1)
        pos_emb_cls1 = self.pos_cls1(NestedTensor(dense_out_up, mask1))
        pos_tok_cls1 = self.pos_cls1_token(NestedTensor(dense_out_up, mask1))

        #####1/16
        if self.args.with_line_depth:
            depth_token = self.gpg1(self.init_token, point_token, H_cls1, W_cls1, is_init=True)
        else:
            depth_token = self.depth_token.expand(B, H_cls1 * W_cls1, -1)
        seg_token = self.seg_token.expand(B, H_cls1 * W_cls1, -1)
        if self.args.points_inline_sample_layers[0]:
            choosen_points = sample_mid_seg(choosen_points, H_cls1, W_cls1, sample_num_seg=2)
        cls_enc_out1, H1, W1, x1, Wh1, Ww1, depth_token, seg_token, _, choosen_points, depth_pred1 = self.class_transformer1(cls_enc_inp1, H_cls1, W_cls1, choosen_points, pos_emb_cls1, 
            depth_token=depth_token, seg_token=seg_token, token_pos=pos_tok_cls1)
        # show_smapled_lines(choosen_points.cpu(), input_images.tensors.cpu())

        # if not self.args.with_line:
        if not self.args.depth_pred_layers[0]:
            df1 = torch.cat([cls_enc_out1, depth_token], dim=-1)
            depth_pred1 = self.depth_pred16(df1).permute(0, 2, 1).reshape(B, -1, H1, W1)
        else:
            depth_pred1, _ = self.depth_pred16_pg(cls_enc_out1.permute(0, 2, 1).reshape(B, -1, H_cls1, W_cls1), 
                                    seg_token.permute(0, 2, 1).reshape(B, -1, H_cls1, W_cls1), 
                                    depth_token.permute(0, 2, 1).reshape(B, -1, H_cls1, W_cls1), 
                                    choosen_points, pos_tok_cls1)
        
        # df16 = self.depth_proj16(df1).reshape(B, -1, H1, W1)
        # df16_8 = self.depth_upsample16to8(F.interpolate(df16, size=layers_size[1], mode='nearest'))
        # df16_8_4 = self.depth_upsample8to4(F.interpolate(df16_8, size=layers_size[2], mode='nearest'))
        # depth_pred1 = self.depth_pred16(df16_8_4.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, -1, *layers_size[2])

        #####1/8
        C1 = cls_enc_out1.shape[-1]
        # cls_enc_out (B, N, C1)
        cls_enc_out1 = cls_enc_out1.permute(0, 2, 1).reshape(-1, C1, H1, W1)
        H_cls2, W_cls2 = layers_size[1]
        cls_enc_out1_up = F.interpolate(cls_enc_out1, size=(H_cls2, W_cls2), mode='nearest')
        cls_enc_inp2 = self.proj_class2(cls_enc_out1_up.flatten(2).permute(0, 2, 1))
        backbn_mat2, mask2 = features[self.args.layer1_num - 2].decompose()
        cls_enc_inp2 = cls_enc_inp2 + self.proj_backbn2(backbn_mat2).flatten(2).permute(0, 2, 1)
        pos_emb_cls2 = self.pos_cls2(NestedTensor(cls_enc_out1_up, mask2))
        pos_tok_cls2 = self.pos_cls2_token(NestedTensor(cls_enc_out1_up, mask2))
        if self.args.with_line_depth:
            depth_token = self.gpg2(depth_token.reshape(B, H_cls1, W_cls1, -1), point_token, H_cls2, W_cls2)
        else:  
            depth_token = depth_token.reshape(B, H_cls1, W_cls1, -1).permute(0, 3, 1, 2)
            d_token_up = F.interpolate(depth_token, size=(H_cls2, W_cls2), mode='nearest')
            depth_token = self.old_depth_token_proj8(d_token_up.flatten(2).permute(0, 2, 1))

            seg_token = seg_token.reshape(B, H_cls1, W_cls1, -1).permute(0, 3, 1, 2)
            s_token_up = F.interpolate(seg_token, size=(H_cls2, W_cls2), mode='nearest')
            seg_token = self.old_seg_token_proj8(s_token_up.flatten(2).permute(0, 2, 1))

        cls_enc_out2, H2, W2, x2, Wh2, Ww2, depth_token, seg_token, _, choosen_points, depth_pred2 = self.class_transformer2(cls_enc_inp2, H_cls2, W_cls2, choosen_points, pos_emb_cls2, 
            depth_token=depth_token, seg_token=seg_token, token_pos=pos_tok_cls2, depth_pred=depth_pred1)
        # show_sampled_points(choosen_points.cpu(), input_images.tensors.cpu())

        # if not self.args.with_line:
        if not self.args.depth_pred_layers[1]:
            df2 = torch.cat([cls_enc_out2, depth_token], dim=-1)
            depth_pred2 = self.depth_pred8(df2).permute(0, 2, 1).reshape(B, -1, H2, W2)
        else:
            depth_pred2, _ = self.depth_pred8_pg(cls_enc_out2.permute(0, 2, 1).reshape(B, -1, H_cls2, W_cls2),
                                seg_token.permute(0, 2, 1).reshape(B, -1, H_cls2, W_cls2), 
                                depth_token.permute(0, 2, 1).reshape(B, -1, H_cls2, W_cls2),
                                choosen_points, pos_tok_cls2, depth_pred=depth_pred1)
        # df8 = self.depth_proj8(df2).reshape(B, -1, H2, W2)
        # df8_4 = self.depth_upsample8to4(F.interpolate(df8, size=layers_size[2], mode='nearest'))
        # depth_pred2 = self.depth_pred16(df8_4.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, -1, *layers_size[2])

        #####1/4
        C2 = cls_enc_out2.shape[-1]
        # cls_enc_out2 (B, N, C2)
        cls_enc_out2 = cls_enc_out2.permute(0, 2, 1).reshape(-1, C2, H2, W2)
        H_cls3, W_cls3 = layers_size[2]
        cls_enc_out2_up = F.interpolate(cls_enc_out2, size=(H_cls3, W_cls3), mode='nearest')
        cls_enc_inp3 = self.proj_class3(cls_enc_out2_up.flatten(2).permute(0, 2, 1))
        backbn_mat3, mask3 = features[self.args.layer1_num - 3].decompose()
        cls_enc_inp3 = cls_enc_inp3 + self.proj_backbn3(backbn_mat3).flatten(2).permute(0, 2, 1)
        pos_emb_cls3 = self.pos_cls3(NestedTensor(cls_enc_out2_up, mask3))
        pos_tok_cls3 = self.pos_cls3_token(NestedTensor(cls_enc_out2_up, mask3))
        if self.args.with_line_depth:
            depth_token = self.gpg3(depth_token.reshape(B, H_cls2, W_cls2, -1), point_token, H_cls3, W_cls3)
        else:
            depth_token = depth_token.reshape(B, H_cls2, W_cls2, -1).permute(0, 3, 1, 2)
            d_token_up = F.interpolate(depth_token, size=(H_cls3, W_cls3), mode='nearest')
            depth_token = self.old_depth_token_proj4(d_token_up.flatten(2).permute(0, 2, 1))

            seg_token = seg_token.reshape(B, H_cls2, W_cls2, -1).permute(0, 3, 1, 2)
            s_token_up = F.interpolate(seg_token, size=(H_cls3, W_cls3), mode='nearest')
            seg_token = self.old_seg_token_proj4(s_token_up.flatten(2).permute(0, 2, 1))
        cls_enc_out3, H3, W3, x3, Wh3, Ww3, depth_token, seg_token, _, choosen_points, depth_pred3 = self.class_transformer3(cls_enc_inp3, H_cls3, W_cls3, choosen_points, pos_emb_cls3, 
            depth_token=depth_token, seg_token=seg_token, token_pos=pos_tok_cls3, depth_pred=depth_pred2)
        # show_sampled_points(choosen_points.cpu(), input_images.tensors.cpu())
        # if not self.args.with_line:
        if not self.args.depth_pred_layers[2]:
            df3 = torch.cat([cls_enc_out3, depth_token], dim=-1)
            depth_pred3 = self.depth_pred4(df3).permute(0, 2, 1).reshape(B, -1, H3, W3)
        else:
            depth_pred3, _ = self.depth_pred4_pg(cls_enc_out3.permute(0, 2, 1).reshape(B, -1, H_cls3, W_cls3),
                                seg_token.permute(0, 2, 1).reshape(B, -1, H_cls3, W_cls3), 
                                depth_token.permute(0, 2, 1).reshape(B, -1, H_cls3, W_cls3), 
                                choosen_points, pos_tok_cls3, depth_pred=depth_pred2)
        
        C3 = cls_enc_out3.shape[-1]
        dense_out_list = [dense_out, #1/32
                        cls_enc_out1, #1/16
                        cls_enc_out2, #1/8
                        cls_enc_out3.permute(0, 2, 1).reshape(-1, C3, H3, W3),] #1/4
        depth_token = depth_token.permute(0, 2, 1).reshape(-1, C3, H3, W3) #1/4
        seg_token = seg_token.permute(0, 2, 1).reshape(-1, C3, H3, W3) #1/4

        # pdepth = self.points_depth_pred(points_depth['depth']).squeeze(3)
        # points_depth['depth'] = pdepth.sigmoid() * self.args.max_depth

        # if sum(self.args.points_double_layers) > 0:
        #     st, gen_points = 0, []
        #     for i in range(len(self.args.points_double_layers)):
        #         st = i * self.args.num_ref
        #         ed = (i + 1) * self.args.num_ref if i < len(self.args.points_double_layers) - 1 else None
        #         gen_points.append(choosen_points[:, st:ed])
        # else:
        #     gen_points = None
        gen_points = None
        
        depth_preds_16_8_4 = [depth_pred1, depth_pred2, depth_pred3]
        return dense_out_list, depth_token, seg_token, gen_points, depth_preds_16_8_4

def build_dense_transformer(args):
    # b = BasicLayer(dim=args.dense_trans_dim, depth=args.dense_trans_layers[0], num_heads=args.dense_trans_heads, \
    #         window_size=7, mlp_ratio=2, num_ref=args.num_ref * 3)
    RT = ReferTransformer(args, [256, 512, 1024, 512])
    return RT
