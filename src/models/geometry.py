from matplotlib.pyplot import polar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import init

import numpy as np

# from models.sne_model import SNE
import pyrealsense2 as rs
from models.bts_token import atrous_conv

def get_cam_param():
    camParam = torch.tensor([[6.360779e+02, 0.000000e+00, 6.348217e+02],
                            [0.000000e+00, 6.352265e+02, 3.570233e+02],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=torch.float32)
    return camParam

def read_camera_intrin(intrin_dict):
    intri = rs.intrinsics()
    intri.height = intrin_dict['height']
    intri.width = intrin_dict['width']
    intri.coeffs = intrin_dict['coeffs']
    intri.fx = intrin_dict['fx']
    intri.fy = intrin_dict['fy']
    intri.ppx = intrin_dict['ppx']
    intri.ppy = intrin_dict['ppy']
    # rs_dis = rs.distortion(2)
    assert intrin_dict['model'] == 'distortion.inverse_brown_conrady', intrin_dict['model']
    intri.model = rs.distortion.inverse_brown_conrady
    return intri, intrin_dict['depth_scale']

# pixel depth is in meter unit
def deproject_to_points(pixel_points, pixel_depths, height, width, depth_scale=0.001):
    camera_intrin = {"depth_scale": 0.0010000000474974513, "height": 720, "width": 1280, 
            "coeffs": [-0.056396592408418655, 0.06423918902873993, -0.00023513064661528915, -3.168615512549877e-05, -0.02033711038529873], 
            "fx": 636.0779418945312, "fy": 635.2265014648438, 
            "ppx": 634.8217163085938, "ppy": 357.0233154296875, "model": "distortion.inverse_brown_conrady"}
    camera_intrin['height'] = height
    camera_intrin['width'] = width
    intrin_obj = read_camera_intrin(camera_intrin)
    world_coors = []
    for pixel_coor, pixel_dpth in zip(pixel_points, pixel_depths):
        coor3d = rs.rs2_deproject_pixel_to_point(intrin_obj, pixel_coor, pixel_dpth)
        world_coors.append(coor3d)
    return world_coors

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

class TokenRelation(nn.Module):
    def __init__(self, seg_dim, num_ref_points, num_coarse_points, size_ratio=0.25, args=None):
        super().__init__()
        self.seg_dim  = seg_dim
        self.num_ref_points = num_ref_points
        self.num_coarse_points = num_coarse_points
        self.seg_proj = Mlp(seg_dim, seg_dim)
        self.coars_depth_proj = Mlp(num_coarse_points, num_coarse_points)
        self.num_line = args.num_ref if args is not None else 50

        self.pool_size1 = int(args.pooling_base_size * size_ratio) ** 2
        self.conv_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((self.pool_size1, None)),
            ConvA(in_features=seg_dim, out_features=seg_dim),
            ConvA(in_features=seg_dim, out_features=seg_dim),
            nn.MaxPool2d(kernel_size=(3,1), padding=(1,0), stride=(2,1)),
            ConvA(in_features=seg_dim, out_features=seg_dim),
            nn.MaxPool2d(kernel_size=(3,1), padding=(1,0), stride=(2,1)),
            ConvA(in_features=seg_dim, out_features=seg_dim),
            nn.MaxPool2d(kernel_size=(3,1), padding=(1,0), stride=(2,1)),
            ConvA(in_features=seg_dim, out_features=seg_dim),
        )
        self.aspp1 = atrous_conv(in_channels=seg_dim, out_channels=seg_dim, dilation=6)
        self.aspp2 = atrous_conv(in_channels=seg_dim, out_channels=seg_dim, dilation=12)
        
        self.relation_module = nn.ModuleList()
        depth = 3
        in_features = self.pool_size1 // 2 // 2 // 2
        for i in range(depth):
            if (in_features // 2) > num_coarse_points:
                hidden_features = in_features // 2
                if (hidden_features // 2) >= num_coarse_points:
                    out_features = hidden_features // 2
                else:
                    out_features = num_coarse_points
            else:
                hidden_features = out_features = num_coarse_points

            self.relation_module.append(
                nn.ModuleDict(
                    {
                        'spatial':Mlp(in_features=in_features, hidden_features=hidden_features, out_features=out_features),
                        'channel':ConvA(in_features=seg_dim, out_features=seg_dim)
                    }
                )
            )
            in_features = out_features
        # self.rela_act = nn.Softmax(dim=-1)

        #
        self.geome_conv = nn.Sequential(
            nn.Conv2d(in_channels=seg_dim, out_channels=seg_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=seg_dim, out_channels=seg_dim, kernel_size=(6, 1), stride=(6, 1)),
        )
        self.gome_depth_proj = nn.Sequential(
            Olp(in_features=seg_dim * num_coarse_points, out_features=(seg_dim  * num_coarse_points) // 8),
            Mlp(in_features=(seg_dim  * num_coarse_points) // 8, hidden_features=(seg_dim  * num_coarse_points) // 32, out_features=1)
        )
        self.old_depth_proj = Mlp(in_features=num_coarse_points, out_features=num_coarse_points)
    
    # seg_token/depth_token (B, C, H, W), refer_coords(line points) (B, num_line, 2, 2), coarse_coords (B, num_line, 1, 2)
    # global_pos (B, C, H, W), depth_pred (B, 1, H, W)
    def forward(self, seg_token, depth_token, refer_coords, coarse_coords, global_pos, old_points_depth=None, with_pos=True):
        H, W = depth_token.shape[-2:]
        samp_depth = F.grid_sample(depth_token, refer_coords, mode='nearest') #(B, C, n_samp_point, 1)
        coarse_depth = F.grid_sample(depth_token, coarse_coords, mode='nearest') #(B, C, n_coarse_pnt, 1)
        # print('coarse_depth', coarse_depth.shape)
        if old_points_depth is not None:
            print('coarse_depth', coarse_depth.shape, 'old_points_depth', old_points_depth.shape)
        else:
            print('old_points_depth is none')


        if with_pos:
            samp_pos = F.grid_sample(global_pos, refer_coords, mode='nearest')
            samp_depth = samp_depth + samp_pos
            refine_pos = F.grid_sample(global_pos, coarse_coords, mode='nearest')
            coarse_depth = coarse_depth + refine_pos
        # (B, C, num_line_or_triangle, 2_or_3) -> (B, C, n_samp_point) -> (B, C, 1, n_samp_point)
        samp_depth = samp_depth.flatten(2).unsqueeze(2)
        seg_token = self.seg_proj(seg_token.flatten(2).permute(0, 2, 1)) # (B, H*W, C)
        seg_token = seg_token.permute(0, 2, 1).unsqueeze(-1).contiguous() # (B, C, H*W, 1)
        samp_rel = seg_token @ samp_depth # (B, C, H*W, n_samp_point)
         
        B, C, N, N_p = samp_rel.shape
        rela_pool = self.conv_pool(samp_rel)
        rela = self.aspp1(rela_pool) + self.aspp2(rela_pool)
        for i, m in enumerate(self.relation_module):
            rela = rela.permute(0, 1, 3, 2).reshape(B, C * N_p, -1).contiguous() # (B, C*n_samp_point, n_coarse_pnt')
            rela = m['spatial'](rela)
            _, _, rH = rela.shape
            rela = rela.reshape(B, C, N_p, -1).permute(0, 1, 3, 2).contiguous()  # (B, C, n_coarse_pnt', n_samp_point)
            rela = m['channel'](rela)
        # rela = self.rela_act(rela)
        # distance between refer points and coarse points
        _, num_line, _, _ = refer_coords.shape
        refer_coords_e = refer_coords.reshape(B, -1, 2).unsqueeze(2)
        coarse_coords_e = coarse_coords.reshape(B, -1, 2).unsqueeze(1)
        dist = torch.sqrt(torch.sum(torch.square(refer_coords_e - coarse_coords_e), dim=-1))

        refr_p1 = refer_coords[:, :, 0] #(B, num_line, 2)
        refr_p2 = refer_coords[:, :, 1] #(B, num_line, 2)
        refr_p1_x = refr_p1[:, :, 0] #(B, num_line)
        refr_p1_y = refr_p1[:, :, 1]
        refr_p2_x = refr_p2[:, :, 0]
        refr_p2_y = refr_p2[:, :, 1]
        coars_x = coarse_coords.squeeze(2)[:, :, 0] #(B, num_line)
        coars_y = coarse_coords.squeeze(2)[:, :, 1]
        # sides_info = (coars_x - refr_p1_x) * (refr_p2_y - refr_p1_y) - (coars_y - refr_p1_y) * (refr_p2_x - refr_p1_x)
        sides_info = (coars_x[:,:,None] - refr_p1_x[:,None]) * (refr_p2_y[:,:,None] - refr_p1_y[:,None]) \
                    - (coars_y[:,:,None] - refr_p1_y[:,None]) * (refr_p2_x[:,:,None] - refr_p1_x[:,None])
        geome_info = torch.cat((dist.reshape(B, num_line, 2, -1), sides_info.reshape(B, num_line, 1, -1)), dim=2) # (B, num_line, 3, num_line)

        geom_e = geome_info[:, None].expand(-1, C, -1, -1, -1) # (B, C, num_line, 3, num_line)
        rela_e = rela.permute(0, 1, 3, 2).reshape(B, C, num_line, 2, num_line) # (B, C, num_line, 2, num_line)
        coars_depth_prj = self.coars_depth_proj(coarse_depth.flatten(2))
        coars_depth_prj_e = coars_depth_prj[:, :, None].expand(-1, -1, num_line, -1).unsqueeze(3) # (B, C, num_line, 1, num_line)
        rela_geom = torch.cat((rela_e, geom_e, coars_depth_prj_e), dim=3) # (B, C, num_line, 6, num_line)
        rela_geom = rela_geom.reshape(B, C, num_line * 6, num_line).contiguous()
        rela_g = self.geome_conv(rela_geom)

        rela_g = rela_g.permute(0, 3, 1, 2).flatten(2)
        refined_depth = self.gome_depth_proj(rela_g)
        return refined_depth
        # coarse_coords_src = coarse_coords.squeeze(2)
        # coarse_coords_src[:, :, 0] = coarse_coords_src[:, :, 0] * W
        # coarse_coords_src[:, :, 1] = coarse_coords_src[:, :, 1] * H
        # coarse_coords_src = torch.ceil(coarse_coords_src)
        # coarse_coords_src[:, :, 0] = torch.clamp(coarse_coords_src[:, :, 0], min=0, max=W-1)
        # coarse_coords_src[:, :, 1] = torch.clamp(coarse_coords_src[:, :, 1], min=0, max=H-1)
        # coarse_coords_src = coarse_coords_src.type(torch.long)
        # depth_token[:, :, coarse_coords_src[:, :, 1], coarse_coords_src[:, :, 0]] = refined_depth.squeeze(-1)
        # return depth_token


class TokenFuse(nn.Module):
    def __init__(self, dim, height, width, args=None):
        super().__init__()
        self.depth_proj = Mlp(in_features=dim, hidden_features=dim // 2, out_features=dim // 4)
        self.depth_pred = torch.nn.Sequential(nn.Conv2d(dim // 4, 1, 3, 1, 1, bias=False),
                                             nn.Sigmoid())

        # def __init__(self, seg_dim, seg_height, seg_width, num_ref_points, num_coarse_points, args=None):
        self.line_center_rela = TokenRelation(args.class_token_dim, height, width, args.num_ref * 2, args.num_ref, args=args)
        self.args = args
    
    def forward(self, depth_token, seg_token, sample_coors, pos_emb, with_pos=True):
        #token [B, H, W, C], sample_coors [B, num_line, 3, 2]
        B, H, W, C = depth_token.shape



if __name__ == '__main__':
    import sys
    sys.path.append('/home/ly/workspace/git/segm/line_segm/letr-depth-2')
    import argparse
    from src.args import get_args_parser

    # def __init__(self, seg_dim, seg_height, seg_width, num_ref_points, args=None):
    parser = argparse.ArgumentParser('LETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    TR = TokenRelation(seg_dim=64, seg_height=30, seg_width=40, num_ref_points=100, num_coarse_points=50, size_ratio=1/16, args=args)
    # def forward(self, seg_token, depth_token, refer_coords, global_pos, with_pos=True):
    seg_tok = torch.randn(2, 64, 30, 40)
    dth_tok = torch.randn(2, 64, 30, 40)
    samp_coors = (torch.rand(2, 50, 2, 2) * 2) - 1
    coarse_coords = (torch.rand(2, 50, 1, 2) * 2) - 1
    global_pos = torch.randn(2, 64, 30, 40)
    depth_pred = torch.rand(2, 1, 30, 40) * 10.0
    print(TR)
    a = TR(seg_tok, dth_tok, refer_coords=samp_coors, coarse_coords=coarse_coords, global_pos=global_pos)