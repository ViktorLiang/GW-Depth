import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import init

from models.bts_token import atrous_conv
from util.commons import batch_isin_triangle

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

    def __init__(self, in_features, out_features=None, kernel_size=3, padding=1, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or 1
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class TokenGlobalRelation(nn.Module):
    def __init__(self, seg_dim, num_ref_points, num_coarse_points, size_ratio=0.25, args=None):
        super().__init__()
        self.seg_dim  = seg_dim
        self.num_ref_points = num_ref_points
        self.num_coarse_points = num_coarse_points
        self.num_plane = args.num_ref or num_ref_points // 3
        self.args = args

        self.seg_proj = Mlp(seg_dim, seg_dim)
        self.depth_proj = Mlp(seg_dim, out_features=seg_dim)
        self.kv_refer_depth = Mlp(seg_dim, out_features=seg_dim * 2)
        self.q_seg_geometry = Mlp(seg_dim + self.num_plane * 4, out_features=seg_dim)

        self.size_ratio = size_ratio
        # if size_ratio > args.adaptive_min_ratio:
        #     self.pre_pool = nn.Sequential(
        #                     nn.AvgPool2d(kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
        #                     ConvA(in_features=seg_dim, out_features=seg_dim)
        #                 )
        # else:
        #     self.pre_pool = nn.Identity()

        self.seg_dpth_softmax = nn.Softmax(dim=-1)
        self.fused_depth_proj = nn.Linear(seg_dim, seg_dim)
    
    # seg_token/depth_token (B, C, H, W), refer_coords(plane points) (B, num_plane, 3, 2), coarse_coords (B, num_line, 1, 2)
    # global_pos (B, C, H, W), depth_pred (B, 1, H, W)
    def forward(self, seg_token, depth_token, refer_coords, coarse_coords, global_pos=None,  with_pos=True):
        B, C, H, W = seg_token.shape
        depth_token = self.depth_proj(depth_token.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        refer_depth = F.grid_sample(depth_token, refer_coords, mode='nearest') #(B, C, n_plane, n_point)
        if with_pos:
            samp_pos = F.grid_sample(global_pos, refer_coords, mode='nearest')
            refer_depth = refer_depth + samp_pos
        refer_d_kv = self.kv_refer_depth(refer_depth.flatten(2).permute(0, 2, 1)) #(B, n_plane*n_point, 3*C)
        k_refer_d = refer_d_kv[:, :, :self.seg_dim]
        v_refer_d = refer_d_kv[:, :, self.seg_dim:]

        seg_relation = self.seg_proj(seg_token.flatten(2).permute(0, 2, 1)) # (B, H*W, C)
        # if self.size_ratio > self.args.adaptive_min_ratio:
        #     seg_relation = self.pre_pool(seg_relation)

        # distance between refer points and global points
        ty, tx = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        global_coords = torch.cat((tx.unsqueeze(-1), ty.unsqueeze(-1)), -1)
        global_coords[:, :, 0] = (global_coords[:,:,0] / (W-1)) * 2 - 1
        global_coords[:, :, 1] = (global_coords[:,:,1] / (H-1)) * 2 - 1
        global_coords = global_coords.cuda(refer_coords.device)
        global_coords_e = global_coords.reshape(-1, 2).unsqueeze(0).expand(B, -1, -1).unsqueeze(2) #(B, N, 1, 2)
        refer_coords_e = refer_coords.reshape(B, -1, 2).unsqueeze(1) #(B, 1, num_refer_point, 2)
        global_dist = torch.sqrt(torch.sum(torch.square(global_coords_e - refer_coords_e), dim=-1))
        global_dist = global_dist / 2.0 #(B, N, num_plane * 3)

        # whether pixels in refered planes that are defined by triangles
        batch_g_coords = global_coords.reshape(-1, 2).unsqueeze(0).expand(B, -1, -1) #(B, N, 2)
        refer_triangels = refer_coords #(B, num_plane, 3, 2)
        # transform coordination from (-1, 1) range back to image size (H, W) range, 
        # because we found it could be more accurate to judge whether pixel in triangles.
        triangles = (refer_triangels + 1) * 2
        pnt_coords = (batch_g_coords + 1) * 2
        triangles[:,:,:,0] = triangles[:,:,:,0] * (W - 1)
        triangles[:,:,:,1] = triangles[:,:,:,1] * (H - 1)
        pnt_coords[:,:,0] = pnt_coords[:,:,0] * (W - 1)
        pnt_coords[:,:,1] = pnt_coords[:,:,1] * (H - 1)
        coors_in_planes = batch_isin_triangle(triangles, pnt_coords)
        num_plane = triangles.shape[1]
        num_coord = batch_g_coords.shape[1]
        intriangle_map = coors_in_planes.reshape(B, num_plane, num_coord).type(torch.float).permute(0, 2, 1).contiguous() # (B, N, num_plane)

        seg_gemo = torch.cat([seg_relation, global_dist, intriangle_map], dim=-1)
        q_seg = self.q_seg_geometry(seg_gemo)
        seg_dpth_attn = q_seg @ k_refer_d.permute(0, 2, 1)
        seg_dpth_attn = seg_dpth_attn * (C ** -0.5)
        seg_dpth_attn = self.seg_dpth_softmax(seg_dpth_attn)
        fused_depth_token = seg_dpth_attn @ v_refer_d
        fused_depth_token = self.fused_depth_proj(fused_depth_token)
        fused_depth_token = fused_depth_token.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        return fused_depth_token

class TokenSampleRelation(nn.Module):
    def __init__(self, seg_dim, num_ref_points, num_coarse_points, size_ratio=0.25, args=None):
        super().__init__()
        self.seg_dim  = seg_dim
        self.num_ref_points = num_ref_points
        self.num_coarse_points = num_coarse_points
        self.seg_proj = Mlp(seg_dim, seg_dim)
        self.coars_depth_proj = Mlp(num_coarse_points, num_coarse_points)
        self.num_line = args.num_ref if args is not None else 50

        self.aspp1 = atrous_conv(in_channels=seg_dim, out_channels=seg_dim, dilation=6)
        self.aspp2 = atrous_conv(in_channels=seg_dim, out_channels=seg_dim, dilation=12)
        self.relation_module = nn.ModuleList()
        depth = 3
        in_features = num_ref_points + num_coarse_points
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

        #
        self.geome_conv = nn.Sequential(
            nn.Conv2d(in_channels=seg_dim, out_channels=seg_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=seg_dim, out_channels=seg_dim, kernel_size=(6, 1), stride=(6, 1)),
        )
        self.gome_depth_proj = nn.Sequential(
            Olp(in_features=seg_dim * num_coarse_points, out_features=(seg_dim  * num_coarse_points) // 8),
            Mlp(in_features=(seg_dim  * num_coarse_points) // 8, hidden_features=(seg_dim  * num_coarse_points) // 32, out_features=1),
            nn.Sigmoid()
        )
        self.old_depth_proj = Mlp(in_features=num_coarse_points, out_features=num_coarse_points)
    
    # seg_token/depth_token (B, C, H, W), refer_coords(line points) (B, num_line, 2, 2), coarse_coords (B, num_line, 1, 2)
    # global_pos (B, C, H, W), depth_pred (B, 1, H, W)
    def forward(self, seg_token, depth_token, refer_coords, coarse_coords, global_pos, old_points_depth=None, with_pos=True):
        B, C, H, W = depth_token.shape
        samp_depth = F.grid_sample(depth_token, refer_coords, mode='nearest') #(B, C, n_samp_point, 1)
        coarse_depth = F.grid_sample(depth_token, coarse_coords, mode='nearest') #(B, C, n_coarse_pnt, 1)
        if old_points_depth is not None:
            coarse_depth = coarse_depth + old_points_depth.unsqueeze(1)
        if with_pos:
            samp_pos = F.grid_sample(global_pos, refer_coords, mode='nearest')
            samp_depth = samp_depth + samp_pos
            refine_pos = F.grid_sample(global_pos, coarse_coords, mode='nearest')
            coarse_depth = coarse_depth + refine_pos
        # (B, C, num_line_or_triangle, 2_or_3) -> (B, C, n_samp_point) -> (B, C, 1, n_samp_point)
        samp_depth = samp_depth.flatten(2).unsqueeze(2)
        seg_token = self.seg_proj(seg_token.flatten(2).permute(0, 2, 1)) # (B, H*W, C)
        samp_seg = F.grid_sample(seg_token.permute(0, 2, 1).reshape(B, -1, H, W), 
                    torch.cat((refer_coords, coarse_coords), dim=2)) # (B, C, refer+coarse)
        samp_seg = samp_seg.flatten(2).unsqueeze(-1).contiguous()
        samp_rel = samp_seg @ samp_depth

        # seg_token = seg_token.permute(0, 2, 1).unsqueeze(-1).contiguous() # (B, C, H*W, 1)
        # samp_rel = seg_token @ samp_depth # (B, C, H*W, n_samp_point)
         
        B, C, N, N_p = samp_rel.shape
        rela = self.aspp1(samp_rel) + self.aspp2(samp_rel)
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




if __name__ == '__main__':
    import sys
    sys.path.append('/home/ly/workspace/git/segm/line_segm/letr-depth-2')
    import argparse
    from src.args import get_args_parser

    # def __init__(self, seg_dim, seg_height, seg_width, num_ref_points, args=None):
    parser = argparse.ArgumentParser('LETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    TR = TokenSampleRelation(seg_dim=64, seg_height=30, seg_width=40, num_ref_points=100, num_coarse_points=50, size_ratio=1/16, args=args)
    # def forward(self, seg_token, depth_token, refer_coords, global_pos, with_pos=True):
    seg_tok = torch.randn(2, 64, 30, 40)
    dth_tok = torch.randn(2, 64, 30, 40)
    samp_coors = (torch.rand(2, 50, 2, 2) * 2) - 1
    coarse_coords = (torch.rand(2, 50, 1, 2) * 2) - 1
    global_pos = torch.randn(2, 64, 30, 40)
    depth_pred = torch.rand(2, 1, 30, 40) * 10.0
    print(TR)
    a = TR(seg_tok, dth_tok, refer_coords=samp_coors, coarse_coords=coarse_coords, global_pos=global_pos)
