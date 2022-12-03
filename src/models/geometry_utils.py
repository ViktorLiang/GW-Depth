import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.layers import trunc_normal_
# from models.sne_model import SNE
#import pyrealsense2 as rs
# from util.commons import batch_isin_triangle
from sklearn.cluster import KMeans

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

    def __init__(self, in_features, out_features=None, kernel_size=3, padding=1, dilation=1, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or 1
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class PyramidConv(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_levels=2):
        super(PyramidConv, self).__init__()
        
        self.conv_pre = nn.ModuleList()
        for i in range(num_levels+1):
            conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels//2, bias=False, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(in_channels=hidden_channels//2, out_channels=hidden_channels, bias=False, kernel_size=3, stride=1, padding=1),
                nn.GELU()
            )    
            self.conv_pre.append(conv1)

        self.conv_scales = nn.ModuleList()
        for i in range(num_levels+1):
            conv2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels//2, bias=False, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(in_channels=hidden_channels//2, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1),
                nn.GELU()
            )
            self.conv_scales.append(conv2)
        
        self.norm_scales = nn.ModuleList()
        for i in range(num_levels+1):
            self.norm_scales.append(nn.LayerNorm(out_channels))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=(num_levels+1)*out_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        
        self.num_levels = num_levels
        self.min_size = self.min_size(msize=2, k=2, stride=2, pool_iter=num_levels)
    
    def output_size(self, rsize, k, stride, pool_iter):
        for i in range(pool_iter):
            rsize = ((rsize - k) / stride) + 1
        return rsize
    
    def min_size(self, msize, k, stride, pool_iter):
        for i in range(pool_iter):
           msize = (msize - 1) * stride + k
        return msize

    def forward(self, x, size=None):
        x_pyramid = [x]
        H, W = x.shape[-2:]
        # pool_h = self.output_size(H, 2, 2, self.num_levels)
        # pool_w = self.output_size(W, 2, 2, self.num_levels)
        xp = x
        padding_h, padding_w = 0, 0
        if H < self.min_size:
            padding_h = math.floor(self.min_size - H)
            xp = F.pad(xp, (0, 0, 0, padding_h), 'constant', 0)
        if W < self.min_size:
            padding_w = math.floor(self.min_size - W)
            xp = F.pad(xp, (0, padding_w), 'constant', 0)

        for i in range(self.num_levels):
            xp = F.avg_pool2d(xp, 2, stride=2)
            x_pyramid.append(xp)

        out = []
        for i, ix in enumerate(x_pyramid):
            # x = self.conv1(ix)
            x = self.conv_pre[i](ix)
            if size is not None:
                up_x = F.interpolate(x, size=size, mode='bilinear')
            else:
                up_x = x
            # out.append(self.conv2(up_x))
            up_x = self.conv_scales[i](up_x)
            b, c, h, w = up_x.shape
            up_x = up_x.flatten(2).permute(0, 2, 1)
            up_x = self.norm_scales[i](up_x)
            out.append(up_x.permute(0, 2, 1).reshape(b, c, h, w))
        x3 = torch.cat(out, dim=1)
        x = self.conv3(x3)
        return x

class TokenFuse(nn.Module):
    def __init__(self, size_ratio=0.25, args=None):
        super().__init__()
        self.seg_dim  = args.class_token_dim
        self.args = args

        self.seg_proj = Mlp(self.seg_dim, self.seg_dim)
        self.depth_proj = Mlp(self.seg_dim, out_features=self.seg_dim)
        self.norm_geometry = nn.LayerNorm(self.seg_dim)
        self.kv_refer_depth = Mlp(self.seg_dim, out_features=self.seg_dim * 2)
        self.q_seg_geometry = Mlp(self.seg_dim, out_features=self.seg_dim)

        self.size_ratio = size_ratio
        # if size_ratio > args.adaptive_min_ratio:
        #     self.pre_pool = nn.Sequential(
        #                     nn.AvgPool2d(kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
        #                     ConvA(in_features=seg_dim, out_features=seg_dim)
        #                 )

        self.seg_dpth_softmax = nn.Softmax(dim=-1)
        self.norm_fuse = nn.LayerNorm(self.seg_dim)
        self.fused_depth_proj = nn.Linear(self.seg_dim, self.seg_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    # seg_token/depth_token (B, C, H, W), refer_coords(plane points) (B, num_plane, 3, 2), coarse_coords (B, num_line, 1, 2)
    # token_pos (B, C, H, W), depth_pred (B, 1, H, W)
    def forward(self, seg_token, depth_token, refer_coords, token_pos,  with_pos=True):
        B, C, H, W = seg_token.shape
        depth_token_shorcut = depth_token
        depth_token = self.depth_proj(depth_token.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        refer_depth = F.grid_sample(depth_token, refer_coords, mode='nearest') #(B, C, n_plane, n_point)
        if with_pos:
            samp_pos = F.grid_sample(token_pos, refer_coords, mode='nearest')
            refer_depth = refer_depth + samp_pos
        refer_d_kv = self.kv_refer_depth(refer_depth.flatten(2).permute(0, 2, 1)) #(B, n_plane*n_point, 3*C)
        k_refer_d = refer_d_kv[:, :, :self.seg_dim]
        v_refer_d = refer_d_kv[:, :, self.seg_dim:]

        seg_relation = self.seg_proj(seg_token.flatten(2).permute(0, 2, 1)) # (B, H*W, C)
        # if self.size_ratio > self.args.adaptive_min_ratio:
        #     seg_relation = self.pre_pool(seg_relation)

        seg_gemo = seg_relation
        q_seg = self.norm_geometry(self.q_seg_geometry(seg_gemo))
        seg_dpth_attn = q_seg @ k_refer_d.permute(0, 2, 1)
        seg_dpth_attn = seg_dpth_attn * (C ** -0.5)
        seg_dpth_attn = self.seg_dpth_softmax(seg_dpth_attn)
        fused_depth_token = seg_dpth_attn @ v_refer_d
        fused_depth_token = self.norm_fuse(fused_depth_token)
        fused_depth_token = self.fused_depth_proj(fused_depth_token)
        fused_depth_token = fused_depth_token.permute(0, 2, 1).reshape(B, C, H, W).contiguous() + depth_token_shorcut

        return fused_depth_token

class NonLocalPlannarGuidance(nn.Module):
    def __init__(self, backbone_dim=128, num_points=50, num_levels=2, args=None):
        super().__init__()
        self.class_dim  = args.class_token_dim
        self.num_points = num_points
        self.args = args

        self.depth_fuse = nn.Sequential(
            nn.Linear(self.class_dim + backbone_dim, self.class_dim*2),
            nn.GELU(),
            nn.Linear(self.class_dim*2, self.class_dim),
            nn.GELU()
        )
        # self.seg_fuse = nn.Sequential(
        #     nn.Linear(self.class_dim + backbone_dim, self.class_dim),
        #     nn.GELU()
        # )
        # self.class_fuse = nn.Sequential(
        #     nn.Linear(self.class_dim * 2, self.class_dim),
        #     nn.GELU()
        # )
        self.class_kv = nn.Sequential(
            nn.Linear(self.class_dim, self.class_dim*2),
            nn.GELU()
        )
        self.softmax =  nn.Softmax(dim=-1)

        self.pre_depth_upsample = PyramidConv(in_channels=1, out_channels=1, hidden_channels=32, num_levels=num_levels)
        
        self.gru = ConvGRU(hidden_dim=self.class_dim, input_dim=1+num_points)
        self.new_depth = nn.Linear(self.class_dim, out_features=1)

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
    
    # seg_token/depth_token (B, C, H, W), refer_coords(plane points) (B, num_plane, 3, 2), coarse_coords (B, num_line, 1, 2)
    # token_pos (B, C, H, W), depth_pred (B, 1, H, W)
    def forward(self, backbone_x, seg_token, depth_token, refer_coords, token_pos, depth_pred=None,  with_pos=True):
        B, C, H, W = depth_token.shape
        x_flatten = backbone_x.flatten(2).permute(0, 2, 1)
        depth_flatten = depth_token.flatten(2).permute(0, 2, 1)
        # seg_flatten = seg_token.flatten(2).permute(0, 2, 1)
        depth_feats = self.depth_fuse(torch.cat([x_flatten, depth_flatten], dim=-1))
        # seg_feats = self.seg_fuse(torch.cat([x_flatten, seg_flatten], dim=-1))
        # class_feats = self.class_fuse(torch.cat([depth_feats, seg_feats], dim=-1))

        depth_pred = self.pre_depth_upsample(depth_pred, size=(H, W))

        class_kv = self.class_kv(depth_feats)
        class_k = class_kv[:, :, :self.class_dim]
        class_v = class_kv[:, :, self.class_dim:]

        class_pnt = F.grid_sample(class_k.permute(0, 2, 1).reshape(B, -1, H, W), refer_coords, mode='nearest', align_corners=False) #(B, C, n_plane, n_point)
        # depth_pnt = F.grid_sample(depth_pred, refer_coords, mode='nearest', align_corners=False) #(B, C, n_plane, n_point)
        # depth_pnt = depth_pnt.flatten(2)
        if with_pos:
            samp_pos = F.grid_sample(token_pos, refer_coords, mode='nearest', align_corners=False)
            class_pnt = class_pnt + samp_pos

        class_pnt = class_pnt.flatten(2) * (self.class_dim ** -0.5)
        pnt_global_corr = class_v @ class_pnt

        #
        pnt_global_corr = pnt_global_corr.permute(0, 2, 1).reshape(B, -1, H, W)
        c1 = torch.cat([pnt_global_corr, depth_pred], dim=1)
        c2 = depth_feats.permute(0, 2, 1).reshape(B, -1, H, W)
        c = self.gru(c2, c1)
        
        c = c.flatten(2).permute(0, 2, 1)
        new_depth = self.new_depth(c)
        new_depth = new_depth.permute(0, 2, 1).reshape(B, -1, H, W)
        new_depth = new_depth.sigmoid()
        return new_depth, None

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class ReflectionReduce(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        rhint_channels = [32, 64, 128, 256]
        self.sp_red1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ELU(),
            upconv(16, rhint_channels[0], ratio=0), #1/2
            )
        self.sp_red2 = nn.Sequential(
            nn.Conv2d(rhint_channels[0], 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ELU(),
            upconv(64, rhint_channels[1]),#1/4
            )
        self.sp_red3 = nn.Sequential(
            nn.Conv2d(rhint_channels[1], 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ELU(),
            upconv(256, rhint_channels[2]),#1/8
            )
        self.sp_red4 = nn.Sequential(
            nn.Conv2d(rhint_channels[2], 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ELU(),
            upconv(256, rhint_channels[3]),#1/16
            )
    
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
        

    def forward(self, reflc_png, layers_size):
        size16, size8, size4 = layers_size
        ht1 = self.sp_red1(reflc_png)#1/2, 32
        ht2 = self.sp_red2[:-1](ht1)
        ht2 = self.sp_red2[-1](ht2, size=size4)#1/4, 128

        ht3 = self.sp_red3[:-1](ht2)#1/8, 256
        ht3 = self.sp_red3[-1](ht3, size=size8)#1/8, 256

        ht4 = self.sp_red4[:-1](ht3)
        ht4 = self.sp_red4[-1](ht4, size=size16)#1/16, 256
        return [ht4, ht3, ht2]

def distance_map(height, width, device=None):
    # # distance between refer points and global points
    ty, tx = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    global_coords = torch.cat((tx.unsqueeze(-1), ty.unsqueeze(-1)), -1).type(torch.float)
    global_coords[:, :, 0] = (global_coords[:,:,0] / (width-1)) * 2.0 - 1.0
    global_coords[:, :, 1] = (global_coords[:,:,1] / (height-1)) * 2.0 - 1.0
    if device is not None:
        global_coords = global_coords.cuda(device)
    global_coords1 = global_coords.reshape(-1, 2).unsqueeze(0) #(1, N, 2)
    global_coords2 = global_coords.reshape(-1, 2).unsqueeze(1) #(N, 1, 2)

    #(N, N)
    global_dist = torch.sqrt(torch.sum(torch.square(global_coords1 - global_coords2), dim=-1))
    # norm to 0-1
    global_dist = global_dist / 4.0 
    return global_dist

def sample_by_centers(center_coors, line_coors, line_logits, 
        input_h, input_w, shortest_ratio = 0.05,
        num_clusters=16, top_num=6, sample_line_num=50, ):
    B = center_coors.shape[0]
    cluster_ids = np.arange(num_clusters)
    center_coors_cpu = center_coors.clone().detach().cpu()
    choosen_lines = []
    for ib in range(B):
        kms = KMeans(n_clusters=num_clusters, random_state=0).fit(center_coors_cpu[ib])
        # get top 4 logits
        center_labels = torch.tensor(kms.labels_).cuda(line_coors.device)
        lines_list = []
        logits_list = []
        for cid in cluster_ids:
            lines = line_coors[ib][center_labels == cid]
            logits = line_logits[ib][center_labels == cid]
            # get topk logits for each label
            _, topid = torch.topk(logits[:, 0], min(top_num, logits.shape[0]))
            top_lines = lines[topid]
            top_logits = logits[topid]
            # filter out too short lines
            rec_top_lines = torch.zeros_like(top_lines)
            rec_top_lines[:, 0::2] = top_lines[:, 0::2] * input_w
            rec_top_lines[:, 1::2] = top_lines[:, 0::2] * input_h
            rec_dist = torch.sqrt(torch.sum((rec_top_lines[:, 0:2] - rec_top_lines[:, 2:]) ** 2, dim=1))
            min_dist = min(input_h, input_w) * shortest_ratio
            keep = rec_dist > min_dist
            print(keep)
            top_lines = top_lines[keep]
            top_logits = top_logits[keep]
            lines_list.append(top_lines)
            logits_list.append(top_logits)
            
        st_lines = torch.cat(lines_list, dim=0)
        st_logits = torch.cat(logits_list, dim=0)
        remain_num = sample_line_num - st_lines.shape[0]
        print('st_lines', st_lines.shape, 'st_logits', st_logits.shape, 'remain_num', remain_num)
        if remain_num > 0:
            _, remian_top_id = torch.topk(line_logits[ib][:, 0], remain_num)
            comp_lines = line_coors[ib][remian_top_id]
            lines_list.append(comp_lines)
            st_lines = torch.cat(lines_list, dim=0)
            comp_logits = line_logits[ib][remian_top_id]
            logits_list.append(comp_logits)
            st_logits = torch.cat(logits_list, dim=0)
        elif remain_num < 0: # 
            _, topid = torch.topk(st_logits[:, 0], sample_line_num)
            st_lines = st_lines[topid]
            st_logits = st_logits[topid]
        choosen_lines.append(st_lines)
    choosen_lines = torch.stack(choosen_lines)
    return choosen_lines

class Global2PointGraph(nn.Module):
    def __init__(self, upsample_ratio, num_point, args):
        super().__init__()
        self.dpeth_token_proj = Mlp(args.class_token_dim, args.class_token_dim)
        self.dim = args.class_token_dim
        self.fix_size = args.class_init_size
        self.upsample_ratio = upsample_ratio
        self.node_relation = Mlp(num_point, hidden_features=4*num_point, out_features=num_point, drop=0.2)
        self.node_attention = Mlp(num_point, hidden_features=4*num_point, out_features=num_point, drop=0.2)
        self.softmax = nn.Softmax(dim=-1)

        self.new_size = args.class_init_size * upsample_ratio
        self.token_node_fuse = Mlp(self.new_size * 2, out_features=1)

    # token_init (1, sH, sW, dim)
    # point_token (1, nPnt, dim)
    def forward(self, token_init, point_token, height, width, is_init=False):
        B, nPnt = point_token.shape[:2]
        if  is_init:
            token_init_expd = token_init
        else:
            token_init_rowexp = torch.repeat_interleave(token_init, 2, 1)
            token_init_expd = torch.repeat_interleave(token_init_rowexp, 2, 2)
        
        sH, sW = token_init_expd.shape[1:3]

        # token_raw = F.adaptive_max_pool2d(token_init_expd.permute(0, 3, 1, 2), (height, width))
        token_raw = F.interpolate(token_init_expd.permute(0, 3, 1, 2), (height, width), mode='nearest')
        if not is_init:
            # token_raw = F.adaptive_max_pool2d(token_init_expd.permute(0, 3, 1, 2), (height, width))
            # token_init_expd = F.adaptive_max_pool2d(token_init_expd.permute(0, 3, 1, 2), (self.new_size, self.new_size))
            token_raw = F.interpolate(token_init_expd.permute(0, 3, 1, 2), (height, width), mode='nearest')
            token_init_expd = F.interpolate(token_init_expd.permute(0, 3, 1, 2), (self.new_size, self.new_size), mode='nearest')
            token_init_expd = token_init_expd.permute(0, 2, 3, 1)
            sH, sW = token_init_expd.shape[1:3]
        token_raw = token_raw.permute(0, 2, 3, 1)

        token_templ = token_init_expd.flatten(start_dim=1, end_dim=2)
        node_adj = torch.matmul(token_templ, point_token.permute(0, 2, 1)) # (B, sH*sW, nPnt)
        node_adj = node_adj * (self.dim ** -0.5)

        # adjacent relation
        node_adj = self.node_relation(node_adj)
        node_adj = node_adj.reshape(B, sH, sW, -1)
        node_w = node_adj.permute(0, 1, 3, 2) @ token_init_expd
        node_w = node_w * (sW ** -0.5)
        node_h = node_adj.permute(0, 2, 3, 1) @ token_init_expd.permute(0, 2, 1, 3)
        node_h = node_h * (sH ** -0.5)
        token_n = torch.cat([node_w, node_h], dim=1)
        token_fused = self.token_node_fuse(token_n.flatten(2).permute(0, 2, 1))
        token_fused = token_fused.reshape(B, nPnt, -1)

        # global attention
        token_raw = token_raw.flatten(start_dim=1, end_dim=2)
        node_attn = torch.matmul(token_raw, point_token.permute(0, 2, 1)) # (B, H*W, nPnt)
        node_attn = node_attn * (self.dim ** -0.5)
        node_attn = self.softmax(self.node_attention(node_attn))

        token_new = node_attn @ token_fused 
        token_new = token_new + token_raw
        return token_new

class PointGuidedTokenFuse(nn.Module):
    def __init__(self, x_dim,  args=None):
        super().__init__()
        self.class_dim  = args.class_token_dim
        self.args = args

        self.xseg_proj = Mlp(self.class_dim + x_dim, hidden_features=x_dim, out_features=self.class_dim)
        self.xdth_proj = Mlp(self.class_dim + x_dim, hidden_features=x_dim, out_features=self.class_dim)

        self.kv_refer_depth = Mlp(self.class_dim, out_features=self.class_dim * 2)
        self.q_seg = Mlp(self.class_dim, out_features=self.class_dim)
        self.norm_seg = nn.LayerNorm(self.class_dim)

        # 
        self.convctx_pre3 = nn.Sequential(ConvA(self.class_dim, self.class_dim * 4, kernel_size=3, padding=1),    
                                ConvA(self.class_dim * 4, self.class_dim * 4, kernel_size=3, padding=1))
        self.convctx_norm3 = nn.LayerNorm(self.class_dim * 4)
        self.convctx_after3 = ConvA(self.class_dim*4, self.class_dim, kernel_size=3, padding=1)

        self.convctx_pre5 = nn.Sequential(ConvA(self.class_dim, self.class_dim * 4, kernel_size=5, padding=2), 
                                        ConvA(self.class_dim * 4, self.class_dim * 4, kernel_size=5, padding=2))
        self.convctx_norm5 = nn.LayerNorm(self.class_dim * 4)
        self.convctx_after5 = ConvA(self.class_dim*4, self.class_dim, kernel_size=5, padding=2)
        self.mlpctx = Mlp(in_features=self.class_dim, hidden_features=self.class_dim*4, out_features=self.class_dim)

        self.seg_dpth_softmax = nn.Softmax(dim=-1)
        self.fuse_proj = nn.Linear(self.class_dim, self.class_dim)
        self.norm_fuse = nn.LayerNorm(self.class_dim)
        self.fused_depth_proj = nn.Linear(self.class_dim, self.class_dim)

        # self.ks_list=[[7,3],[9,4],[11,5]]
        self.ks_list=[[11,5], [17, 8]]
        self.mutil_depth_fuse = nn.Linear(self.class_dim*len(self.ks_list), self.class_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def conv_process(self, x):
        b, _, h, w = x.shape
        x = self.convctx_pre3(x)
        x = self.convctx_norm3(x.flatten(2).permute(0, 2, 1))
        c = x.shape[-1]
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        x = self.convctx_after3(x)

        x = self.convctx_pre5(x)
        x = self.convctx_norm5(x.flatten(2).permute(0, 2, 1))
        c = x.shape[-1]
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        x = self.convctx_after5(x)
        return x
    
    def depth_seg_fuse(self, depth_token, refer_depth, seg_ctx, ks):
        H, W = depth_token.shape[-2:]
        dt = depth_token

        msize = 2
        min_size = (msize - 1) * ks[1] + ks[0]
        padding_h, padding_w = 0, 0
        if H < min_size:
            padding_h = math.floor(min_size - H)
            dt = F.pad(dt, (0, 0, 0, padding_h), 'constant', 0)
        if W < min_size:
            padding_w = math.floor(min_size - W)
            dt = F.pad(dt, (0, padding_w), 'constant', 0)

        dtx1 = F.avg_pool2d(dt, kernel_size=ks[0], stride=ks[1])
        dtx1 = self.conv_process(dtx1)
        dtx1 = dtx1.flatten(2).permute(0, 2, 1)

        ctx1 = self.mlpctx(torch.cat([dtx1, refer_depth], dim=1))
        refer_d_kv = self.kv_refer_depth(ctx1) #(B, n_plane*n_point, 3*C)
        k_refer_d = refer_d_kv[:, :, :self.class_dim]
        v_refer_d = refer_d_kv[:, :, self.class_dim:]

        seg_dpth_attn = seg_ctx @ k_refer_d.permute(0, 2, 1)
        seg_dpth_attn = self.seg_dpth_softmax(seg_dpth_attn)
        fused_depth_token = seg_dpth_attn @ v_refer_d
        fused_depth_token = self.fuse_proj(fused_depth_token)
        fused_depth_token = self.fused_depth_proj(self.norm_fuse(fused_depth_token))
        return fused_depth_token
    
    # seg_token/depth_token (B, C, H, W), refer_coords(plane points) (B, num_plane, 3, 2), coarse_coords (B, num_line, 1, 2)
    # token_pos (B, C, H, W), depth_pred (B, 1, H, W)
    def forward(self, backbone_x, seg_token, depth_token, refer_coords, token_pos, with_pos=True):
        B, C, H, W = seg_token.shape
        # depth_token_shorcut = depth_token
        st = seg_token.flatten(2).permute(0, 2, 1)
        dt = depth_token.flatten(2).permute(0, 2, 1)
        stx = torch.cat([st, backbone_x], dim=2)
        dtx = torch.cat([dt, backbone_x], dim=2)
        stx = self.xseg_proj(stx)
        dtx = self.xdth_proj(dtx)
        depth_token_f = dtx.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        refer_depth = F.grid_sample(depth_token_f, refer_coords, mode='nearest') #(B, C, n_plane, n_point)
        if with_pos:
            samp_pos = F.grid_sample(token_pos, refer_coords, mode='nearest')
            refer_depth = refer_depth + samp_pos
        refer_depth = refer_depth.flatten(2).permute(0, 2, 1)
        
        q_seg = self.norm_seg(self.q_seg(stx))
        q_seg = q_seg * (self.class_dim ** -0.5)

        new_dtokens = []
        for ks in self.ks_list:
            new_depth_token = self.depth_seg_fuse(depth_token, refer_depth, q_seg, ks)
            new_dtokens.append(new_depth_token)
        fused_depth_token = self.mutil_depth_fuse(torch.cat(new_dtokens, dim=-1))
        fused_depth_token = fused_depth_token.permute(0, 2, 1).reshape(B, -1, H, W)
        return fused_depth_token

if __name__ == '__main__':
    import sys
    sys.path.append('/home/ly/workspace/git/segm/line_segm/letr-depth-2')
    import argparse
    from src.args import get_args_parser

    parser = argparse.ArgumentParser('LETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # print(distance_map(3, 3,))
    NLPG = PointGuidedTokenFuse(256, args=args)
    backbone_x = torch.randn(2, 50 * 50, 256)
    st = torch.randn(2, 64, 50, 50)
    dt = torch.randn(2, 64, 50, 50)
    points = (torch.rand(2, 50, 3, 2) * 2.0) - 1.0
    global_pose = torch.randn(2, 64, 50, 50) 
    a = NLPG(backbone_x, st, dt, points, global_pose)
