import audioop
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.spatial import ConvexHull

from timm.models.layers import trunc_normal_
from models.geometry_utils import Olp, Mlp

class ConvLn(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, pad, dilation):
        super().__init__()
        self.conv =  nn.Conv2d(in_planes, out_planes,
                            kernel_size=kernel_size, stride=stride,
                            padding=dilation if dilation > 1 else pad,
                            dilation=dilation, bias=False)
        self.layer_norm = nn.LayerNorm(out_planes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer_norm(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(ConvLn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.GELU())
        self.conv2 = ConvLn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out

class PyramidLayer(nn.Module):
    def __init__(self, in_dim, pool_sizes):
        super().__init__()
        self.in_dim = in_dim
        self.pool_sizes = pool_sizes

        self.firstconv = nn.Sequential(ConvLn(in_dim, in_dim, 3, 1, 1, 1), nn.GELU(),
                                       ConvLn(in_dim, in_dim*2, 3, 1, 1, 1), nn.GELU())

        self.inplanes = in_dim*2
        self.layer1 = self._make_layer(BasicBlock, in_dim*2, 1, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, in_dim*2, 2, 1, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, in_dim*2, 2, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, in_dim*2, 1, 1, 1, 2)

        branch_dim = self.inplanes
        self.branch1 = nn.Sequential(nn.AvgPool2d(pool_sizes[0], stride=pool_sizes[0]),
                                     ConvLn(branch_dim, branch_dim, 3, 1, 1, 1),
                                     nn.GELU())

        self.branch2 = nn.Sequential(nn.AvgPool2d(pool_sizes[1], stride=pool_sizes[1]),
                                     ConvLn(branch_dim, branch_dim, 3, 1, 1, 1),
                                     nn.GELU())

        self.branch3 = nn.Sequential(nn.AvgPool2d(pool_sizes[2], stride=pool_sizes[2]),
                                     ConvLn(branch_dim, branch_dim, 3, 1, 1, 1),
                                     nn.GELU())

        self.branch4 = nn.Sequential(nn.AvgPool2d(pool_sizes[3], stride=pool_sizes[3]),
                                     ConvLn(branch_dim, branch_dim, 3, 1, 1, 1),
                                     nn.GELU())

        self.lastconv = nn.Sequential(ConvLn(5*branch_dim, branch_dim*2, 3, 1, 1, 1),
                                      nn.GELU(),
                                      nn.Conv2d(branch_dim*2, in_dim, kernel_size=1, padding=0, stride=1, bias=False))
    
    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvLn(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, pad=pad, dilation=dilation)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)
    
    def pad_before_pool(self, x):
        H, W = x.shape[-2:]
        bottom_pad, right_pad = 0, 0
        if H - self.pool_sizes[0] < 0:
            bottom_pad =  math.ceil(self.pool_sizes[0] - H)
        if W - self.pool_sizes[0] < 0:
            right_pad =  math.ceil(self.pool_sizes[0] - W)
        if bottom_pad > 0 or right_pad > 0:
            x = F.pad(x, (0, right_pad, 0, bottom_pad), value=0)

        return x

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pad_before_pool(x)

        x1 = self.branch1(x)
        x1 = F.interpolate(x1, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        x2 = self.branch2(x)
        x2 = F.interpolate(x2, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        x3 = self.branch3(x)
        x3 = F.interpolate(x3, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        x4 = self.branch4(x)
        x4 = F.interpolate(x4, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        xx = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = self.lastconv(xx)

        return x

def area(p):
    return abs(sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])

class OffsetGeneration(nn.Module):
    def __init__(self, x_dim, num_ref_points, pool_sizes=[32, 16, 8, 4], args=None):
        super().__init__()
        self.class_dim  = args.class_token_dim
        self.num_ref_points = num_ref_points
        self.num_plane = args.num_ref or num_ref_points // 3
        self.args = args

        self.backbone_norm = nn.LayerNorm(x_dim)
        self.backbone_fc = nn.Sequential(
            nn.Conv2d(x_dim, x_dim//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(x_dim//2, self.class_dim, kernel_size=1),
            nn.GELU()
        )

        #self.token_fuse = Mlp(self.class_dim*2, hidden_features=self.class_dim*2, out_features=self.class_dim, drop=0.1)
        self.channel_attention_fc = Olp(self.class_dim, self.class_dim)
        self.channel_attention_softmax = nn.Softmax(-1)
        self.v_proj = nn.Linear(self.class_dim, out_features=self.class_dim)
        self.qk_refer = nn.Linear(self.class_dim, out_features=self.class_dim * 2)
        self.refer_softmax = nn.Softmax(dim=-1)

        self.global_norm = nn.LayerNorm(self.class_dim)
        self.global_offset = nn.Sequential(
                nn.Conv2d(self.class_dim, self.class_dim//2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self.class_dim//2, self.class_dim//2, kernel_size=3, padding=1),
                nn.Conv2d(self.class_dim//2, self.class_dim//2, kernel_size=3, dilation=6, padding=6),
                nn.Conv2d(self.class_dim//2, self.class_dim//2, kernel_size=3, dilation=16, padding=16),
                nn.Conv2d(self.class_dim//2, self.class_dim//2, kernel_size=3, padding=1),
                nn.Conv2d(self.class_dim//2, self.class_dim//4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self.class_dim//4, self.class_dim//4, kernel_size=1),
        )

        self.refer_proj = nn.Linear(x_dim, out_features=self.class_dim//4)
        self.pyramid = PyramidLayer(num_ref_points, pool_sizes=pool_sizes)


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
    def forward(self, x, depth_token, refer_coords, token_pos, with_pos=True, size=None):
        H, W = size
        B, N, C = x.shape
        line_num = refer_coords.shape[1]
        # token based channel attention
        channel_attn = self.channel_attention_softmax(self.channel_attention_fc(depth_token))
        ch_attn = channel_attn.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.backbone_norm(x)
        x_spatial = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_offset = self.backbone_fc(x_spatial)
        x_offset = ch_attn * x_offset + x_offset # B, Cs, H, W

        x_offset = x_offset.permute(0, 2, 3, 1)
        x_offset = self.global_norm(x_offset)
        global_offset = self.global_offset(x_offset.permute(0, 3, 1, 2))

        # sample offset for guided points
        refer_x = F.grid_sample(x_spatial, refer_coords)
        if with_pos:
            refer_pos = F.grid_sample(token_pos, refer_coords)
            refer_x = refer_x + refer_pos
        refer_x = refer_x.flatten(2)
        refer_x = self.refer_proj(refer_x.permute(0, 2, 1))

        ref_g = torch.matmul(refer_x, global_offset.flatten(2))
        ref_g = ref_g.reshape(B, -1, H, W).contiguous()

        ref_g_c = self.pyramid(ref_g)
        ref_g = ref_g_c.flatten(3).permute(0, 3, 1, 2)
        g_coords = F.sigmoid(ref_g)
        num_pixel = g_coords.shape[1]

        b_hull_areas = []
        for i in range(B):
            hull_areas = []
            for j in range(num_pixel):
                gc = ConvexHull(g_coords[i, j].detach().cpu().numpy())
                v_points = gc.points[gc.vertices]
                a = area(v_points)
                hull_areas.append(a)
            b_hull_areas.append(hull_areas)
        b_hull_areas = torch.tensor(b_hull_areas, device=refer_coords.device) # (b, num_global_pixles)
        max_area_id = torch.argmax(b_hull_areas, dim=-1)
        choosen_coords = g_coords[:, max_area_id]
        choosen_coords = choosen_coords.squeeze(1).reshape(B, -1, 2, 2) * 2 - 1
        all_coords = torch.cat([refer_coords, choosen_coords], dim=1)
        return all_coords

class PointBasedPred(nn.Module):
    def __init__(self, dim, token_dim, pool_sizes, point_num):
        super().__init__()
        self.dim = dim
        # in_dim, pool_sizes, first_stride=1
        self.pre_proj = nn.Linear(dim+token_dim, dim)
        self.refer_proj = nn.Linear(dim, dim*2)
        self.pyramid = PyramidLayer(point_num, pool_sizes)
    
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
    
    def forward(self, x, depth_token, pre_depth, coords, height, width, with_pos=True, pos_embedding=None):
        x_global = self.pre_proj(torch.cat([x, depth_token], dim=-1))
        x_refer = self.refer_proj(x_global)
        xg = x_refer[:, :, :self.dim]
        xr = x_refer[:, :, self.dim:]
        B = xr.shape[0]
        xr = xr.permute(0, 2, 1).reshape(B, -1, height, width)
        refer_x = F.grid_sample(xr, coords)
        if with_pos:
            refer_pos = F.grid_sample(pos_embedding, coords)
            refer_x = refer_x + refer_pos
        anchor_depth = F.grid_sample(pre_depth, coords)
        anchor_depth = anchor_depth.permute(0, 2, 1, 3)
        
        refer = refer_x.flatten(2)
        rg = xg @ refer
        rg = rg * (self.dim ** -2)

        rg = rg.permute(0, 2, 1).reshape(B, -1, height, width)
        rg_py = self.pyramid(rg)
        rg_attn = F.softmax(rg_py, dim=1)
        anchor_xr_depth = rg_attn * anchor_depth
        pred = torch.sum(anchor_xr_depth, dim=1, keepdim=True)
        return pred

class CertainSample(nn.Module):
    def __init__(self, dim, min_depth, max_depth, sample_num):
        super().__init__()
        self.dim = dim
        self.min_depth = min_depth/max_depth
        self.max_depth = 1.0
        assert sample_num > 0, sample_num
        self.sample_num = sample_num
    
    def forward(self, pred_small, pred_large, interval):
        B, C, H, W = pred_large.shape
        small_interp = F.interpolate(pred_small, size=(H, W), mode='bilinear', align_corners=True)
        variance = (small_interp - pred_large) ** 2
        full_interval = [self.min_depth] + interval
        full_interval.append(self.max_depth)

        # generating depth mask for each interval
        inter_sampled_coors = []
        total_depth_num = H * W
        for b in range(B):
            batch_sampled_coors = []
            already_num = 0
            inter_sampled_num = []
            for i in range(len(full_interval)-1):
                start_depth = full_interval[i]
                end_depth = full_interval[i+1]
                inter_mask = (pred_large[b] >= start_depth) & (pred_large[b] < end_depth)
                # inter_var = variance[b][inter_mask]
                # inter_masks.append(inter_mask)
                # inter_vars.append(inter_var)
                inter_total = torch.sum(inter_mask)
                sample_inter_d_num = (inter_total / total_depth_num) * self.sample_num
                # # at least keep one point when the interval has depth points
                # if inter_total > 0 and sample_inter_d_num < 1:
                #     sample_inter_d_num = torch.tensor(1.0)
                sample_inter_d_num = int(torch.min(torch.floor(sample_inter_d_num), inter_total))
                if sample_inter_d_num > 0:
                    val, indx = torch.topk(variance[b].flatten(0), sample_inter_d_num)
                    indx, _ = indx.sort()
                    row = torch.div(indx, W, rounding_mode='floor')
                    col = indx % W
                    coors = torch.stack([col, row])
                    batch_sampled_coors.append(coors)
                    inter_sampled_num.append(coors.shape[-1])
                    already_num += sample_inter_d_num
            
            if len(batch_sampled_coors) > 0:
                batch_catted_sampled_coors = torch.cat(batch_sampled_coors, dim=1) # (2, N)
                remain_num = self.sample_num - already_num
            else:
                # sample globally when no interval points found.
                v_f = variance[b].flatten(1)
                val, indx = torch.topk(v_f, self.sample_num, dim=-1)
                indx, _ = indx.sort()
                row = torch.div(indx, W, rounding_mode='floor')
                col = indx % W
                batch_catted_sampled_coors = torch.cat([col, row], dim=0)
                remain_num = 0

            # complement or remove to make fixed size
            ## repeat when too much shortage
            if remain_num > 0 and remain_num >= already_num:
                copy_times = remain_num // already_num + 1
                batch_catted_sampled_coors = batch_catted_sampled_coors.repeat(1, copy_times)
                remain_num = self.sample_num - already_num * copy_times
            ## copy to complement
            if remain_num > 0:
                copy_coors = batch_catted_sampled_coors[:, -remain_num:]
                batch_catted_sampled_coors = torch.cat([batch_catted_sampled_coors, copy_coors], dim=1)
            if remain_num < 0:
                # remove from top size interval
                maxid = torch.argmax(torch.tensor(inter_sampled_num))
                batch_sampled_coors[maxid] = batch_sampled_coors[maxid][:, :remain_num]
                batch_catted_sampled_coors = torch.cat(batch_sampled_coors, dim=1)
            
            inter_sampled_coors.append(batch_catted_sampled_coors)
        
        sampled_coors = torch.stack(inter_sampled_coors).type(torch.float)
        sampled_coors = sampled_coors.permute(0, 2, 1)
        sampled_coors[:, :, 0] = (sampled_coors[:, :, 0] / W) * 2 -1
        sampled_coors[:, :, 1] = (sampled_coors[:, :, 1] / H) * 2 -1
        sampled_coors = sampled_coors[:, :, None]
        return sampled_coors


# random_lines, (-1, 1)
def sample_along_seg(random_lines, height, width, sample_num_seg = 10, ratio_sample=False):
    # inverse norm to from (-1, 1) -> (0, 1)
    random_lines = (random_lines + 1) / 2
    # inverse norm to width, height
    random_lines[:, :, :, 0] = random_lines[:, :, :, 0] * width
    random_lines[:, :, :, 1] = random_lines[:, :, :, 1] * height
    # img = np.ones((height, width, 3)) * 255
    if ratio_sample:
        max_dist = math.sqrt(height ** 2 + width ** 2)
        line_length = torch.sqrt(torch.sum((random_lines[:, 0] - random_lines[:, 1]) ** 2, dim=-1))
        sample_num = []
        for i, len in enumerate(line_length):
            sample_num.append((len / max_dist) * sample_num_seg)

    st = torch.argmin(random_lines[:, :, :, 0], axis=2)
    end = torch.argmax(random_lines[:, :, :, 0], axis=2)
    st_ids = st[:, :, None, None].expand(-1, -1, -1, 2)
    end_ids = end[:, :, None, None].expand(-1, -1, -1, 2)
    st_points = torch.gather(random_lines, 2, index=st_ids).squeeze(2)
    end_points = torch.gather(random_lines, 2, index=end_ids).squeeze(2)

    dist = torch.sqrt((st_points[:, :, 0] - end_points[:, :, 0]) ** 2 + (st_points[:, :, 1] - end_points[:, :, 1]) ** 2)
    x_dist =  torch.sqrt((st_points[:, :, 0] - end_points[:, :, 0]) ** 2)
    y_dist =  torch.sqrt((st_points[:, :, 1] - end_points[:, :, 1]) ** 2)
    cosin = x_dist / dist
    sin = y_dist / dist
    dist_seg = dist / sample_num_seg
    seg_x = dist_seg * cosin
    seg_y = dist_seg * sin
    new_points_x = []
    is_row_ascent = st_points[:, :, 1] < end_points[:, :, 1]
    row_oper = is_row_ascent.type(torch.int8)
    row_oper = row_oper * 2 -1

    for seg_i in range(1, sample_num_seg+1):
        p_x = st_points[:, :, 0] + seg_x * seg_i
        p_y = st_points[:, :, 1] + seg_y * seg_i * row_oper
        seg_p = torch.cat([p_x[:, :, None], p_y[:, :, None]], dim=-1)
        seg_p = seg_p[:, :, None]
        new_points_x.append(seg_p)
    
    new_seg_points = torch.cat(new_points_x, dim=-2)
    all_seg_points = torch.cat([random_lines, new_seg_points], dim=2)
    all_seg_points[:, :, :, 0] = all_seg_points[:, :, :, 0] / width
    all_seg_points[:, :, :, 1] = all_seg_points[:, :, :, 1] / height

    # norm to [-1, 1]
    all_seg_points = (all_seg_points * 2) -1
    return all_seg_points

# random_lines, (-1, 1)
def sample_mid_seg(random_lines, height, width, sample_num_seg = 10, ratio_sample=False):
    mid_x = (random_lines[:, :, 0, 0] + random_lines[:, :, 1, 0]) / 2
    mid_y = (random_lines[:, :, 0, 1] + random_lines[:, :, 1, 1]) / 2
    mid_points = torch.cat([mid_x[:, :, None, None], mid_y[:, :, None, None]], dim=-1)
    new_points = torch.cat([random_lines, mid_points], dim=-2)
    return new_points    