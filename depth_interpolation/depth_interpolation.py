from enum import EnumMeta
import os
import sys
import time

from tqdm import tqdm
import cv2
from matplotlib.pyplot import axis
import numpy as np
from numpy.lib.function_base import interp
from scipy import interpolate
import glob
import pyrealsense2 as rs
import json

from commons import gen_pairs, within_poly
from read_binfile import read_depth_npy, vis_depth_mat
from raw_preprocess import GLASS_LABELS

MIN_VALUE = 1e-6

def readjust_coordinates(vertex_p, vertical_line=True):
    # vertex_p [[column_no, row_no],[],[],[]]
    # readjust the coordinates to make their value to be integers.
    # column and row coordinations are rounded to make them more close to the center region
    vertex_p[0:2, 0] = np.ceil(vertex_p[0:2, 0])
    vertex_p[2:, 0] = np.floor(vertex_p[2:, 0])

    vertex_p[0, 1] = np.ceil(vertex_p[0, 1])
    vertex_p[3, 1] = np.ceil(vertex_p[3, 1])
    vertex_p[1, 1] = np.floor(vertex_p[1, 1])
    vertex_p[2, 1] = np.floor(vertex_p[2, 1])
    vertex_p = vertex_p.astype(np.int32)
    # vertex of vertical line should have same column number
    if vertical_line:
        vertex_p[:2, 0] = vertex_p[:2, 0].max()
        vertex_p[2:, 0] = vertex_p[2:, 0].min()
    
    return vertex_p

def get_depth_from_mat(dpth_mat, edge_points):
    d_values = []
    for i in range(len(edge_points)):
        x, y = edge_points[i]
        d_values.append(dpth_mat[y, x])
    return np.array(d_values, dtype=np.float32)

def get_sides_length(measured_sides_lengths, proj_world_coors, measured_in_centimeter=True):
    sides_lens = []
    # use measured length firstly
    if len(measured_sides_lengths) > 0:
        m_scale = 10.0 if measured_in_centimeter else 1.0
        for vex_i, ml in enumerate(measured_sides_lengths):
            if ml > 0:
                sides_lens.append(ml * m_scale)
            else:
                st_p = proj_world_coors[vex_i]
                if vex_i < len(proj_world_coors) - 1:
                    ed_p = proj_world_coors[vex_i + 1]
                else:
                    ed_p = proj_world_coors[0]
                abs_dist = coor_dist(st_p, ed_p)
                sides_lens.append(abs_dist)
    else:
        for vex_i, _ in enumerate(proj_world_coors):
            st_p = proj_world_coors[vex_i]
            if vex_i < len(proj_world_coors) - 1:
                ed_p = proj_world_coors[vex_i + 1]
            else:
                ed_p = proj_world_coors[0]
            abs_dist = coor_dist(st_p, ed_p)
            sides_lens.append(abs_dist)
    return sides_lens

def calculate_depth_by_collinear(point1_3d, point2_3d, point1_to_unknown_3dist, unkown_point_inc):
    # 0: unkown point located above or left of point1, 1: between point1 nad point2, 2: under or right of point2
    # assert unkown_point_subsec in [0,1,2]
    p12_dist = coor_dist(point1_3d, point2_3d)
    if p12_dist <= 4000:
        DEPTH_ACCURACY = 0.02 * 4000
    else:
        DEPTH_ACCURACY = 0.02 * 4000 + 0.04 * (p12_dist - 4000)

    x1, y1, z1 = point1_3d
    x2, y2, z2 = point2_3d
    # direction vector of space line p2->p1
    dvec = (x2-x1, y2-y1, z2-z1)

    if abs(dvec[0]) > DEPTH_ACCURACY and abs(dvec[1]) > DEPTH_ACCURACY:

        if (unkown_point_inc[0] * dvec[0]) > 0:
            t = point1_to_unknown_3dist / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        else:
            t = -1 * point1_to_unknown_3dist / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            
        x0 = x1 + t * (x2 - x1)
        y0 = y1 + t * (y2 - y1)
        z0 = z1 + t * (z2 - z1)
    elif abs(dvec[0]) <= DEPTH_ACCURACY:
        x0 = x1
        if unkown_point_inc[1] > 0:
            y0 = y1 + point1_to_unknown_3dist / np.sqrt(((z2 - z1) / (y2 - y1)) ** 2 + 1)
        else:
            y0 = y1 - point1_to_unknown_3dist / np.sqrt(((z2 - z1) / (y2 - y1)) ** 2 + 1)
        z0 = z1 + ((y0 - y1) / (y2 - y1)) * (z2 - z1)
    elif abs(dvec[1]) <= DEPTH_ACCURACY:
        y0 = y1
        if unkown_point_inc[0] > 0:
            x0 = x1 + point1_to_unknown_3dist / np.sqrt(((z2 - z1) / (x2 - x1)) ** 2 + 1)
        else:
            x0 = x1 - point1_to_unknown_3dist / np.sqrt(((z2 - z1) / (x2 - x1)) ** 2 + 1)
        z0 = z1 + ((x0 - x1) / (x2 - x1)) * (z2 - z1)
        
    return [x0, y0, z0]    



"""
sides_vex, [[row-number, col-number], [], [], []]
"""
def calculate_sides_depth(vertex_pixels, vertex_points, camera_intrin, milli_step=6):
    sv_range = np.arange(len(vertex_pixels))
    id_pairs = gen_pairs(sv_range)
   
    sides_pixels = [[] for i in range(len(vertex_pixels))]
    sides_points = [[] for i in range(len(vertex_pixels))]
    for sid in range(len(vertex_pixels)):
        i1, i2 = id_pairs[sid]
        v12_pixel_dist = np.sqrt(np.sum((vertex_pixels[i1]-vertex_pixels[i2])**2))
        assert v12_pixel_dist > 0, "line length incorrect:{}".format(v12_pixel_dist)

        # append start point
        sides_pixels[sid].append(vertex_pixels[i1])
        sides_points[sid].append(vertex_points[i1])

        inline_sides_points = inline_points_interpolation(vertex_points[i1], vertex_points[i2], is_millimeter=True, milli_step=milli_step, to_meter=False, sideno=sid)
        inline_sides_pixels = project_to_pixel(inline_sides_points, camera_intrin)

        sides_points[sid] += inline_sides_points
        sides_pixels[sid] += inline_sides_pixels
        # append end point
        sides_pixels[sid].append(vertex_pixels[i2])
        sides_points[sid].append(vertex_points[i2])
    return sides_pixels, sides_points


def list_split(data_list, split_num):
    d_num = len(data_list)
    per_proc_pnum = d_num // split_num
    plist = [data_list[i*per_proc_pnum:(i+1)*per_proc_pnum] for i in range(split_num)]
    if d_num % split_num > 0:
        plist[-1] += data_list[per_proc_pnum*split_num:]
    return plist

def sample_points(sides_points, sample_ratio=0.1, min_side_inter_point_num=2):
    sampled_points = []
    # sample example points firstly
    for sp in sides_points:
        sampled_points.append(sp[0])
    for side_ps in sides_points:
        pnum = max(int((len(side_ps) - 2) * sample_ratio), min_side_inter_point_num)
        step = max(round(len(side_ps) / (pnum + 1)), 1)
        sampled_points += side_ps[1:-2:step]

    return sampled_points


def calculate_region_depth(sides_pixels, sides_points, camera_intrin, compl_depth_mat=None, milli_step=6, sample_ratio=0.2):
    assert len(sides_points) >= 3, 'polygon points num shoud be at least 3'
    assert len(sides_pixels) == len(sides_points)
    sides_points_num = [len(s) for s in sides_points]
    start_points = sample_points(sides_points, sample_ratio=sample_ratio, min_side_inter_point_num=100)
    end_points = []
    # for i, ps in enumerate(sides_points[1:]):
    for i, ps in enumerate(sides_points):
        end_points += ps
    # print('start_points num:', len(start_points), 'end_points num:', len(end_points))
    region_pixels_dict = {}

    end_points = np.array(end_points)
    inline_time = 0
    project_time = 0
    for start_point in tqdm(start_points, desc='start points:{}, end points:{}'.format(len(start_points), len(end_points))):
    # for start_point in end_points:
        for end_point in end_points:
            inline_stime = time.time()
            inline_points = inline_points_interpolation(start_point, end_point, is_millimeter=True, milli_step=milli_step, to_meter=True)
            proj_stime = time.time()
            inline_time += proj_stime - inline_stime
            inline_pixels = project_to_pixel(inline_points, camera_intrin, unit_meter=True)
            project_time += time.time() - proj_stime
            if len(inline_points) > 0 and len(inline_pixels) > 0:
                for one_pixel, one_point in zip(inline_pixels, inline_points):
                    column_no = int(one_pixel[0])
                    row_no = int(one_pixel[1])
                    dpth = one_point[-1]
                    pkey = '{},{}'.format(column_no, row_no)
                    if pkey in region_pixels_dict:
                        region_pixels_dict[pkey] = (region_pixels_dict[pkey] + dpth) / 2
                    else:
                        region_pixels_dict[pkey] = dpth
    for k, v in region_pixels_dict.items():
        x, y = k.split(',')
        compl_depth_mat[int(y), int(x)] = v * 1000.0

    return compl_depth_mat, (inline_time, project_time)

def region_range(start_value, end_value):
    xrange_st = min(start_value, end_value)  
    xrange_ed = max(start_value, end_value)
    region_range = list(range(xrange_st, xrange_ed))
    if start_value != xrange_st:
        region_range.reverse()
    return region_range

def interpolate_region_depth(inline_pixels_list, inline_points_list, interp=False, compl_depth_mat=None):
    x_arr = []
    y_arr = []
    d_arr = []

    if interp:
        for line_pxl, line_pnt in zip(inline_pixels_list, inline_points_list):
            for pxl, pnt in zip(line_pxl, line_pnt):
                x_arr.append(pxl[0])
                y_arr.append(pxl[1])
                d_arr.append(pnt[-1])
        x_arr = np.array(x_arr, dtype=np.float32)
        y_arr = np.array(y_arr, dtype=np.float32)
        d_arr = np.array(d_arr, dtype=np.float32)
        region_x = region_range(int(min(x_arr)), int(max(x_arr)))
        region_y = region_range(int(min(y_arr)), int(max(y_arr)))
        inter_func = interpolate.interp2d(x_arr, y_arr, d_arr, kind='linear')
        region_depth = inter_func(region_x, region_y)
    else:
        assert compl_depth_mat is not None, 'depth matrix needed when interpolation is not used.'
        coors_depth_dict = {}
        for line_pxl, line_pnt in zip(inline_pixels_list, inline_points_list):
            for pxl, pnt in zip(line_pxl, line_pnt):
                key = '{},{}'.format(int(pxl[0]), int(pxl[1]))
                if key in coors_depth_dict:
                    coors_depth_dict[key] = (coors_depth_dict[key] + pnt[-1]) / 2
                else:
                    coors_depth_dict[key] = pnt[-1]
        for k, v in coors_depth_dict.items():
            x, y = k.split(',')
            compl_depth_mat[int(y), int(x)] = v * 1000.0
        return compl_depth_mat, None

    return region_depth, (region_x, region_y)

def interplate_pixel_depth(extreme_pixels, inline_pixels_list, inline_points_list, compl_depth_mat, raw_depth_mat):
    print('extreme_pixels', extreme_pixels)

    region_x = region_range(min(extreme_pixels[:, 0]), max(extreme_pixels[:, 0]))
    region_y = region_range(min(extreme_pixels[:, 1]), max(extreme_pixels[:, 1]))

    #generate completion needed mask
    mx, my = np.meshgrid(region_x, region_y)
    coords_vertex_rectangle = np.concatenate([mx[:,:,np.newaxis], my[:,:,np.newaxis]], axis=2)
    print('coords_vertex_rectangle.shape', coords_vertex_rectangle.shape)
    s_row, s_col, _ = coords_vertex_rectangle.shape
    coors_in = []
    # meter to millimeter
    interp_pixels = []
    interp_pixels_depth = []
    unchange_pixels = []
    unchange_pixels_depth = []
    for i in range(s_row):
        for j in range(s_col):
            co = coords_vertex_rectangle[i][j]
            is_in, within_info = within_poly(extreme_pixels, [co])
            if is_in[0]:
                coors_in.append(co)
                # compl_depth_mat[co[1], co[0]] = region_depth[i, j]
                comp_dpth = compl_depth_mat[co[1], co[0]]
                raw_dpth = raw_depth_mat[co[1], co[0]]
                if comp_dpth == raw_dpth:
                    unchange_pixels.append(co)
                    unchange_pixels_depth.append(compl_depth_mat[co[1], co[0]])
                else:
                    interp_pixels.append(co)
                    interp_pixels_depth.append(compl_depth_mat[co[1], co[0]])

    interp_pixels = np.array(interp_pixels)
    interp_pixels_depth = np.array(interp_pixels_depth)
    unchange_pixels = np.array(unchange_pixels)
    for pxl in unchange_pixels:
        pxl_dist = np.sqrt(np.sum((interp_pixels - pxl) ** 2, axis=1))
        before = compl_depth_mat[pxl[1], pxl[0]]
        
        dist_ascend_order = np.argsort(pxl_dist)
        top100_idx = dist_ascend_order[0:1000]
        interp_pixels_ = interp_pixels[top100_idx]
        interp_pixels_depth_ = interp_pixels_depth[top100_idx]
        inter_func = interpolate.interp2d(interp_pixels_[:, 0], interp_pixels_[:, 1], interp_pixels_depth_, kind='linear')
        compl_depth_mat[pxl[1], pxl[0]] = inter_func(pxl[0], pxl[1])

        print('before', before, 'after', compl_depth_mat[pxl[1], pxl[0]])

    return compl_depth_mat


def interpolate_region_depth_bylines(coor, depth):
    depth_dict = {}
    for c_list, d_list in zip(coor, depth):
        for c, d in zip(c_list, d_list):
            ck = str(int(c[0]))+','+str(int(c[1]))
            if ck not in depth_dict or depth_dict[ck] <= 0:
                depth_dict[ck] = d[-1]
            else:
                depth_dict[ck] = (d[-1] + depth_dict[ck]) / 2

    return depth_dict

def fuse_region_depth(origion_depth_mat, region_depth, region_coors):
    assert len(region_depth) == 2
    # fill the zero depth place with depth value from othe depth mat(right-left to left-right)
    np.putmask(region_depth[0], region_depth[0] <= 0.0, region_depth[1])
    rx, ry = np.meshgrid(region_coors[0][0], region_coors[0][1])
    origion_depth_mat[ry, rx] = region_depth[0]
    return origion_depth_mat

def region_depth_completion(raw_depth_mat, coor_depth_dict_lr, show=False, save_dir=None, anno_id=0):
    if show:
        save_file = save_dir + '/depth_calculated_anno{}_{}.png'.format(anno_id, time.time())
        zeros_mat = np.zeros_like(raw_depth_mat)
        # assign
        for k, v in coor_depth_dict_lr.items():
            col, row = k.split(',')
            zeros_mat[int(row), int(col)] = v
   
        comp_depth = vis_depth_mat(zeros_mat, zeros_mat.shape[0], zeros_mat.shape[1], plt_show=False)
        cv2.imshow('calculated_depth', comp_depth)
        cv2.imwrite(save_file, comp_depth)
        # raw_depth_mat_color = vis_depth_mat(raw_depth_mat, raw_depth_mat.shape[0], raw_depth_mat.shape[1], plt_show=False)
        # cv2.imshow('filled_depth01', raw_depth_mat_color)


    # assign
    for k, v in coor_depth_dict_lr.items():
        col, row = k.split(',')
        raw_depth_mat[int(row), int(col)] = v
    
    if show:
        save_file = save_dir + '/depth_completion_anno{}_{}.png'.format(anno_id, time.time())
        a = vis_depth_mat(raw_depth_mat, raw_depth_mat.shape[0], raw_depth_mat.shape[1], plt_show=False)
        cv2.imwrite(save_file, a)
        cv2.imshow('region_dist_completion', a)
        # if cv2.waitKeyEx() == 27:
        #     cv2.destroyAllWindows()
        # exit()


    return raw_depth_mat

def read_json_label(json_label, key=None):
    with open(json_label) as f:
        annos = json.load(f)
        if key is not None:
            assert key in annos, 'key {} not exists in keys:{}'.format(key, str(annos.keys()))
            return annos[key]
        else:
            return annos

def read_camera_intrin(intrin_json):
    intrin_dict = read_json_label(intrin_json)

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


def deproject_to_points(pixel_points, pixel_depths, intrin_obj, depth_scale=0.001):
    world_coors = []
    for pixel_coor, pixel_dpth in zip(pixel_points, pixel_depths):
        coor3d = rs.rs2_deproject_pixel_to_point(intrin_obj, pixel_coor, pixel_dpth * depth_scale)
        world_coors.append(coor3d)
    return world_coors

def project_to_pixel(points, intrin_obj, unit_meter=True):
    assert unit_meter, 'points coorination must be in meter.'
    pixel_coors = []
    for p in points:
        pixel_coor = rs.rs2_project_point_to_pixel(intrin_obj, p)
        pixel_coors.append(pixel_coor)
    return pixel_coors

def inline_points_interpolation(start_point, end_point, is_millimeter=True, scale_to_millimeter=0.001, milli_step = 1, to_meter=False, sideno=0):
    point_dist = coor_dist(start_point, end_point)
    points_interpolated = []
    if point_dist <= 0.0:
        return points_interpolated
    
    
    dist_millimeter = point_dist / scale_to_millimeter if not is_millimeter else point_dist
    dire_vec = end_point - start_point
    cosin_x = dire_vec[0] / point_dist
    cosin_y = dire_vec[1] / point_dist
    cosin_z = dire_vec[2] / point_dist

    for inc in range(1, int(dist_millimeter), milli_step):
        x = inc * cosin_x  + start_point[0]
        y = inc * cosin_y  + start_point[1]
        z = inc * cosin_z  + start_point[2]
        if to_meter and is_millimeter:
            x = x * 0.001
            y = y * 0.001
            z = z * 0.001
            
        points_interpolated.append([x, y, z])
    return points_interpolated

def coor_dist(world_coor1, world_coor2):
    assert len(world_coor1) == len(world_coor2), (world_coor1, world_coor2)
    assert len(world_coor1) in [2, 3], world_coor1
    # return np.sqrt(np.sum((world_coor1 - world_coor2)**2))
    if len(world_coor1) == 3:
        return np.sqrt((world_coor1[0] - world_coor2[0]) ** 2 +
                (world_coor1[1] - world_coor2[1]) ** 2 +
                (world_coor1[2] - world_coor2[2]) ** 2)
    else:
        return np.sqrt((world_coor1[0] - world_coor2[0]) ** 2 +
                (world_coor1[1] - world_coor2[1]) ** 2)

def get_norm_vec(points):
    vec1 = points[0] - points[1]
    vec2 = points[0] - points[2]
    normal_vec = (vec1[1] * vec2[2] - vec1[2] * vec2[1], vec1[2] * vec2[0] - vec1[0] * vec2[2], vec1[0] * vec2[1] - vec1[1] * vec2[0])
    return normal_vec

def get_point_by_dist_in_pane(pixel_coors, points_coors, points_depth, is_depth_valid, sides_len):
    plane_norm = get_norm_vec(points_coors[is_depth_valid == 1])
    vex_num = len(pixel_coors)
    for i, d in enumerate(is_depth_valid):
        if d == 0:
            if i == 0:
                p1 = points_coors[vex_num - 1]
                p2 = points_coors[i + 1]
                d1 = sides_len[vex_num - 1]
                d2 = sides_len[i]
            elif i == vex_num - 1:
                p1 = points_coors[i - 1]
                p2 = points_coors[0]
                d1 = sides_len[i - 1]
                d2 = sides_len[i]
            else:
                p1 = points_coors[i - 1]
                p2 = points_coors[i + 1]
                d1 = sides_len[i - 1]
                d2 = sides_len[i]
            
            L, M, N  = plane_norm
            P = (M / L) ** 2 + 1
            Q = (N / L) ** 2 + 1
            R = (2 * M * N) / (L ** 2)
            X1, Y1, Z1 = p1
            X2, Y2, Z2 = p2
            const_yz = P * (Y1 ** 2 - Y2 ** 2) + Q * (Z1 ** 2 - Z2 ** 2) + R * (Y1 * Z1 - Y2 * Z2)
            y_denom_const = 2 * P * (Y2 - Y1) + R * (Z2 - Z1)
            y_numer_const = d1 ** 2 - d2 ** 2 - const_yz
            z_numer_coeff = 2 * Q * (Z2 - Z1) + R * (Y2 - Y1)
            A = y_numer_const / y_denom_const
            B = z_numer_coeff / y_denom_const
            # print('P:', P, ' Q:', Q, ' R:', R, ' B:', B, 'p1-p2', (p1, p2))
            U = P * (B ** 2) + Q - R * B

            V = R * A - 2 * P * A * B + B * (2 * P * Y1 + R * Z1) - (2 * Q * Z1 + R * Y1)
            W = P * (A ** 2) - A * (2 * P * Y1 + R * Z1) + P * (Y1 ** 2) + Q * (Z1 ** 2) + R * Y1 * Z1 - d1 ** 2
            # print('U:', U, ' V:', V, ' W:', W)
            # print('v2 4uw', V ** 2 - 4 * U * W)

            z_solu1 = (-1 * V + np.sqrt(V ** 2 - 4 * U * W)) / (2 * U)
            y_solu1 = A - z_solu1 * B
            x_solu1 = X1 + (N * (Z1 - z_solu1) - M * (y_solu1 - Y1)) / L

            z_solu2 = (-1 * V - np.sqrt(V ** 2 - 4 * U * W)) / (2 * U)
            y_solu2 = A - z_solu2 * B
            x_solu2 = X1 + (N * (Z1 - z_solu2) - M * (y_solu2 - Y1)) / L
            
            x = x_solu1
            y = y_solu1
            z = z_solu1
            if i == 0 and y_solu2 < y_solu1:
                x = x_solu2
                y = y_solu2
                z = z_solu2
            elif i == 1 and y_solu2 > y_solu1:
                x = x_solu2
                y = y_solu2
                z = z_solu2

            elif i == 2 and y_solu2 > y_solu1:
                x = x_solu2
                y = y_solu2
                z = z_solu2
            elif i == 3 and y_solu2 < y_solu1:
                x = x_solu2
                y = y_solu2
                z = z_solu2

            points_coors[i][0] = x
            points_coors[i][1] = y
            points_coors[i][2] = z
            points_depth[i] = z * 1000 # meter to milimeter

    return points_coors, points_depth

def check_depth(depth_raw_file, json_camera_intri, json_label,):
    # raw_dpth_mat = read_raw_depth(width, height, depth_raw_file)
    raw_dpth_mat = read_depth_npy(depth_raw_file)
    raw_dpth_mat = raw_dpth_mat.astype(np.uint16)
    poly_shapes = read_json_label(json_label, key='shapes')
    camera_intrin, depth_scale = read_camera_intrin(json_camera_intri)
    
    # collect points firstly
    depth_point_dict = {}
    for i, ann in enumerate(poly_shapes):
        if ann['shape_type'] == 'point':
            depth_point_dict[ann['label']] = np.array(ann['points'], dtype=np.int32)

    for idx, ann in enumerate(poly_shapes):
        if ann['shape_type'] == 'rectangle' and ann['label'] in ['delete', 'point']:
            continue
        if ann['shape_type'] != 'polygon':
            continue

        sides_vex_dpth, _ = get_depth_by_pixel_coordinates(ann, json_label, idx, raw_dpth_mat, depth_point_dict)
        
        # if len(label_list) > 1:
        #     is_depth_valid = np.array([int(li) for li in list(str(label_list[1]))])
        #     sides_vex_depth_valid = sides_vex[is_depth_valid == 1]

        #     sides_vex_points = deproject_to_points(sides_vex, sides_vex_dpth, camera_intrin, depth_scale=depth_scale)
        #     sides_vex_points = np.array(sides_vex_points, dtype=np.float32)

        #     if len(sides_vex_depth_valid) != len(sides_vex):
        #         print('anno:', idx, 'label_list:', label_list, 'origin sides_vex_points', sides_vex_points)
        #         assert len(label_list) == 3, 'label {} invalid'.format(ann['points'])
        #         sides_len = np.array([int(l) for l in label_list[2].split(',')])
        #         sides_len = sides_len / 100.0 # centemeter to meter
        #         print('sides_len meter:', sides_len)

        #         assert len(sides_len) == len(sides_vex), 'label {} measured sizes uncompleted'.format(ann['points'])
        #         sides_vex_points, sides_vex_dpth = get_point_by_dist_in_pane(sides_vex, sides_vex_points, sides_vex_dpth, is_depth_valid, sides_len)
        #         print('anno:', idx, 'new points_coors', sides_vex_points)
        # else:
        #     is_depth_valid = np.ones(len(sides_vex))
        #     sides_vex_depth_valid = sides_vex
        print('sides_vex_dpth', sides_vex_dpth)
        
        assert np.prod(sides_vex_dpth) > 0, 'anno:{}/{} zero depth found {}'.format(idx+1, len(poly_shapes), str(sides_vex_dpth))

def get_depth_by_pixel_coordinates(ann, json_label, idx, raw_dpth_mat, depth_point_dict):
    label_list = ann['label'].split('-')
    assert len(label_list) > 0, '{},{}'.format(json_label, str(label_list))
    sides_vex = np.array(ann['points'], dtype=np.float32)
    assert len(sides_vex) >= 3, 'anno:{},sides_vex:{}'.format(idx, str(sides_vex))
    sides_vex = np.floor(sides_vex.reshape(-1, 2)).astype(np.int32)

    sides_vex_dpth = get_depth_from_mat(raw_dpth_mat, sides_vex)  

    # complete zero depth with selected depth valid points (generally close to corresponding vertex)
    if len(label_list) > 1:
        for id, dpth in enumerate(sides_vex_dpth):
            if ann['shape_type'] == 'polygon':
                prefix = 'p' if label_list[0] in GLASS_LABELS else 'f'
            else:
                continue
            # pnt_key = "{}-wall{}-{}".format(prefix, label_list[1], id)
            pnt_key = "{}-{}{}-{}".format(prefix, label_list[0], label_list[1], id)
            # assert pnt_key in depth_point_dict, '{} have to valid points selected'.format(pnt_key)
            if pnt_key in depth_point_dict:
                close_depth = get_depth_from_mat(raw_dpth_mat, depth_point_dict[pnt_key])
                sides_vex_dpth[id] = close_depth
    
    return sides_vex_dpth, sides_vex

def draw_generated_sides_points(genr_pixels, depth_mat, png_img=None):
    if png_img is None:
        png_img = vis_depth_mat(depth_mat)
    
    sides_color = [(255, 0, 0),(0, 0, 255),(255, 0, 255), (0, 255, 255)]
    for i, side_pxls in enumerate(genr_pixels):
        print('side no-{}, num:{}'.format(i, len(side_pxls)))
        for pxl in side_pxls:
            cv2.circle(png_img, (int(pxl[0]), int(pxl[1])), radius=1, color=sides_color[i], thickness=1)
    cv2.imshow('generated sides points', png_img)
    if cv2.waitKeyEx() == 27:
        cv2.destroyAllWindows()
        
def depth_completion(depth_raw_file, json_camera_intri, json_label, save_dir, vis_save_dir=None, milli_step=6, sample_ratio=0.2):
    raw_dpth_mat = read_depth_npy(depth_raw_file)
    raw_dpth_mat = raw_dpth_mat.astype(np.uint16)
    fname = os.path.basename(depth_raw_file).split('.')[0]
    poly_shapes = read_json_label(json_label, key='shapes')
    camera_intrin, depth_scale = read_camera_intrin(json_camera_intri)
    
    show_depth_comp = True

    # collect points firstly
    depth_point_dict = {}
    for i, ann in enumerate(poly_shapes):
        if ann['shape_type'] == 'point':
            depth_point_dict[ann['label']] = np.array(ann['points'], dtype=np.int32)

    compl_depth_mat = raw_dpth_mat.copy()
    for idx, ann in enumerate(poly_shapes):
        tstart = time.time()
        if ann['shape_type'] == 'rectangle' and ann['label'] in ['delete']:
            continue
        
        if ann['shape_type'] != 'polygon':
            continue


        sides_vex_dpth, vertex_pixels = get_depth_by_pixel_coordinates(ann, json_label, idx, raw_dpth_mat, depth_point_dict)
        sides_vex_points = deproject_to_points(vertex_pixels, sides_vex_dpth, camera_intrin, depth_scale=depth_scale)
        sides_vex_points = np.array(sides_vex_points, dtype=np.float32)
        assert np.prod(sides_vex_dpth) > 0, 'zero depth found:'+str(sides_vex_dpth)

        # label_list = ann['label'].split('-')
        # if len(label_list) > 1:
        #     is_depth_valid = np.array([int(li) for li in list(str(label_list[1]))])
        #     sides_vex_depth_valid = vertex_pixels[is_depth_valid == 1]

        #     if len(sides_vex_depth_valid) != len(vertex_pixels):
        #         assert len(label_list) == 3, 'label {} invalid'.format(ann['points'])
        #         sides_len_org = np.array([int(l) for l in label_list[2].split(',')])
        #         sides_len_meter = sides_len_org / 100.0 # centemeter to meter
        #         assert len(sides_len_meter) == len(vertex_pixels), 'label {} measured sizes uncompleted'.format(ann['points'])
        #         sides_vex_points, sides_vex_dpth = get_point_by_dist_in_pane(vertex_pixels, sides_vex_points, sides_vex_dpth, is_depth_valid, sides_len_meter)
        #         print('anno:', idx, 'new points_coors', sides_vex_points)
        # else:
        #     is_depth_valid = np.ones(len(vertex_pixels))
        #     sides_vex_depth_valid = vertex_pixels

        sides_vex_points /= depth_scale
        # calculate sides depth
        sides_pixels, sides_points = calculate_sides_depth(vertex_pixels, sides_vex_points, camera_intrin, milli_step=milli_step)
        compl_depth_mat, time_used = calculate_region_depth(sides_pixels, sides_points, camera_intrin, compl_depth_mat, 
        milli_step=milli_step, sample_ratio=sample_ratio)

        print("anno {}, {:3.3f} seconds used for inline, {:3.3f} seconds used for project.".format(idx, time_used[0], time_used[1]))
        # region_depth is a list with X columns and Y rows,
        # region range has X coordinations respect to x axis, and Y coordinations respect to y axis
        inter_mat = False
        if inter_mat:
            #generate completion needed mask
            mx, my = np.meshgrid(region_range[0], region_range[1])

            coords_vertex_rectangle = np.concatenate([mx[:,:,np.newaxis], my[:,:,np.newaxis]], axis=2)
            s_row, s_col, _ = coords_vertex_rectangle.shape
            coors_in = []
            # meter to millimeter
            region_depth = region_depth * 1000.0
            for i in range(s_row):
                for j in range(s_col):
                    co = coords_vertex_rectangle[i][j]
                    is_in, within_info = within_poly(vertex_pixels, [co])
                    if is_in[0]:
                        coors_in.append(co)
                        compl_depth_mat[co[1], co[0]] = region_depth[i, j]
            assert len(co) > 0, 'zeros coors in vertex polygon {}'.format(len(co))

        if show_depth_comp and vis_save_dir is not None:
            comp_depth_vis = vis_depth_mat(compl_depth_mat, plt_show=False)
            cv2.imwrite(os.path.join(vis_save_dir, 'depth_calculated_anno{}_{}.png'.format(idx, time.time())), comp_depth_vis)
            # for i, dpm in enumerate(dpth_mats):
            #     dpm_vis = vis_depth_mat(dpm, plt_show=False)
            #     cv2.imwrite(os.path.join(vis_save_dir, 'depth_calculated_anno{}_process{}_{}.png'.format(idx, i, time.time())), dpm_vis)

        # if show_sides_lines:
        #     raw_dpth_color = vis_depth_mat(raw_dpth_mat, plt_show=False)
        #     sides_color = [(255, 0, 0),(0, 0, 255),(255, 0, 255), (0, 255, 255)]
        #     for sid in range(len(sides_vex)):
        #         if sid != len(sides_vex) - 1:
        #             cv2.line(raw_dpth_color, sides_vex[sid], sides_vex[sid+1], color=sides_color[sid], thickness=2)
        #         else:
        #             cv2.line(raw_dpth_color, sides_vex[sid], sides_vex[0], color=sides_color[sid], thickness=2)
        #     cv2.imwrite(os.path.join(vis_save_dir, 'depth_completion_anno{}_{}.png'.format(idx, time.time())), raw_dpth_color)
        print("anno:{}, {} minutes used.".format(fname, idx, (time.time() - tstart) / 60))


    save_name = os.path.join(save_dir, fname+'_filled.npy')
    np.save(save_name, compl_depth_mat)
    print('anno:{} saved to {}\n'.format(fname, save_name))

    if vis_save_dir is not None:
        raw_depth_mat_vis = vis_depth_mat(compl_depth_mat, plt_show=False)
        cv2.imwrite(os.path.join(vis_save_dir, 'depth_completion_{}_{}.png'.format(fname, time.time())), raw_depth_mat_vis)


def read_polyinfo(txt_file):
    pinfo_list = []
    with open(txt_file, 'r') as f:
        pinfo = f.readline()
        while pinfo:
            pinfo = pinfo.strip('\n')
            pinfo_list.append(pinfo.split(' '))
            pinfo = f.readline()
    return pinfo_list

def per_folder_gen(root_dir, intrin_json, save_dir, milli_step=6, sample_ratio=0.2):
    # root_dir = '/home/ly/data/datasets/trans-depth/temp_test/aligned_depth_color/'
    npy_save_dir = os.path.join(save_dir, 'completed_depth_npy')
    vis_save_dir = os.path.join(save_dir, 'completed_depth_vis')
    # color_dir_name = 'depth_vis'
    json_dir_name = 'depth_vis'
    color_dir_name = ''
    # json_dir_name = ''
    os.makedirs(npy_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    aux_dir = 'depth_vis'
    if len(json_dir_name) == 0:
        json_dir = root_dir
        anno_json = glob.glob(root_dir+'/*.json')
    else:
        json_dir = os.path.join(root_dir, json_dir_name)
        anno_json = glob.glob(json_dir+'/*.json')

    anno_json.sort()

    if sys.argv[1] == 'check':
        fnames_file = os.path.join(json_dir, 'fnames_all.txt')
        f_names = open(fnames_file, 'w+')
        for i, anno in enumerate(anno_json):
            fname = os.path.basename(anno).split('.')[0]
            color_png = os.path.join(root_dir, color_dir_name, fname+'.png')
            depth_npy = os.path.join(root_dir, fname+'.npy')
            poly_json = anno
            print("checking color_png:", color_png)
            check_depth(depth_npy, intrin_json, poly_json)
            f_names.write(fname+'\n')
        f_names.close()
        print("Depth check done! {} checked files saved in{}".format(len(anno_json), fnames_file))
    else:
        ##################start calculate depth map######################
        if len(sys.argv) > 2:
            fnames_run = os.path.join(json_dir, sys.argv[2])
        else:
            fnames_run = os.path.join(json_dir, 'fnames_run.txt')

        f_run = open(fnames_run, 'r')
        run_names = f_run.readlines()
        run_names = [rn.strip() for rn in run_names]
        run_num = len(run_names)
        print("{} Files to make depth completion:{}".format(run_num, run_names))
        anid = 1
        for anno in anno_json:
            fname = os.path.basename(anno).split('.')[0]
            if fname not in run_names:
                continue
            color_png = os.path.join(root_dir, fname+'.png')
            depth_npy = os.path.join(root_dir, fname+'.npy')
            poly_json = anno
            print("runing image {}/{}, {}".format(anid, run_num, color_png))
            vis_save_dir_f = os.path.join(vis_save_dir, fname)
            if not os.path.isdir(vis_save_dir_f):
                os.mkdir(vis_save_dir_f)

            depth_completion(depth_npy, intrin_json, poly_json, npy_save_dir, vis_save_dir=vis_save_dir_f, 
                milli_step=milli_step, sample_ratio=sample_ratio)
            anid += 1
            print()
        f_run.close()

def main_gen(base_save_dir, base_dir='', milli_step=6, sample_ratio=0.2):
    assert len(sys.argv) > 1, "choose check or run in command line"
    run_type = sys.argv[1]
    assert run_type in ['check', 'run']
    os.makedirs(base_save_dir, exist_ok=True)

    # folders = ['20210927', '20210928', '20210929', '20211002', '20211004', '20211016', '20211017', '20211018',
    #     '20211019', '20211020', '20211021', '20211027', '20211028', '20211031', '20211104']
    # folders = ['20211105', '20211107', '20211111', '20220430', '20220514']
    # folders = ['20220821']
    folders = ['20220514']
    intrin_json = '/home/ly/data/datasets/trans-depth/d455_intrin.json'
    root_dir_format = '/home/ly/data/datasets/trans-depth/{}/{}/aligned_depth_color/'
    

    for folder_name in folders:
        root_dir = root_dir_format.format(base_dir, folder_name)
        save_dir = os.path.join(base_save_dir, folder_name)
        per_folder_gen(root_dir, intrin_json, save_dir, milli_step=milli_step)
    

if __name__ == '__main__':
    base_save_dir = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Test'
    milli_step=6
    sample_ratio=0.5
    # base_save_dir = '/home/ly/data/datasets/trans-depth/Opaque-Test/millistep{}_sampleratio{}'.format(string(milli_step), string(sample_ratio))
    main_gen(base_save_dir, base_dir='', milli_step=milli_step, sample_ratio=sample_ratio)