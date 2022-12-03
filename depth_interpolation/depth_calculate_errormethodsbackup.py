import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import cv2
import glob
import pyrealsense2 as rs

from read_binfile import depth_from_mat, read_depth_npy, vis_depth_mat
import json
import multiprocessing as mp

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


def gen_pairs(np_vector):
    d0 = np_vector[:, np.newaxis]
    d1 = np_vector[1:].tolist()
    d1.append(np_vector[0])
    d1 = np.array(d1)[:, np.newaxis]
    d_pairs = np.concatenate((d0, d1), axis=1)
    return d_pairs

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
    # print('point1_3d', point1_3d, ' point2_3d', point2_3d, 'dvec', dvec, ' p12_dist:', p12_dist, ' DEPTH_ACCURACY:', DEPTH_ACCURACY)
    # assert not (abs(dvec[0]) <= DEPTH_ACCURACY and abs(dvec[1]) <= DEPTH_ACCURACY), str([point1_3d, point2_3d, dvec])

    if abs(dvec[0]) > DEPTH_ACCURACY and abs(dvec[1]) > DEPTH_ACCURACY:
        # t = point1_to_unknown_3dist / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        # if unkown_point_inc[0] > 0:
        #     x0 = x1 + t * (x2 - x1)
        # else:
        #     x0 = x1 - t * (x2 - x1)
        
        # if unkown_point_inc[1] > 0:
        #     y0 = y1 + t * (y2 - y1)
        # else:
        #     y0 = y1 - t * (y2 - y1)
        # z0 = z1 + ((y0 - y1)/(y2 - y1)) * (z2 - z1)

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

    # ratio_unkwn_to_p12 = point1_to_unknown_3dist/p1_to_p2_3dist
    # p1_to_unkwn_2d_dist = p1_to_p2_2dist * ratio_unkwn_to_p12

    # d_1u = p1_to_unkwn_2d_dist

    # if abs(direc_vec[0]) <= MIN_VALUE and abs(direc_vec[1]) > MIN_VALUE:
    #     x0 = 0.0
    #     if unkown_point_inc[1] > 0: # new point has y value increment, so y0>y1
    #         y0 = y1 + np.sqrt(d_1u ** 2 - x1 ** 2)
    #     else:
    #         y0 = y1 - np.sqrt(d_1u ** 2 - x1 ** 2)
    # elif abs(direc_vec[0]) > MIN_VALUE and abs(direc_vec[1]) <= MIN_VALUE:
    #     y0 = 0.0
    #     if unkown_point_inc[0] > 0: # new point has x value increment, so x0>x1
    #         x0 = x1 + np.sqrt(d_1u ** 2 - y1 ** 2)
    #     else:
    #         x0 = x1 - np.sqrt(d_1u ** 2 - y1 ** 2)
    # else:
    #     slope_inv = (x1-x2) / (y1-y2)
    #     if unkown_point_inc[1] > 0:
    #         y0 = y1 + d_1u / np.sqrt(slope_inv ** 2 + 1)
    #     else:
    #         y0 = y1 - d_1u / np.sqrt(slope_inv ** 2 + 1)

    #     x0 = x1 - slope_inv * (y1 - y0)
    
    # # because the smaller angle between unknown point to origin point is always not big than 90 degree, so the cosin value is not less than zero
    # unkown_to_o_2d_dist = coor_dist(np.array([x0, y0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32))
    # cosin_ukn2plane = unkown_to_o_2d_dist / unkown_to_o_3d_dist
    # if cosin_ukn2plane <= MIN_VALUE: # when the connection of unkown point to origin point is perpendicular to z plane, the unkown_to_o_3d_dist is the depth value
    #     ukn_depth = unkown_to_o_3d_dist
    # else:
    #     ukn_depth = np.sqrt(unkown_to_o_2d_dist ** 2 + unkown_to_o_3d_dist ** 2 - 2 * unkown_to_o_2d_dist * unkown_to_o_3d_dist *  cosin_ukn2plane)
    # print((x0, y0), ' unkown_point_inc:', unkown_point_inc, 'unkown_to_o_2d_dist', unkown_to_o_2d_dist, 'unkown_to_o_3d_dist', unkown_to_o_3d_dist, 'cosin_ukn2plane', cosin_ukn2plane, 'ukn_depth', ukn_depth)
    # if cosin_ukn2plane > 1:
    #     print('(x1, y1)', (x1, y1), '(x2, y2)', (x2, y2),'direc_vec', direc_vec)
    #     exit()

    # print('unkown_to_o_3d_dist', unkown_to_o_3d_dist, '(x1, y1):', (x1, y1), ' (x2, y2):', (x2, y2), ' (x0, y0):', (x0, y0))
    # ukn_depth = np.sqrt(unkown_to_o_3d_dist ** 2 - x0 ** 2 - y0 ** 2)
    # return ukn_depth


"""
sides_vex, [[row-number, col-number], [], [], []]
"""
def calculate_sides_depth(sides_vex_2d_coors, sides_vex_3d_coors, sides_lengths):
    sv_range = np.arange(len(sides_vex_2d_coors))
    id_pairs = gen_pairs(sv_range)
   
    sides_points_2d_coors = [[] for i in range(len(sides_vex_2d_coors))]
    sides_points_3d_coors = [[] for i in range(len(sides_vex_2d_coors))]
    # sides_points_2d_dist2vex = [[] for i in range(len(sides_vex_2d_coors))]
    #[0 1 2 3]
    for sid in range(len(sides_vex_2d_coors)):
        i1, i2 = id_pairs[sid]
        v12_pixel_dist = np.sqrt(np.sum((sides_vex_2d_coors[i1]-sides_vex_2d_coors[i2])**2))
        assert v12_pixel_dist > 0, "line length incorrect:{}".format(v12_pixel_dist)

        # append start point
        sides_points_2d_coors[sid].append(sides_vex_2d_coors[i1])
        sides_points_3d_coors[sid].append(sides_vex_3d_coors[i1])
        # sides_points_2d_dist2vex[sid].append(0.0)

        new_pixel_list, new_pixel_dist2start_list, increment_units = dist_completion(sides_vex_2d_coors[i1], 
                                                                                sides_vex_2d_coors[i2], sides_lengths[sid])
        # sides_points_2d_dist2vex[sid] += new_pixel_dist2start_list
        
        for newp_dist2start, inc_col_row in zip(new_pixel_dist2start_list, increment_units):
            newp_3d_coor = calculate_depth_by_collinear(sides_vex_3d_coors[i1], sides_vex_3d_coors[i2], newp_dist2start, unkown_point_inc=inc_col_row)
            sides_points_3d_coors[sid].append(newp_3d_coor)

        sides_points_2d_coors[sid] += new_pixel_list

        # append end point
        sides_points_2d_coors[sid].append(sides_vex_2d_coors[i2])
        sides_points_3d_coors[sid].append(sides_vex_3d_coors[i2])
        # sides_points_2d_dist2vex[sid].append(v12_pixel_dist*1.0)

    sides_points_2d_coors_dict = {'left':sides_points_2d_coors[0], 'bottom':sides_points_2d_coors[1], 'right':sides_points_2d_coors[2], 'top':sides_points_2d_coors[3]}
    sides_points_3d_coors_dict = {'left':sides_points_3d_coors[0], 'bottom':sides_points_3d_coors[1], 'right':sides_points_3d_coors[2], 'top':sides_points_3d_coors[3]}

    return sides_points_2d_coors_dict, sides_points_3d_coors_dict

def dist_completion(start_coor, end_coor, se_mmeter_dist):
    assert se_mmeter_dist > 0, se_mmeter_dist
    # se_pixel_dist = np.sqrt(np.sum((start_coor-end_coor)**2))
    se_pixel_dist = np.sqrt((start_coor[0] - end_coor[0]) ** 2 + (start_coor[1] - end_coor[1]) ** 2)
    mmeter_per_piexl = se_mmeter_dist / se_pixel_dist

    # cosinLR = (se_mmeter_dist ** 2 + start_dist ** 2 - end_dist ** 2) / (2 * se_mmeter_dist * start_dist)
    # increment length of sides points need line angle(alpha) corresponding to horizontal line
    cosinalpha = (end_coor[0] - start_coor[0]) / se_pixel_dist
    sinalpha = (end_coor[1] - start_coor[1]) / se_pixel_dist
    new_pixel_list = []
    # new_p_dist_list = []
    new_pixel_dist2start_list = []
    increment_units = []
    for t in range(1, int(se_pixel_dist)):
        column_inc = 1.0 * t * cosinalpha # increment along column
        row_inc = 1.0 * t * sinalpha # increment along row
        new_p = [start_coor[0] + column_inc, start_coor[1] + row_inc]                
        new_p_len = mmeter_per_piexl * (t + 1)
        # new_p_dist = np.sqrt(start_dist ** 2 + new_p_len ** 2 - 2 * start_dist * new_p_len * cosinLR)
        new_pixel_list.append(new_p)
        # new_p_dist_list.append(new_p_dist)
        new_pixel_dist2start_list.append(new_p_len)
        increment_units.append([column_inc, row_inc])
    
    return new_pixel_list, new_pixel_dist2start_list, increment_units

def depth_completion_inline(start_pixel, end_pixel, start_point, end_point, se_mmeter_dist):
    assert se_mmeter_dist > 0, se_mmeter_dist
    se_pixel_dist = np.sqrt((start_pixel[0] - end_pixel[0]) ** 2 + (start_pixel[1] - end_pixel[1]) ** 2)
    mmeter_per_piexl = se_mmeter_dist / se_pixel_dist

    # cosinLR = (se_mmeter_dist ** 2 + start_dist ** 2 - end_dist ** 2) / (2 * se_mmeter_dist * start_dist)
    # increment length of sides points need line angle(alpha) corresponding to horizontal line
    cosinalpha_pixel = (end_pixel[0] - start_pixel[0]) / se_pixel_dist
    sinalpha_pixel = (end_pixel[1] - start_pixel[1]) / se_pixel_dist
    new_pixel_list = []
    new_pixel_dist2start_list = []
    increment_units = []
    for t in range(1, int(se_pixel_dist)):
        column_inc = 1.0 * t * cosinalpha_pixel # increment along column
        row_inc = 1.0 * t * sinalpha_pixel # increment along row
        new_p = [start_pixel[0] + column_inc, start_pixel[1] + row_inc]                
        new_p_len = mmeter_per_piexl * (t + 1)
        new_pixel_list.append(new_p)
        new_pixel_dist2start_list.append(new_p_len)
        increment_units.append([column_inc, row_inc])
    
    return new_pixel_list, new_pixel_dist2start_list, increment_units

def list_split(data_list, split_num):
    d_num = len(data_list)
    per_proc_pnum = d_num // split_num
    plist = [data_list[i*per_proc_pnum:(i+1)*per_proc_pnum] for i in range(split_num)]
    if d_num % split_num > 0:
        plist[-1] += data_list[per_proc_pnum*split_num:]
    return plist

def calculate_region_depth(sides_pixel_coors, sides_points_coors, completed_mat, process_num=4):
    left_pixel_coors = sides_pixel_coors['left']
    right_pixel_coors = sides_pixel_coors['right']

    left_points_coors = sides_points_coors['left']
    right_points_coors = sides_points_coors['right']

    l_pixles_list = list_split(left_pixel_coors, process_num)
    l_points_list = list_split(left_points_coors, process_num)
    
    height, width = completed_mat.shape
    # max_line_length = int(np.ceil(np.sqrt(height ** 2 + width ** 2)).item())
    # raw_arr_list = [mp.RawArray('i', len(lp) * len(right_pixel_coors) * max_line_length * 3) for lp in l_pixles_list]
    raw_arr_list = [mp.RawArray('i', height * width) for lp in l_pixles_list]

    def part_region_depth(pixels_list, points_list, raw_arr):
        dpth_mat = np.reshape(np.frombuffer(raw_arr, dtype=np.float32), (height, width))
        for i, (l_px_coor, l_pnt_coor) in enumerate(zip(pixels_list, points_list)):
            for r_px_coor, r_pnt_coor in zip(right_pixel_coors, right_points_coors):
                pnt_dist = coor_dist(l_pnt_coor, r_pnt_coor)
                new_px_list, newp_px_dist2vex, increment_units = dist_completion(l_px_coor, r_px_coor, pnt_dist)
                
                coor3ds = []
                for new_px_dist2vex, inc_col_row, new_px_coor in zip(newp_px_dist2vex, increment_units, new_px_list):
                    new_pnt_coor = calculate_depth_by_collinear(l_pnt_coor, r_pnt_coor, new_px_dist2vex, unkown_point_inc=inc_col_row)
                    col =  int(new_px_coor[0])
                    row =  int(new_px_coor[1])
                    dpth_mat[row, col] = new_pnt_coor[-1]
                    coor3ds.append(new_pnt_coor)
                    
                # if i == len(pixels_list) - 1:
                #     print('new_px_list', new_px_list)
                #     print('newp_px_dist2vex', newp_px_dist2vex)
                #     print('increment_units', increment_units)
                #     print('coor3ds', coor3ds)
                #     print()


        # return d_mat
    # queue_mats = mp.Queue()
    mul_proc = [mp.Process(target=part_region_depth, args=(l_pixles_list[i], l_points_list[i], raw_arr_list[i])) for i in range(process_num)]
    for p in mul_proc:
        p.start()
    for p in mul_proc:
        p.join()
    dpth_mats = [np.reshape(np.frombuffer(raw_arr, dtype=np.float32), (height, width)) for raw_arr in raw_arr_list]
    # dpth_mat = [dm[np.absolute(dm).sum(axis=1) > 0] for dm in dpth_mat]

    # out_mats = [queue_mats.get() for p in mul_proc]
    # for p in mul_proc:
    #     q_mat = queue_mats.get()
    #     completed_mat = np.where(q_mat > completed_mat, q_mat, completed_mat)
    
    for i, p_mat in enumerate(dpth_mats):
        # print(i, p_mat.shape)
        completed_mat = np.where(p_mat > 0, p_mat, completed_mat)
        # completed_mat = np.where(p_mat > completed_mat, p_mat, completed_mat)
        # for v in p_mat:
        #     print(v)
        #     col = int(v[0])
        #     row = int(v[1])
        #     dpth = v[2]
        #     if dpth > completed_mat[row, col]:
        #         completed_mat[row, col] = dpth

    return completed_mat, dpth_mats

def region_range(start_value, end_value):
    xrange_st = min(start_value, end_value)  
    xrange_ed = max(start_value, end_value)
    region_range = list(range(xrange_st, xrange_ed))
    if start_value != xrange_st:
        region_range.reverse()
    return region_range

def interpolate_region_depth(coor, depth):
    x_arr = []
    y_arr = []
    d_arr = []
    for c_list, d_list in zip(coor, depth):
        for c, d in zip(c_list, d_list):
            x_arr.append(c[0])
            y_arr.append(c[1])
            d_arr.append(d)
    x_arr = np.array(x_arr, dtype=np.float32)
    y_arr = np.array(y_arr, dtype=np.float32)
    d_arr = np.array(d_arr, dtype=np.float32)

    region_x = region_range(int(min(x_arr)), int(max(x_arr)))
    region_y = region_range(int(min(y_arr)), int(max(y_arr)))

    inter_func = interpolate.interp2d(x_arr, y_arr, d_arr, kind='linear')
    region_depth = inter_func(region_x, region_y)
    return region_depth, (region_x, region_y)

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
        cv2.imshow('dist_completion', a)
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

def inline_points_interpolation(start_point, end_point, scale_to_millimeter=0.001):
    point_dist = coor_dist(start_point, end_point)
    dist_millimeter = point_dist / scale_to_millimeter
    dire_vec = end_point - start_point
    cosin_x = dire_vec[0] / point_dist
    cosin_y = dire_vec[1] / point_dist
    cosin_z = dire_vec[2] / point_dist
    points_interpolated = []
    for inc in range(1, int(dist_millimeter)):
        x = inc * cosin_x * scale_to_millimeter + start_point[0]
        y = inc * cosin_y * scale_to_millimeter + start_point[1]
        z = inc * cosin_z * scale_to_millimeter + start_point[2]
        points_interpolated.append([x, y, z])
    return points_interpolated

def coor_dist(world_coor1, world_coor2):
    assert len(world_coor1) == len(world_coor2)
    assert len(world_coor1) in [2, 3]
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
    print("plane_norm:", plane_norm)
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
            
            L, M, N = plane_norm
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
            print('P:', P, ' Q:', Q, ' R:', R, ' B:', B, 'p1-p2', (p1, p2))
            U = P * (B ** 2) + Q - R * B

            V = R * A - 2 * P * A * B + B * (2 * P * Y1 + R * Z1) - (2 * Q * Z1 + R * Y1)
            W = P * (A ** 2) - A * (2 * P * Y1 + R * Z1) + P * (Y1 ** 2) + Q * (Z1 ** 2) + R * Y1 * Z1 - d1 ** 2
            print('U:', U, ' V:', V, ' W:', W)
            print('v2 4uw', V ** 2 - 4 * U * W)

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

                # print('2-s1', (x_solu1, y_solu1, z_solu1))
                # print('2-s2', (x_solu2, y_solu2, z_solu2))
                # print('A:{},B:{}'.format(A, B))
                # print("y_numer_const:{},  y_denom_const:{}".format(y_numer_const , y_denom_const), " d1:{},d2:{}, const_yz:{}".format(d1, d2, const_yz))
            elif i == 2 and y_solu2 > y_solu1:
                x = x_solu2
                y = y_solu2
                z = z_solu2
                # print('2-s1', (x_solu1, y_solu1, z_solu1))
                # print('2-s2', (x_solu2, y_solu2, z_solu2))
                # print('A:{},B:{}'.format(A, B))
                # print("y_numer_const:{},  y_denom_const:{}".format(y_numer_const , y_denom_const), " d1:{},d2:{}, const_yz:{}".format(d1, d2, const_yz))
            elif i == 3 and y_solu2 < y_solu1:
                x = x_solu2
                y = y_solu2
                z = z_solu2

            # dist_const = (p1 ** 2 - p2 ** 2).sum() - d1 ** 2 + d2 ** 2  - 2 * y * (p1[1] - p2[1])
            # print('sides_len', sides_len, 'plane_norm:', plane_norm, 'dist_const', dist_const)
            # if abs(plane_norm[0]) <= MIN_VALUE:
            #     z = (plane_norm[1] * p1[1] + plane_norm[2] * p1[2] - plane_norm[1] * y) / plane_norm[2]
            #     dist_numerator =  dist_const - 2 * z * (p1[2] - p2[2])
            #     x = dist_numerator / 2 * (p1[0] - p2[0])
            # elif abs(plane_norm[1]) <= MIN_VALUE:
            #     dist_numerator = (dist_const / (2 * (p1[0] - p2[0]))) - (plane_norm[0] * p1[0] + plane_norm[2] * p1[2]) / plane_norm[0]
            #     dist_denominator = ((p1[2] - p2[2]) / (p1[0] - p2[0])) - plane_norm[2] / plane_norm[0]
            #     z = dist_numerator / dist_denominator
            #     x = (plane_norm[0] * p1[0] + plane_norm[2] * (p1[2] - z)) / plane_norm[0]
            # else:
            #     dist_numerator = (dist_const / (2 * (p1[0] - p2[0]))) - (plane_norm[0] * p1[0] + plane_norm[2] * p1[2] - plane_norm[1] * (y - p1[1])) / plane_norm[0]
            #     dist_denominator = ((p1[2] - p2[2]) / (p1[0] - p2[0])) - plane_norm[2] / plane_norm[0]
            #     print('dist_numerator', dist_numerator)
            #     print('dist_denominator', dist_denominator)
            #     z = dist_numerator / dist_denominator
            #     x = (plane_norm[0] * p1[0] + plane_norm[2] * (p1[2] - z) - plane_norm[1] * (y - p1[1])) / plane_norm[0]
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
    
    for idx, ann in enumerate(poly_shapes):
        if ann['shape_type'] == 'rectangle' and ann['label'] in ['delete']:
            continue
        label_list = ann['label'].split('-')
        assert len(label_list) > 0, '{},{}'.format(json_label, str(label_list))
        sides_vex = np.array(ann['points'], dtype=np.float32)
        assert len(sides_vex) == 4, 'anno:{},sides_vex:{}'.format(idx, str(sides_vex))
        # sides_vex = readjust_coordinates(sides_vex.reshape(-1, 2), vertical_line=False)
        sides_vex = np.floor(sides_vex.reshape(-1, 2)).astype(np.int32)
        # sides_vex = sides_vex.reshape(-1, 2)

        sides_vex_dpth = get_depth_from_mat(raw_dpth_mat, sides_vex)  
        sides_vex_world_coors = deproject_to_points(sides_vex, sides_vex_dpth, camera_intrin, depth_scale=depth_scale)
        sides_vex_world_coors = np.array(sides_vex_world_coors, dtype=np.float32)
        
        if len(label_list) > 1:
            is_depth_valid = np.array([int(li) for li in list(str(label_list[1]))])
            sides_vex_depth_valid = sides_vex[is_depth_valid == 1]

            if len(sides_vex_depth_valid) != len(sides_vex):
                print('anno:', idx, 'label_list:', label_list, 'origin sides_vex_world_coors', sides_vex_world_coors)
                assert len(label_list) == 3, 'label {} invalid'.format(ann['points'])
                sides_len = np.array([int(l) for l in label_list[2].split(',')])
                sides_len = sides_len / 100.0 # centemeter to meter
                print('sides_len meter:', sides_len)

                assert len(sides_len) == len(sides_vex), 'label {} measured sizes uncompleted'.format(ann['points'])
                sides_vex_world_coors, sides_vex_dpth = get_point_by_dist_in_pane(sides_vex, sides_vex_world_coors, sides_vex_dpth, is_depth_valid, sides_len)
                print('anno:', idx, 'new points_coors', sides_vex_world_coors)
        else:
            is_depth_valid = np.ones(len(sides_vex))
            sides_vex_depth_valid = sides_vex
        
        assert np.prod(sides_vex_dpth) > 0, 'anno:{} zero depth found {}'.format(idx, str(sides_vex_dpth))

        
def depth_completion(depth_raw_file, json_camera_intri, json_label, save_dir, vis_save_dir=None):
    # raw_dpth_mat = read_raw_depth(width, height, depth_raw_file)
    raw_dpth_mat = read_depth_npy(depth_raw_file)
    raw_dpth_mat = raw_dpth_mat.astype(np.uint16)
    fname = os.path.basename(depth_raw_file).split('.')[0]
    poly_shapes = read_json_label(json_label, key='shapes')
    camera_intrin, depth_scale = read_camera_intrin(json_camera_intri)
    
    show_sides_lines = True
    show_depth_comp = True

    sides_points_list = []
    sides_points_depths_list = []
    tstart = time.time()
    for idx, ann in enumerate(poly_shapes):
        if ann['shape_type'] == 'rectangle' and ann['label'] in ['delete']:
            continue

        label_list = ann['label'].split('-')
        assert len(label_list) in [1, 3], '{},{}'.format(json_label, str(label_list))
        sides_vex = np.array(ann['points'], dtype=np.float32)
        sides_vex = np.floor(sides_vex.reshape(-1, 2)).astype(np.int32)

        sides_vex_dpth = get_depth_from_mat(raw_dpth_mat, sides_vex)  
        print('label_list:', label_list, 'sides_vex_dpth:', sides_vex_dpth)
        sides_vex_world_coors = deproject_to_points(sides_vex, sides_vex_dpth, camera_intrin, depth_scale=depth_scale)
        sides_vex_world_coors = np.array(sides_vex_world_coors, dtype=np.float32)

        if len(label_list) > 1:
            is_depth_valid = np.array([int(li) for li in list(str(label_list[1]))])
            sides_vex_depth_valid = sides_vex[is_depth_valid == 1]

            if len(sides_vex_depth_valid) != len(sides_vex):
                assert len(label_list) == 3, 'label {} invalid'.format(ann['points'])
                sides_len_org = np.array([int(l) for l in label_list[2].split(',')])
                sides_len_meter = sides_len_org / 100.0 # centemeter to meter
                assert len(sides_len_meter) == len(sides_vex), 'label {} measured sizes uncompleted'.format(ann['points'])
                sides_vex_world_coors, sides_vex_dpth = get_point_by_dist_in_pane(sides_vex, sides_vex_world_coors, sides_vex_dpth, is_depth_valid, sides_len_meter)
        else:
            is_depth_valid = np.ones(len(sides_vex))
            sides_vex_depth_valid = sides_vex

        assert np.prod(sides_vex_dpth) > 0, 'zero depth found:'+str(sides_vex_dpth)

        # label_name = label_list[0]
        # label_height_mm = float(label_list[1]) * 10 # centimeter to millimeter
        # label_width_mm = float(label_list[2]) * 10
        sides_vex_world_coors /= depth_scale
        sides_len_meas = np.array([int(l) for l in label_list[2].split(',')]) if len(label_list) > 1 else []
        sides_len = get_sides_length(sides_len_meas, sides_vex_world_coors, measured_in_centimeter=True)

        # calculate sides depth
        sides_points_2d_coors_dict, sides_points_3d_coors_dict = calculate_sides_depth(sides_vex, sides_vex_world_coors, sides_len)

        compl_mat = np.zeros_like(raw_dpth_mat)
        compl_mat, dpth_mats = calculate_region_depth(sides_points_2d_coors_dict, sides_points_3d_coors_dict, compl_mat)
        raw_dpth_mat = np.where(compl_mat > 0, compl_mat, raw_dpth_mat)
        if show_depth_comp and vis_save_dir is not None:
            comp_depth_vis = vis_depth_mat(compl_mat, plt_show=False)
            cv2.imwrite(os.path.join(vis_save_dir, 'depth_calculated_anno{}_{}.png'.format(idx, time.time())), comp_depth_vis)
            for i, dpm in enumerate(dpth_mats):
                dpm_vis = vis_depth_mat(dpm, plt_show=False)
                cv2.imwrite(os.path.join(vis_save_dir, 'depth_calculated_anno{}_process{}_{}.png'.format(idx, i, time.time())), dpm_vis)

        sides_points_list.append(sides_points_2d_coors_dict)
        sides_points_depths_list.append(sides_points_3d_coors_dict)

        if show_sides_lines:
            raw_dpth_color = vis_depth_mat(raw_dpth_mat, plt_show=False)
            sides_color = [(255, 0, 0),(0, 0, 255),(255, 0, 255), (0, 255, 255)]
            for sid in range(len(sides_vex)):
                if sid != len(sides_vex) - 1:
                    cv2.line(raw_dpth_color, sides_vex[sid], sides_vex[sid+1], color=sides_color[sid], thickness=2)
                else:
                    cv2.line(raw_dpth_color, sides_vex[sid], sides_vex[0], color=sides_color[sid], thickness=2)
            cv2.imwrite(os.path.join(vis_save_dir, 'depth_completion_anno{}_{}.png'.format(idx, time.time())), raw_dpth_color)
        print("image:{}, anno:{}, {:3.3f} minutes used.".format(fname, idx, (time.time() - tstart) / 60))


    save_name = os.path.join(save_dir, fname+'_filled.npy')
    np.save(save_name, raw_dpth_mat)
    print('anno:{} saved to {}'.format(fname, save_name))

    if vis_save_dir is not None:
        raw_depth_mat_vis = vis_depth_mat(raw_dpth_mat, plt_show=False)
        cv2.imwrite(os.path.join(vis_save_dir, 'depth_completion_{}_{}.png'.format(fname, time.time())), raw_depth_mat_vis)

    
    save_sides_depth_name = os.path.join(save_dir, fname+'_sides_depth.npy')
    np.save(save_sides_depth_name, {'points':sides_points_list, 'depths':sides_points_depths_list})
    print('sides depth saved to {}'.format(save_sides_depth_name))

def read_polyinfo(txt_file):
    pinfo_list = []
    with open(txt_file, 'r') as f:
        pinfo = f.readline()
        while pinfo:
            pinfo = pinfo.strip('\n')
            pinfo_list.append(pinfo.split(' '))
            pinfo = f.readline()
    return pinfo_list


if __name__ == '__main__':
    assert len(sys.argv) > 1, "choose check or run in command line"
    run_type = sys.argv[1]
    assert run_type in ['check', 'run']

    root_dir = '/home/ly/data/datasets/trans-depth/20211019/aligned_depth_color/'
    npy_save_dir = os.path.join(root_dir, 'completed_depth_npy')
    vis_save_dir = os.path.join(root_dir, 'completed_depth_vis')
    color_dir_name = 'depth_vis'
    json_dir_name = 'depth_vis'
    if not os.path.isdir(npy_save_dir):
        os.mkdir(npy_save_dir)
    if not os.path.isdir(vis_save_dir):
        os.mkdir(vis_save_dir)

    aux_dir = 'depth_vis'
    if len(json_dir_name) == 0:
        json_dir = root_dir
        anno_json = glob.glob(root_dir+'/*.json')
    else:
        json_dir = os.path.join(root_dir, json_dir_name)
        anno_json = glob.glob(json_dir+'/*.json')

    anno_json.sort()
    intrin_json = '/home/ly/data/datasets/trans-depth/d455_intrin.json'

    if sys.argv[1] == 'check':
        fnames_file = os.path.join(json_dir, 'fnames_all.txt')
        f_names = open(fnames_file, 'w+')
        for anno in anno_json:
            fname = os.path.basename(anno).split('.')[0]
            color_png = os.path.join(root_dir, color_dir_name, fname+'.png')
            depth_npy = os.path.join(root_dir, fname+'.npy')
            poly_json = anno
            print("checking color_png:", color_png)
            check_depth(depth_npy, intrin_json, poly_json)
            f_names.write(fname+'\n')
            print()
        f_names.close()
        print("Depth check done!")
    else:
        ##################start calculate depth map######################
        fnames_run = os.path.join(json_dir, 'fnames_run.txt')
        f_run = open(fnames_run, 'r')
        run_names = f_run.readlines()
        run_names = [rn.strip() for rn in run_names]
        print("Files to make depth completion:", run_names)
        for anno in anno_json:
            fname = os.path.basename(anno).split('.')[0]
            if fname not in run_names:
                continue
            color_png = os.path.join(root_dir, fname+'.png')
            depth_npy = os.path.join(root_dir, fname+'.npy')
            poly_json = anno
            print("runing image:", color_png)
            vis_save_dir_f = os.path.join(vis_save_dir, fname)
            if not os.path.isdir(vis_save_dir_f):
                os.mkdir(vis_save_dir_f)
            depth_completion(depth_npy, intrin_json, poly_json, npy_save_dir, vis_save_dir=vis_save_dir_f)
            print()
        f_run.close()