from enum import unique
import os
import math
import cv2
import glob
from matplotlib.pyplot import axis
import torch
import numpy as np
import json
from matplotlib.path import Path

import sys
sys.path.append('/home/ly/workspace/my_linux_lib/realsense_libs/depth_generation')

from commons import read_json_label, decode_parsing_numpy
from sne_model import SNE, get_d455_cam_param
from depth_interpolation import read_camera_intrin, deproject_to_points, get_depth_from_mat


def gen_normal_images(base_folder):
    normal_save_dir = os.path.join(base_folder, 'normal_estimated')
    normal_vis_save_dir = os.path.join(base_folder, 'visualize', 'normal_estimated_vis')
    os.makedirs(normal_save_dir, exist_ok=True)
    os.makedirs(normal_vis_save_dir, exist_ok=True)
    
    depth_png_dir = base_folder + '/depth/'
    depth_pngs = glob.glob(depth_png_dir+'/*.png')
    depth_pngs.sort()
    
    sne_model = SNE()
    sne_model.eval()
    camParam = get_d455_cam_param()
    for dpng in depth_pngs:
        depth_image = cv2.imread(dpng, cv2.IMREAD_ANYDEPTH)
        imname = os.path.basename(dpng)
        normal = sne_model(torch.tensor(depth_image.astype(np.float32)/1000), camParam)
        normal_npy = normal.cpu().numpy()
        normal_npy = np.transpose(normal_npy, [1, 2, 0])
        normal_png = 255*(1+normal_npy)/2
        # normal_color = cv2.cvtColor(normal_png, cv2.COLOR_RGB2BGR)
        save_file = normal_save_dir+'/'+imname
        cv2.imwrite(save_file, normal_png.astype(np.uint8))
        # cv2.imwrite(normal_vis_save_dir+'/'+imname, normal_color)
        print(save_file)

def caculate_normal_of_plane(pixel_coords, camera_intrin, depth_scale, raw_dpth_mat, is_counter_clock_label=False):
    H, W = raw_dpth_mat.shape
    sides_vex = np.array(pixel_coords, dtype=np.float32)
    # assert len(sides_vex) >= 3, 'anno:{},sides_vex:{}'.format(idx, str(sides_vex))
    sides_vex = np.floor(sides_vex.reshape(-1, 2)).astype(np.int32)
    # sides_vex[:, 0] = np.clip(sides_vex[:, 0], a_min=0, a_max=W-1)
    # sides_vex[:, 1] = np.clip(sides_vex[:, 1], a_min=0, a_max=H-1)
    sides_vex_dpth = get_depth_from_mat(raw_dpth_mat, sides_vex)  
    sides_vex_points = deproject_to_points(sides_vex, sides_vex_dpth, camera_intrin, depth_scale=depth_scale)

    sides_vex_points = np.array(sides_vex_points)
    p1 = sides_vex_points[0]
    p2 = sides_vex_points[1]
    p3 = sides_vex_points[2]
    norm = get_xoy_respected_norm(p1, p2, p3)
    # l1 = p2 - p1 # vector of p1->p2
    # l2 = p3 - p2
    # dir1 = np.cross(l1, l2)
    # norm = dir1 /  np.sqrt(np.sum(np.square(dir1)))

    if len(sides_vex_points) > 3:
        p4 = sides_vex_points[3]
        norm2 = get_xoy_respected_norm(p3, p4, p1)
        # l3 = p4 - p3
        # l4 = p1 - p4
        # dir2 = np.cross(l3, l4)
        # norm2 = dir2 /  np.sqrt(np.sum(np.square(dir2)))
        plane1_depth_std = np.std(np.array([p1[-1], p2[-1], p3[-1]]))
        plane2_depth_std = np.std(np.array([p3[-1], p4[-1], p1[-1]]))
        norm = norm if plane1_depth_std < plane2_depth_std else norm2

    return norm.tolist()

def gen_normal_by_polygon_label(base_folder, intrin_json, date_folders):
    trans_labels = ['wall', 'door', 'window', 'guardrail']
    camera_intrin, depth_scale = read_camera_intrin(intrin_json)
    for df in date_folders:
        date_folder = os.path.join(base_folder, df) 
        normal_save_dir = os.path.join(date_folder, 'aligned_depth_color', 'poly_annotation_with_normal')
        os.makedirs(normal_save_dir, exist_ok=True)
        
        depth_npy_dir = os.path.join(date_folder, 'aligned_depth_color', 'completed_depth_npy')
        depth_npys = glob.glob(depth_npy_dir+'/*.npy')
        depth_npys.sort()

        ply_dir = os.path.join(date_folder, 'aligned_depth_color', 'depth_vis')
        ply_jsn = glob.glob(ply_dir+'/*.json')
        ply_jsn.sort()
        assert len(depth_npys) == len(ply_jsn)
        
        # deproject_to_points()
        for dnpy, pjsn in zip(depth_npys, ply_jsn):
            depth_mat = np.load(dnpy)
            with open(pjsn) as f:
                ply_dicts = json.load(f)
            for pln in ply_dicts['shapes']:
                is_counter_colock_poly = True
                # if pln['label'] in trans_labels:
                #     is_counter_colock_poly = True
                plane_normal = caculate_normal_of_plane(pln['points'], camera_intrin, depth_scale, depth_mat, is_counter_clock_label=is_counter_colock_poly)
                pln['normal'] = plane_normal
            save_json = os.path.join(normal_save_dir, os.path.basename(pjsn))
            with open(save_json, 'w') as f:
                json.dump(ply_dicts, f)
            print(save_json)
            # cv2.imwrite(save_file, normal_png.astype(np.uint8))
            # cv2.imwrite(normal_vis_save_dir+'/'+imname, normal_color)
        print()

def get_proper_coords(coord, depth_mat, min_depth, max_depth, sizes, iter_more=False):
    height, width = sizes
    r, c = coord
    # k*k grids without center
    def gridk_4_nocenter(r, c, kernel_size=5):
        s = (kernel_size - 1) // 2
        grid5 = []
        grid5.append([r-s, c-s])
        grid5.append([r+s, c-s])
        grid5.append([r+s, c+s])
        grid5.append([r-s, c+s])
        return grid5

    kernel_size = 9
    sample_func = gridk_4_nocenter
    #right
    gcoords = sample_func(r, c, kernel_size=kernel_size)
    selected_coords = []
    for cd in gcoords:
        if (cd[0] > 0 and cd[0] < height) and (cd[1] > 0 and cd[1] < width):
            if depth_mat[cd[0], cd[1]] > min_depth and depth_mat[cd[0], cd[1]] < max_depth:
                selected_coords.append(cd)

    if iter_more:
        max_iter = 6
        iter = 0
        new_r = r
        new_c = c
        while len(selected_coords) < 2 and iter < max_iter:
            if new_r+1 < height:
                new_r += 1
                new_c += 1
                gcoords_rb = sample_func(new_r, new_c, kernel_size=kernel_size)
            else:
                new_r -= 1
                new_c -= 1
                gcoords_rb = sample_func(new_r, new_c, kernel_size=kernel_size)
            
            for cd in gcoords_rb:
                if (cd[0] > 0 and cd[0] < height) and (cd[1] > 0 and cd[1] < width):
                    if depth_mat[cd[0], cd[1]] > min_depth and depth_mat[cd[0], cd[1]] < max_depth:
                        selected_coords.append(cd)
            iter += 1
    return selected_coords

def np_angle(v1, v2):
    vdot = np.sum(v1 * v2)
    norm = vdot / (np.sqrt(np.sum(np.square(v1))) * np.sqrt(np.sum(np.square(v2))) + 1e-6)
    return np.arccos(norm)

def get_xoy_respected_norm(coor1, coor2, coor3):
    l1 = coor2 - coor1 # vector of coor1->coor2
    l2 = coor3 - coor2
    dir1 = np.cross(l1, l2)
    dir1_reverse = np.cross(l2, l1)
    
    angle_to_xoy = np_angle(dir1, np.array([0, 0, 1]))
    dir = dir1 if angle_to_xoy <= math.pi/2 else dir1_reverse
    # dir = dir1_reverse if angle_to_xoy <= math.pi/2 else dir1
    norm = dir / (np.sqrt(np.sum(np.square(dir))) + 1e-6)
    return norm

def calculate_normal(coords, depth_mat, camera_intrin, depth_scale):
    # coords_sorted = sorted(coords, key=lambda p: math.atan2(p[0], p[1]))
    coords = np.array(coords)
    coords_f = np.flip(coords, axis=1) # (row, column) -> (column, row)
    sides_vex_dpth = get_depth_from_mat(depth_mat, coords_f)
    sides_vex_points = deproject_to_points(coords_f, sides_vex_dpth, camera_intrin, depth_scale=depth_scale)
    
    # sides_vex_points = sorted(sides_vex_points, key=lambda p: math.atan2(p[0], p[1]))
    sides_vex_points = np.array(sides_vex_points)
    p1 = sides_vex_points[0]
    p2 = sides_vex_points[1]
    p3 = sides_vex_points[2]
    norm = get_xoy_respected_norm(p1, p2, p3)
    
    if len(sides_vex_points) > 3:
        p4 = sides_vex_points[3]
        norm2 = get_xoy_respected_norm(p3, p4, p1)

        plane1_depth_std = np.std(np.array([p1[-1], p2[-1], p3[-1]]))
        plane2_depth_std = np.std(np.array([p3[-1], p4[-1], p1[-1]]))
        norm = norm if plane1_depth_std < plane2_depth_std else norm2
    return norm

def polygon2normal(label_dirs, labels_dir_root, intrin_json, 
        save_norm_dir_name='completed_normal', save_seg_dir_name='completed_segmentation', width=1280, height=720):
    all_labels = ['wall', 'window', 'door', 'guardrail']
    all_ids = [1, 2, 3]
    label_id_map = {1:['wall', 'window'], 2:['door'], 3:['guardrail']}
    label_name_map = {'wall':1, 'window':1, 'door':2, 'guardrail':3}
    min_depth_milli = 1
    max_depth_milli = 10000
    camera_intrin, depth_scale = read_camera_intrin(intrin_json)
    for date_dir in label_dirs:
        date_norm_json_dir = os.path.join(labels_dir_root, date_dir, 'aligned_depth_color', 'poly_annotation_with_normal')
        date_depth_npy_dir = os.path.join(labels_dir_root, date_dir, 'aligned_depth_color', 'completed_depth_npy')
        save_norm_dir = os.path.join(labels_dir_root, date_dir, 'aligned_depth_color', save_norm_dir_name)
        save_seg_dir = os.path.join(labels_dir_root, date_dir, 'aligned_depth_color', save_seg_dir_name)
        os.makedirs(save_norm_dir, exist_ok=True)
        os.makedirs(save_seg_dir, exist_ok=True)
        norm_jsns = glob.glob(date_norm_json_dir+'/*.json')
        norm_jsns.sort()
        for one_jsn in norm_jsns:
            seg_map = np.zeros((height, width), dtype=np.uint8)
            labels = read_json_label(one_jsn, key='shapes')
            all_points = [[] for _ in all_ids]
            x, y = np.mgrid[:height, :width]
            all_coors = np.hstack((y.reshape(-1, 1), x.reshape(-1, 1)))

            # generate seg map
            for l in labels:
                if l['label'] in all_labels:
                    real_id = label_name_map[l['label']]
                    id_from0 = real_id - 1
                    all_points[id_from0].append(l['points'])
            for label_id_from0, ap in enumerate(all_points):
                real_id = label_id_from0 + 1
                for p in ap:
                    poly_path = Path(p)
                    mask = poly_path.contains_points(all_coors)
                    mask = mask.reshape(height, width)
                    seg_map = np.where(mask, real_id, seg_map)
            
            ## generate normal map
            normal_map = np.zeros((3, height, width), dtype=np.uint8)
            # assign normal value for ploly labels
            assigned_map = np.zeros((height, width), dtype=np.uint8)
            fname = os.path.basename(one_jsn)
            name_no_suffix = fname.split('.')[0]
            depth_mat = np.load(os.path.join(date_depth_npy_dir, name_no_suffix+'_filled.npy'))
            for ply in labels:
                poly_path = Path(ply['points'])
                ply_mask = poly_path.contains_points(all_coors)
                ply_mask = ply_mask.reshape(height, width)
                pl_normal = ((np.array(ply['normal'], dtype=np.float32) + 1.0) / 2.0) * 255
                pl_normal = pl_normal.astype(np.uint8)
                normal_map[0] = np.where(ply_mask, pl_normal[0], normal_map[0])
                normal_map[1] = np.where(ply_mask, pl_normal[1], normal_map[1])
                normal_map[2] = np.where(ply_mask, pl_normal[2], normal_map[2])
                assigned_map = np.where(ply_mask, 1, assigned_map)
            # generate normal for unlabled area
            rows, cols = np.where(assigned_map == 0)
            zero_depth_num = 0
            no_plane_num = 0
            for r, c in zip(rows, cols):
                if not (depth_mat[r, c]  > min_depth_milli and depth_mat[r, c] < max_depth_milli):
                    zero_depth_num += 1
                    continue
                near_coords = get_proper_coords([r,c], depth_mat=depth_mat, min_depth=min_depth_milli, max_depth=max_depth_milli, 
                        sizes=(height, width), iter_more=True)
                plane_coords = [[r, c]]
                plane_coords += near_coords
                if len(plane_coords) > 2:
                    point_dir = calculate_normal(plane_coords, depth_mat, camera_intrin, depth_scale)
                    pl_normal = ((np.array(point_dir, dtype=np.float32) + 1.0) / 2.0) * 255
                    pl_normal = pl_normal.astype(np.uint8)
                    normal_map[0, r, c] = pl_normal[0]
                    normal_map[1, r, c] = pl_normal[1]
                    normal_map[2, r, c] = pl_normal[2]
                else:
                    no_plane_num += 1

                
            normal_map_png = normal_map.transpose(1,2,0)
            print(rows.shape, 'zero_depth_num', zero_depth_num, 'no_plane_num', no_plane_num)


            # seg_color = decode_parsing_numpy(seg_map)
            # cv2.imshow("normal_map_png", normal_map_png)
            # cv2.imshow("seg_color", seg_color)
            # if cv2.waitKeyEx() == 27:
            #     cv2.destroyAllWindows()
            # exit()

            norm_name = os.path.join(save_norm_dir, name_no_suffix+'.png')
            seg_name = os.path.join(save_seg_dir, name_no_suffix+'.npy')
            print(norm_name)
            print(seg_name)
            cv2.imwrite(norm_name, normal_map_png)
            np.save(seg_name, seg_map)

def main_polygon2normal():
    intrin_json = '/home/ly/data/datasets/trans-depth/d455_intrin.json'
    root_path = '/home/ly/data/datasets/trans-depth/'
    # date_folders = ['20210928-copy']
    date_folders = ['20210927', '20210928', '20210929', '20211002', '20211016', '20211017', '20211018',
                    '20211019', '20211020', '20211021', '20211027', '20211028', '20211031', '20211104']
    # polygon2normal(date_folders, root_path, intrin_json, save_norm_dir_name='calculated_normal', save_seg_dir_name='calculated_segmentation')
    polygon2normal(date_folders, root_path, intrin_json, save_norm_dir_name='completed_acute_znormal', save_seg_dir_name='completed_segmentation')

def main_gen_normal_images():
    save_dir_base = '/home/ly/data/datasets/trans-depth/Glass-RGBD/'
    gen_normal_images(save_dir_base)

def main_gen_normal_by_polygon_label():
    save_dir_base = '/home/ly/data/datasets/trans-depth/'
    intrin_json = '/home/ly/data/datasets/trans-depth/d455_intrin.json'
    date_folders = ['20210927', '20210928', '20210929', '20211002', '20211016', '20211017', '20211018',
                    '20211019', '20211020', '20211021', '20211027', '20211028', '20211031', '20211104']
    # date_folders = ['20210928-copy']
    gen_normal_by_polygon_label(save_dir_base, intrin_json, date_folders)

if __name__ == '__main__':
    # main_gen_normal_by_polygon_label()
    main_polygon2normal()