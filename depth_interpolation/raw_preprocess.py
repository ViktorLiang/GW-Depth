from cProfile import label
import enum
import os
import math

import cv2
from matplotlib import image
import numpy as np
from numpy import inf, mat
import glob
import json
from matplotlib.path import Path
from shapely.geometry import Polygon

from commons import read_depth_npy, read_json_label, vis_depth_mat, decode_parsing_numpy, intersect_remap, clamp_lines

GLASS_LABELS = ['wall', 'door', 'window', 'guardrail']
LABELS_ID = [1, 2, 3, 4]
LABELS_ID_MAP = {'wall':1, 'window':2, 'door':3, 'guardrail':4}
# draw polygon of labels
def vis_img_polygon_labelme(img_file, json_label, raw_depth_npy, compl_depth_npy, save_dir, show=True):
    I = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    raw_depth = read_depth_npy(raw_depth_npy)
    raw_depth = raw_depth.astype(np.uint16)
    raw_depth_colored = vis_depth_mat(raw_depth, plt_show=False)

    compl_depth = read_depth_npy(compl_depth_npy)
    compl_depth = compl_depth.astype(np.uint16)
    compl_depth_colored = vis_depth_mat(compl_depth, plt_show=False)

    poly_shapes = read_json_label(json_label, key='shapes')
    for idx, ann in enumerate(poly_shapes):
        label_list = ann['label'].split('-')
        assert len(label_list) > 0, '{},{}'.format(json_label, str(label_list))
        sides_vex = np.array(ann['points'], dtype=np.float32)
        assert len(sides_vex) == 4, 'anno:{},sides_vex:{}'.format(idx, str(sides_vex))
        sides_vex = np.floor(sides_vex.reshape(-1, 2)).astype(np.int32)

        sides_color = [(255, 0, 0),(0, 0, 255),(255, 0, 255), (0, 255, 255)]
        for sid in range(len(sides_vex)):
            if sid != len(sides_vex) - 1:
                cv2.line(I, sides_vex[sid], sides_vex[sid+1], color=sides_color[sid], thickness=2)
            else:
                cv2.line(I, sides_vex[sid], sides_vex[0], color=sides_color[sid], thickness=2)

    imname = os.path.basename(img_file)
    fname = imname.split('.')[0]
    raw_depth_name = os.path.join(save_dir, fname+'_raw-depth.png')
    compl_depth_name = os.path.join(save_dir, fname+'_depth-completion.png')
    img_lined_name = os.path.join(save_dir, fname+'.png')
    cv2.imwrite(img_lined_name, I)
    cv2.imwrite(raw_depth_name, raw_depth_colored)
    cv2.imwrite(compl_depth_name, compl_depth_colored)
    print(raw_depth_name, compl_depth_name, img_lined_name)
    if show:
        cv2.imshow('compl_depth_colored', compl_depth_colored)
        cv2.imshow('raw_depth', raw_depth_colored)
        cv2.imshow('img_lined', I)
        if cv2.waitKeyEx() == 27:
            cv2.destroyAllWindows()

def crop_by_labelme(json_label, rgb_png, raw_depth_npy, save_dir):
    CROP_LABEL_NAME = 'crop'
    img = cv2.imread(rgb_png)
    depth_mat = read_depth_npy(raw_depth_npy)
    json_dict = read_json_label(json_label, key='shapes')
    save_vis_dir = save_dir+'/depth_vis'
    if not os.path.isdir(save_vis_dir):
        os.makedirs(save_vis_dir)
    for ann in json_dict:
        if ann['label'] == CROP_LABEL_NAME and ann['shape_type'] == 'rectangle':
            top_lfet = list(map(int, ann['points'][0])) 
            right_bottom = list(map(int, ann['points'][1]))

            img_cropped = img[top_lfet[1]:right_bottom[1]+1, top_lfet[0]:right_bottom[0]+1]
            raw_dpeth_cropped = depth_mat[top_lfet[1]:right_bottom[1]+1, top_lfet[0]:right_bottom[0]+1]
            raw_dpeth_cropped_int = raw_dpeth_cropped.astype(np.uint16)

            raw_depth_vis = vis_depth_mat(raw_dpeth_cropped_int, show=False)
            
            fname = os.path.basename(rgb_png).split('.')[0]
            save_img_name = os.path.join(save_dir, fname+'.png')
            save_depth_vis_name = os.path.join(save_vis_dir, fname+'-vis.png')
            save_depth_name = os.path.join(save_dir, fname+'.npy')
            cv2.imwrite(save_img_name, img_cropped)
            cv2.imwrite(save_depth_vis_name, raw_depth_vis)
            np.save(save_depth_name, raw_dpeth_cropped)

            # cv2.imshow('img_cropped:', img_cropped)
            # cv2.imshow("raw_depth_cropped_vis", raw_depth_vis)
            # if cv2.waitKeyEx() == 27:
            #     cv2.destroyAllWindows()


def crop_for_folders():
    root_dir = '/home/ly/data/datasets/trans-depth/20211019/aligned_depth_color/'
    npy_save_dir = os.path.join(root_dir, 'cropped_depth_npy')
    png_save_dir = os.path.join(root_dir, 'cropped_depth_vis')
    if not os.path.isdir(npy_save_dir):
        os.mkdir(npy_save_dir)
    if not os.path.isdir(png_save_dir):
        os.mkdir(png_save_dir)
    json_dir_name = 'depth_vis'
    
    if len(json_dir_name) == 0:
        json_dir = root_dir
        anno_jsons = glob.glob(root_dir+'/*.json')
    else:
        json_dir = os.path.join(root_dir, json_dir_name)
        anno_jsons = glob.glob(json_dir+'/*.json')

    anno_jsons.sort()
    intrin_json = '/home/ly/data/datasets/trans-depth/d455_intrin.json'
    cropped_save_dir = root_dir+'/cropped'

    ##################start calculate depth map######################
    fnames_run = os.path.join(json_dir, 'fnames_crop.txt')
    f_run = open(fnames_run, 'r')
    run_names = f_run.readlines()
    run_names = [rn.strip() for rn in run_names]
    print("Files to crop:", run_names)
    for json_label in anno_jsons:
        fname = os.path.basename(json_label).split('.')[0]
        if fname not in run_names:
            continue
        color_png = os.path.join(root_dir, fname+'.png')
        depth_npy = os.path.join(root_dir, fname+'.npy')

        print("runing image:", color_png)
        png_save_file = os.path.join(png_save_dir, fname)
        npy_save_file = os.path.join(npy_save_dir, fname)
        crop_by_labelme(json_label, color_png, depth_npy, cropped_save_dir)
        print()
    f_run.close()

def label_copy(depth_labels_folder, crop_labels_folder, ):
    depth_jsons = glob.glob(depth_labels_folder+'/*.json')
    depth_jsons.sort()
    for dj in depth_jsons:
        jname = os.path.basename(dj)
        save_file = os.path.join(crop_labels_folder, jname)
        dj_dict = read_json_label(dj)
        crop_label = dj_dict.copy()
        crop_label['imageData'] = ''
        with open(save_file, 'w') as f:
            json.dump(crop_label, f)
        print(save_file)

def gen_filled_depth_vis(dirs, root_path, save_dir='crop_vis'):
    tnum_list = []
    tnum = 0
    for d in dirs:
        print(d)
        comp_dpth_dir = os.path.join(root_path, d, 'aligned_depth_color', 'completed_depth_npy')
        filled_vis_save_dir = os.path.join(root_path, d, 'aligned_depth_color', 'crop_vis')
        if not os.path.isdir(filled_vis_save_dir):
            os.mkdir(filled_vis_save_dir)
        depth_files = glob.glob(comp_dpth_dir+'/*.npy')
        t_l_num = 0
        for df in depth_files:
            fname = os.path.basename(df)
            fname = fname.split('.')[0]
            fname_list = fname.split('_')
            if fname_list[-1] == 'filled':
                fname = '_'.join(fname_list[:-1])
            dp_npy = read_depth_npy(df)
            dp_npy = dp_npy.astype(np.uint16)

            visd = vis_depth_mat(dp_npy, show=False)
            vis_file = filled_vis_save_dir+'/'+fname+'.png'
            cv2.imwrite(vis_file, visd)
            print('saved to {}'.format(vis_file))
            tnum += 1
            t_l_num += 1
        tnum_list.append(t_l_num)
    print(tnum_list)
    print(tnum)

def polygon2mask(label_dirs, labels_dir_root, save_dir_name='completed_segmentation', width=1280, height=720):
    all_labels = GLASS_LABELS
    all_ids = LABELS_ID
    label_name_map = LABELS_ID_MAP
    for one_dir_json in label_dirs:
        one_dir = os.path.join(labels_dir_root, one_dir_json, 'aligned_depth_color', 'depth_vis')
        save_dir = os.path.join(labels_dir_root, one_dir_json, 'aligned_depth_color', save_dir_name)
        one_dir_labels = glob.glob(one_dir+'/*.json')
        one_dir_labels.sort()
        for one_label in one_dir_labels:
            
            label_map = np.zeros((height, width), dtype=np.uint8)
            labels = read_json_label(one_label, key='shapes')
            all_points = [[] for _ in all_ids]
            x, y = np.mgrid[:height, :width]
            all_coors = np.hstack((y.reshape(-1, 1), x.reshape(-1, 1)))
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
                    label_map = np.where(mask, real_id, label_map)

            # aa = decode_parsing_numpy(label_map)
            # cv2.imshow("dpth_arr", aa)
            # if cv2.waitKeyEx() == 27:
            #     cv2.destroyAllWindows()

            fname = os.path.basename(one_label)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            npyname = os.path.join(save_dir, fname.split('.')[0])
            print(npyname)
            np.save(npyname, label_map)


def main_label_copy():
    folder_names = ['20210928-copy']
    for f in folder_names:
        crop_labels_folder = '/home/ly/data/datasets/trans-depth/{}/aligned_depth_color/crop_vis'.format(f)
        depth_labels_folder = '/home/ly/data/datasets/trans-depth/{}/aligned_depth_color/depth_vis'.format(f)
        label_copy(depth_labels_folder, crop_labels_folder)

def label_crop(init_labels_dir, crop_labels_dir, save_dir):
    trans_labels = GLASS_LABELS
    init_json = glob.glob(init_labels_dir+'/*.json')
    crop_json = glob.glob(crop_labels_dir+'/*.json')
    init_json.sort()
    crop_json.sort()
    assert len(init_json) == len(crop_json)
    for init_j in init_json:
        init_j_name = os.path.basename(init_j)
        init_name_noext = init_j_name.split('.')[0]
        crop_j = os.path.join(crop_labels_dir, init_name_noext)
        assert os.path.isfile(crop_j), '{} not found'.format(crop_j)
        init_label = read_json_label(init_j)
        crop_label = read_json_label(crop_j, key='shapes')
        for crp in crop_label:
            # modify point coordination
            left_top = crp['points'][0]
            right_bottom = crp['points'][1]
            for ply_id, ply in enumerate(init_label['shapes']):
                for i, p in enumerate(ply['points']):
                    print(i)
                    if not (p[0] >= left_top[0] and p[1] >= left_top[1]):
                        print(crop_j)
                        print('init label id:'+str(ply_id))
                        print('left_top:{}, {}'.format(str(left_top), str(p)))
                        exit()
                    assert p[0] <= right_bottom[0] and p[1] <= right_bottom[1], 'right_bottom:{}, {}'.format(str(right_bottom), str(p))
                    new_p = [p[0] - left_top[0], p[1] - left_top[1]]
                    ply['points'][i] = new_p
            exit()
        print()

def crop_valid(process_dates, save_dir, max_distance=10000):
    trans_labels = GLASS_LABELS
    base_dir = '/home/ly/data/datasets/trans-depth/{}/aligned_depth_color/'
    depth_save_dir = os.path.join(save_dir, 'depth')
    depth_camera_save_dir = os.path.join(save_dir, 'depth_camera')
    seg_save_dir = os.path.join(save_dir, 'segmentation')
    rgb_save_dir = os.path.join(save_dir, 'images')
    crp_json_save_dir = os.path.join(save_dir, 'polygon_json')
    save_dirs = [depth_save_dir, depth_camera_save_dir, seg_save_dir, rgb_save_dir, crp_json_save_dir]
    for sd in save_dirs:
        if not os.path.isdir(sd):
            os.mkdir(sd)

    dpth_vis_save_dir = os.path.join(save_dir, 'visualize', 'depth_vis')
    dpth_camera_vis_save_dir = os.path.join(save_dir, 'visualize', 'depth_camera_vis')
    seg_vis_save_dir = os.path.join(save_dir, 'visualize', 'segmentation_vis')
    save_vis_dirs = [dpth_vis_save_dir, dpth_camera_vis_save_dir, seg_vis_save_dir]
    for sd in save_vis_dirs:
        if not os.path.isdir(sd):
            os.makedirs(sd, exist_ok=True)
    
    image_id = 0
    for f in process_dates:
        base_folder = base_dir.format(f)
        init_label_dir = base_folder + '/completed_depth_npy/'
        seg_label_dir = base_folder + '/completed_segmentation/'
        crop_label_dir = base_folder + '/crop_vis/'
        crop_orgin_images = base_folder + '/depth_vis/fnames_image.txt'
        init_json_dir = base_folder + '/depth_vis/'
        frame_json_dir = base_folder + '/frame_vis/'

        depth_npy = glob.glob(init_label_dir+'/*.npy')
        seg_npy = glob.glob(seg_label_dir+'/*.npy')
        crop_jsn = glob.glob(crop_label_dir+'/*.json')
        init_jsn = glob.glob(init_json_dir+'/*.json')
        frame_jsn = glob.glob(frame_json_dir+'/*.json')
        depth_npy.sort()
        seg_npy.sort()
        crop_jsn.sort()
        init_jsn.sort()
        frame_jsn.sort()

        # maps to find no cover picture
        im_maps = {}
        if os.path.isfile(crop_orgin_images):
            with open(crop_orgin_images) as cf:
                im_list = cf.readlines()
            for im in im_list:
                im_map = im.split(' ')
                if len(im_map) == 2:
                    im_label, im_origin = im_map
                else:
                    im_label = im
                    im_origin = ''
                im_maps[im_label.strip()] = im_origin.strip()

        assert len(depth_npy) == len(seg_npy) == len(crop_jsn) == len(init_jsn) == len(frame_jsn), (len(depth_npy), len(seg_npy), len(crop_jsn), len(init_jsn), len(frame_jsn), f)
        
        for c_jsn, ini_jsn, frm_jsn, d_npy, s_npy in zip(crop_jsn, init_jsn, frame_jsn, depth_npy, seg_npy):
            c_points = read_json_label(c_jsn, key='shapes')
            ini_points = read_json_label(ini_jsn, key='shapes')
            frm_points = read_json_label(frm_jsn, key='shapes')
            d_mat = read_depth_npy(d_npy)
            s_mat = read_depth_npy(s_npy)
            if d_mat.dtype == np.float16:
                d_mat = d_mat.astype(np.uint16)
            fname = os.path.basename(s_npy).split('.')[0]
            if fname in im_maps:
                if len(im_maps[fname]) > 0:
                    fname = im_maps[fname].split('.')[0]
            img_mat = cv2.imread(base_folder+'/'+fname+'.png')
            d_camera_mat = read_depth_npy(base_folder+'/'+fname+'.npy')
            d_camera_mat = d_camera_mat.astype(np.uint16)
            for rec_id, crp in enumerate(c_points):
                fn_list = fname.split('_')
                fn_list[-1] = str(rec_id)
                fname = '_'.join(fn_list)
                print(fname)

                # generate crop area coordinates
                left_top = crp['points'][0]
                right_bottom = crp['points'][1]
                lx = math.ceil(left_top[0])
                ly = math.ceil(left_top[1])
                bx = math.floor(right_bottom[0])
                by = math.floor(right_bottom[1])
                image_width = bx - lx + 1
                image_height = by - ly + 1

                # update polygon coordinates in new cropped images
                new_poly_list = []
                poly_id = 0 
                for pn_id, (ini, frm) in enumerate(zip(ini_points, frm_points)):
                    if not ini['label'] in trans_labels:
                        continue
                    crop_1 = [left_top[0], right_bottom[1]]
                    crop_2 = [right_bottom[0], left_top[1]]
                    crop_coors = [left_top, crop_1, right_bottom, crop_2]

                    # generate new polygon for origin lines
                    new_points = intersect_remap(crop_coors, ini['points'])
                    if len(new_points) == 0:
                        continue
                    
                    # generate frame lines, labels have form as 'wall-0110' where '0' denotes corresponding line is not frame
                    frm_label_splits = frm['label'].split('-')
                    assert len(frm_label_splits) in [1, 2], frm
                    is_frames = np.ones(len(frm['points']), dtype=np.uint8)
                    raw_lines = []
                    if len(frm_label_splits) == 2:
                        for idx, isframe in enumerate(frm_label_splits[1]):
                            is_frames[idx] = int(isframe)
                    edge_vex = np.array(frm['points'], dtype=np.float32)
                    edge_vex = np.floor(edge_vex.reshape(-1, 2)).astype(np.int32)
                    for sid in range(len(edge_vex)):
                        if not is_frames[sid]:
                            continue
                        if sid != len(edge_vex) - 1:
                            raw_lines.append([edge_vex[sid].tolist(), edge_vex[sid+1].tolist()])
                        else:
                            raw_lines.append([edge_vex[sid].tolist(), edge_vex[0].tolist()])
                    raw_lines = np.array(raw_lines, dtype=np.float32)
                    raw_lines = raw_lines.reshape(-1, 4) # (Num_frames, 4)
                    # clamp within cropped rectangle
                    clamped_frames = clamp_lines(left_top, right_bottom, raw_lines)

                    new_json = {'label':ini['label'], 'shape_type':ini['shape_type']}
                    new_json['points'] = new_points[:-1] # last point is same with the first one, thus remove here
                    new_json['frames'] = clamped_frames.tolist()
                    new_json['poly_id'] = poly_id # unique id for polygon in one image.
                    poly_id += 1
                    new_poly_list.append(new_json)

                if len(new_poly_list) == 0:
                    continue
                save_crpped_json = os.path.join(crp_json_save_dir, fname+'.json')
                save_json = {'shapes': new_poly_list, 'imagePath':fname+'.png', 
                            'imageWidth':image_width, 'imageHeight':image_height, 'imageId':image_id}
                with open(save_crpped_json, 'w') as scf:
                    json.dump(save_json, scf)

                # crop valid new image/depth/segmentation/ area
                valid_dpth_mat = d_mat[ly:by+1, lx:bx+1]
                valid_dpth_cam_mat = d_camera_mat[ly:by+1, lx:bx+1]
                valid_seg_mat = s_mat[ly:by+1, lx:bx+1]
                valid_img_mat = img_mat[ly:by+1, lx:bx+1]
                # relpace inf value and larger than max distance value to zero distance.
                valid_dpth_mat[valid_dpth_mat > max_distance] = 0.0
                valid_dpth_mat[np.isinf(valid_dpth_mat)] = 0.0
                valid_dpth_mat[np.isnan(valid_dpth_mat)] = 0.0
                valid_dpth_cam_mat[valid_dpth_cam_mat > max_distance] = 0.0
                valid_dpth_cam_mat[np.isinf(valid_dpth_cam_mat)] = 0.0
                valid_dpth_cam_mat[np.isnan(valid_dpth_cam_mat)] = 0.0

                depth_mat_name = depth_save_dir+'/'+fname+'.png'
                depth_camera_mat_name = depth_camera_save_dir+'/'+fname+'.png'
                seg_mat_name = seg_save_dir+'/'+fname+'.png'
                rgb_png_name = rgb_save_dir+'/'+fname+'.png'
                cv2.imwrite(depth_mat_name, valid_dpth_mat)
                cv2.imwrite(depth_camera_mat_name, valid_dpth_cam_mat)
                cv2.imwrite(seg_mat_name, valid_seg_mat)
                cv2.imwrite(rgb_png_name, valid_img_mat)

                # save visualized images
                vis_dpth = vis_depth_mat(valid_dpth_mat, show=False)
                vis_dpth_camera = vis_depth_mat(valid_dpth_cam_mat, show=False)
                vis_seg = decode_parsing_numpy(valid_seg_mat)
                dpth_vis_png_name = dpth_vis_save_dir + '/' + fname + '.png'
                dpth_cam_vis_png_name = dpth_camera_vis_save_dir + '/' + fname + '.png'
                seg_vis_png_name = seg_vis_save_dir + '/' + fname + '.png'
                cv2.imwrite(dpth_vis_png_name, vis_dpth)
                cv2.imwrite(dpth_cam_vis_png_name, vis_dpth_camera)
                cv2.imwrite(seg_vis_png_name, vis_seg)
                image_id += 1
                
        print()


# generate annotation with cropped valid areas
def crop_valid_noframe(process_dates, save_dir, max_distance=10000, start_image_id=0):
    trans_labels = GLASS_LABELS
    base_dir = '/home/ly/data/datasets/trans-depth/{}/aligned_depth_color/'
    depth_save_dir = os.path.join(save_dir, 'depth')
    depth_camera_save_dir = os.path.join(save_dir, 'depth_camera')
    seg_save_dir = os.path.join(save_dir, 'segmentation')
    rgb_save_dir = os.path.join(save_dir, 'images')
    crp_json_save_dir = os.path.join(save_dir, 'polygon_json')
    save_dirs = [depth_save_dir, depth_camera_save_dir, seg_save_dir, rgb_save_dir, crp_json_save_dir]
    for sd in save_dirs:
        if not os.path.isdir(sd):
            os.mkdir(sd)

    dpth_vis_save_dir = os.path.join(save_dir, 'visualize', 'depth_vis')
    dpth_camera_vis_save_dir = os.path.join(save_dir, 'visualize', 'depth_camera_vis')
    seg_vis_save_dir = os.path.join(save_dir, 'visualize', 'segmentation_vis')
    save_vis_dirs = [dpth_vis_save_dir, dpth_camera_vis_save_dir, seg_vis_save_dir]
    for sd in save_vis_dirs:
        if not os.path.isdir(sd):
            os.makedirs(sd, exist_ok=True)
    
    image_id = start_image_id
    for f in process_dates:
        base_folder = base_dir.format(f)
        init_label_dir = base_folder + '/completed_depth_npy/'
        seg_label_dir = base_folder + '/completed_segmentation/'
        crop_label_dir = base_folder + '/crop_vis/'
        crop_orgin_images = base_folder + '/depth_vis/fnames_image.txt'
        init_json_dir = base_folder + '/depth_vis/'

        depth_npy = glob.glob(init_label_dir+'/*.npy')
        seg_npy = glob.glob(seg_label_dir+'/*.npy')
        crop_jsn = glob.glob(crop_label_dir+'/*.json')
        init_jsn = glob.glob(init_json_dir+'/*.json')
        depth_npy.sort()
        seg_npy.sort()
        crop_jsn.sort()
        init_jsn.sort()

        # maps to find no cover picture
        im_maps = {}
        if os.path.isfile(crop_orgin_images):
            with open(crop_orgin_images) as cf:
                im_list = cf.readlines()
            for im in im_list:
                im_map = im.split(' ')
                if len(im_map) == 2:
                    im_label, im_origin = im_map
                else:
                    im_label = im
                    im_origin = ''
                im_maps[im_label.strip()] = im_origin.strip()

        assert len(depth_npy) == len(seg_npy) == len(crop_jsn) == len(init_jsn), (len(depth_npy), len(seg_npy), len(crop_jsn), len(init_jsn), f)
        
        for c_jsn, ini_jsn, d_npy, s_npy in zip(crop_jsn, init_jsn, depth_npy, seg_npy):
            c_points = read_json_label(c_jsn, key='shapes')
            ini_points = read_json_label(ini_jsn, key='shapes')
            d_mat = read_depth_npy(d_npy)
            s_mat = read_depth_npy(s_npy)
            if d_mat.dtype == np.float16:
                d_mat = d_mat.astype(np.uint16)
            fname = os.path.basename(s_npy).split('.')[0]
            if fname in im_maps:
                if len(im_maps[fname]) > 0:
                    fname = im_maps[fname].split('.')[0]
            img_mat = cv2.imread(base_folder+'/'+fname+'.png')
            d_camera_mat = read_depth_npy(base_folder+'/'+fname+'.npy')
            d_camera_mat = d_camera_mat.astype(np.uint16)
            for rec_id, crp in enumerate(c_points):
                fn_list = fname.split('_')
                fn_list[-1] = str(rec_id)
                fname = '_'.join(fn_list)
                print(fname)

                # generate crop area coordinates
                left_top = crp['points'][0]
                right_bottom = crp['points'][1]
                lx = math.ceil(left_top[0])
                ly = math.ceil(left_top[1])
                bx = math.floor(right_bottom[0])
                by = math.floor(right_bottom[1])
                image_width = bx - lx + 1
                image_height = by - ly + 1

                # update polygon coordinates in new cropped images
                new_poly_list = []
                poly_id = 0 
                for pn_id, ini in enumerate(ini_points):
                    label_splits = ini['label'].split('-')
                    real_label = label_splits[0]
                    # if not ini['label'] in trans_labels:
                    if not real_label in trans_labels:
                        continue
                    crop_1 = [left_top[0], right_bottom[1]]
                    crop_2 = [right_bottom[0], left_top[1]]
                    crop_coors = [left_top, crop_1, right_bottom, crop_2]

                    # generate new polygon for origin lines
                    new_points = intersect_remap(crop_coors, ini['points'])
                    if len(new_points) == 0:
                        continue
                    
                    # new_json = {'label':ini['label'], 'shape_type':ini['shape_type']}
                    new_json = {'label':real_label, 'shape_type':ini['shape_type']}
                    new_json['points'] = new_points[:-1] # last point is same with the first one, thus remove here
                    new_json['poly_id'] = poly_id # unique id for polygon in one image.
                    poly_id += 1
                    new_poly_list.append(new_json)

                if len(new_poly_list) == 0:
                    print(image_id, 'new_poly_list empty', 'ini_points', ini_points)
                    continue
                save_crpped_json = os.path.join(crp_json_save_dir, fname+'.json')
                save_json = {'shapes': new_poly_list, 'imagePath':fname+'.png', 
                            'imageWidth':image_width, 'imageHeight':image_height, 'imageId':image_id}
                with open(save_crpped_json, 'w') as scf:
                    json.dump(save_json, scf)

                # crop valid new image/depth/segmentation/ area
                valid_dpth_mat = d_mat[ly:by+1, lx:bx+1]
                valid_dpth_cam_mat = d_camera_mat[ly:by+1, lx:bx+1]
                valid_seg_mat = s_mat[ly:by+1, lx:bx+1]
                valid_img_mat = img_mat[ly:by+1, lx:bx+1]
                # relpace inf value and larger than max distance value to zero distance.
                valid_dpth_mat[valid_dpth_mat > max_distance] = 0.0
                valid_dpth_mat[np.isinf(valid_dpth_mat)] = 0.0
                valid_dpth_mat[np.isnan(valid_dpth_mat)] = 0.0
                valid_dpth_cam_mat[valid_dpth_cam_mat > max_distance] = 0.0
                valid_dpth_cam_mat[np.isinf(valid_dpth_cam_mat)] = 0.0
                valid_dpth_cam_mat[np.isnan(valid_dpth_cam_mat)] = 0.0

                depth_mat_name = depth_save_dir+'/'+fname+'.png'
                depth_camera_mat_name = depth_camera_save_dir+'/'+fname+'.png'
                seg_mat_name = seg_save_dir+'/'+fname+'.png'
                rgb_png_name = rgb_save_dir+'/'+fname+'.png'
                print(image_id)
                cv2.imwrite(depth_mat_name, valid_dpth_mat)
                cv2.imwrite(depth_camera_mat_name, valid_dpth_cam_mat)
                cv2.imwrite(seg_mat_name, valid_seg_mat)
                cv2.imwrite(rgb_png_name, valid_img_mat)

                # save visualized images
                vis_dpth = vis_depth_mat(valid_dpth_mat, show=False)
                vis_dpth_camera = vis_depth_mat(valid_dpth_cam_mat, show=False)
                vis_seg = decode_parsing_numpy(valid_seg_mat)
                dpth_vis_png_name = dpth_vis_save_dir + '/' + fname + '.png'
                dpth_cam_vis_png_name = dpth_camera_vis_save_dir + '/' + fname + '.png'
                seg_vis_png_name = seg_vis_save_dir + '/' + fname + '.png'
                cv2.imwrite(dpth_vis_png_name, vis_dpth)
                cv2.imwrite(dpth_cam_vis_png_name, vis_dpth_camera)
                cv2.imwrite(seg_vis_png_name, vis_seg)
                image_id += 1
                
        print()


def main_vis_img_polygon_labelme():
    img_file = '/home/ly/data/datasets/trans-depth/20211018/aligned_depth_color/20211018_165254_2.png'
    json_label = '/home/ly/data/datasets/trans-depth/20211018/aligned_depth_color/20211018_165254_2.json'
    raw_depth_npy = '/home/ly/data/datasets/trans-depth/20211018/aligned_depth_color/20211018_165254_2.npy'
    compl_depth_npy = '/home/ly/data/datasets/trans-depth/20211018/aligned_depth_color/completed_depth_npy/20211018_165254_2_filled.npy'
    save_dir = '/home/ly/data/datasets/trans-depth/temp_test/vis_save'
    vis_img_polygon_labelme(img_file, json_label, raw_depth_npy, compl_depth_npy, save_dir, show=True)

# generate visulized depth according to completed depth map
def main_gen_filled_depth_vis():
    root_path = '/home/ly/data/datasets/trans-depth/'
    # dirs = ['20210927', '20210928', '20210929',
    # '20211002', '20211004', '20211016', '20211017', '20211018', 
    # '20211019', '20211020', '20211021', '20211027', '20211028', '20211031',
    # '20211104', '20220821']
    dirs = ['20220821']
    gen_filled_depth_vis(dirs, root_path)

def main_crop_valid():
    save_dir_base = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/'
    # dirs = ['20210927', '20210928', '20210929',
    # '20211002', '20211016', '20211017', '20211018', 
    # '20211019', '20211020', '20211021', '20211027', '20211028', '20211031',
    # '20211104', '20211105', '20211107', '20211111', '20220430', '20220514', '20220821']
    crop_valid_noframe(dirs, save_dir_base, start_image_id=0)


def main_label_crop():
    folder_names = ['20210928-copy']
    for f in folder_names:
        init_label_dir = '/home/ly/data/datasets/trans-depth/{}/aligned_depth_color/depth_vis/'.format(f)
        crop_labels_dir = '/home/ly/data/datasets/trans-depth/{}/aligned_depth_color/filled_vis/'.format(f)
        save_dir = '/home/ly/data/datasets/trans-depth/{}/aligned_depth_color/crop_vis/'.format(f)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        label_crop(init_label_dir, crop_labels_dir, save_dir)

# generate segmentation mask according to polygon labels
def main_polygon2mask():
    root_path = '/home/ly/data/datasets/trans-depth/'
    # dirs = ['20210927', '20210928', '20210929',
    # '20211002', '20211016', '20211017', '20211018', 
    # '20211019', '20211020', '20211021', '20211027', '20211028', '20211031',
    # '20211104', '20211105', '20211107', '20211111', '20220430', '20220514', '20220821']
    dirs = ['20220821']
    polygon2mask(dirs, root_path, save_dir_name='completed_segmentation')

def split_train_val():
    val_ratio = 0.333
    root_path = '/home/ly/data/datasets/trans-depth/Glass-RGBD'
    image_path = os.path.join(root_path, 'images')
    split_train = os.path.join(root_path, 'train.txt')
    split_val = os.path.join(root_path, 'val.txt')
    fnames = os.listdir(image_path)
    import random
    random_num = 3
    for _ in range(random_num):
        random.shuffle(fnames)
    f_train = open(split_train, "w")
    f_test = open(split_val, "w")
    val_num = int(len(fnames) * val_ratio)
    train_num = len(fnames) - val_num
    fnames_train = fnames[:train_num]
    fnames_val = fnames[train_num:]
    fnames_train.sort()
    fnames_val.sort()
    for fnm in fnames_train:
        fnm = fnm.split('.')[0]
        f_train.write(fnm+'\n')
    for fnm in fnames_val:
        fnm = fnm.split('.')[0]
        f_test.write(fnm+'\n')
    f_train.close()
    f_test.close()

if __name__ == '__main__':
    # #generate visulaized depth png for cropping
    # main_gen_filled_depth_vis()
    # #polygon to mask
    # main_polygon2mask()
    # main_crop_valid()
    pass