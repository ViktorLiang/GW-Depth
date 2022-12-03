from glob import glob
import os
import cv2
import sys
import shutil

import numpy as np
from commons import gen_pairs, read_json_label, COLORS
from raw_preprocess import GLASS_LABELS

#(128,128,128), ,(192,0,0)
label_color_map = {'wall':(128,0,128), 'window':(0,128,0), 'door':(0,128,128), 'guardrail':(0,0,128), 
        'opaquewall':(64,128,0), 'ceiling':(128,128,0), 'floor':(192,0,0)}

def show_polygon_from_json(img_file, json_label, show=True, save_dir='saved_vis', label_color=False,
         oper_type='', draw_marker=False, draw_centers=False, show_labels=None):
    # I = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    I = cv2.imread(img_file, cv2.IMREAD_COLOR)
    print(img_file)
    height, width = I.shape[0:2]
    if show_labels is not None:
        for l in show_labels:
            assert l in label_color_map,  l

    if oper_type != 'vis_crop':
        poly_shapes = read_json_label(json_label, key='shapes')
    else:
        poly_shapes = read_json_label(json_label[0], key='shapes')
        crop_shapes = read_json_label(json_label[1], key='shapes')

    for idx, ann in enumerate(poly_shapes):
        if ann['shape_type'] in ['rectangle'] and ann['label'] == 'delete':
            continue
        label_list = ann['label'].split('-')
        real_label = label_list[0]
        assert len(label_list) > 0, '{},{}'.format(json_label, str(label_list))
        if oper_type == 'frame':
            if real_label not in GLASS_LABELS:
                continue
            is_frames = np.ones(len(ann['points']), dtype=np.uint8)
            if len(label_list) == 2:
                for idx, isframe in enumerate(label_list[1]):
                    is_frames[idx] = int(isframe)
        if show_labels is not None and real_label not in show_labels:
            continue
        sides_vex = np.array(ann['points'], dtype=np.float32)
        # assert len(sides_vex) == 4, 'anno:{},sides_vex:{}'.format(idx, str(sides_vex))
        sides_vex = np.floor(sides_vex.reshape(-1, 2)).astype(np.int32)

        sides_color = [(255, 0, 0),(0, 0, 255),(255, 0, 255), (0, 255, 255)]
        if ann['shape_type'] == 'rectangle':
            new_vex = np.zeros((4, 2), dtype=np.uint16)
            new_vex[0] = sides_vex[0]
            new_vex[2] = sides_vex[1]
            new_vex[1][0] = sides_vex[0][0]
            new_vex[1][1] = sides_vex[1][1]
            new_vex[3][0] = sides_vex[1][0]
            new_vex[3][1] = sides_vex[0][1]
            sides_vex = new_vex
        
        # connect end to head when meets end points
        if ann['shape_type'] == 'polygon':
            if not label_color:
                for sid in range(len(sides_vex)):
                    if oper_type == 'frame' and not is_frames[sid]:
                        continue
                    if sid != len(sides_vex) - 1:
                        cv2.line(I, sides_vex[sid], sides_vex[sid+1], color=sides_color[sid%4], thickness=4)
                    else:
                        cv2.line(I, sides_vex[sid], sides_vex[0], color=sides_color[sid%4], thickness=4)
            else:
                # label = label_list[0]
                ply_color = list(COLORS[(idx+10) % len(COLORS)])
                for sid in range(len(sides_vex)):
                    if oper_type == 'frame' and not is_frames[sid]:
                        continue
                    if sid != len(sides_vex) - 1:
                        cv2.line(I, sides_vex[sid], sides_vex[sid+1], color=ply_color, thickness=4)
                        if draw_marker:
                            cv2.drawMarker(I, sides_vex[sid], markerType=cv2.MARKER_TRIANGLE_UP, color=ply_color, thickness=4)
                    else:
                        cv2.line(I, sides_vex[sid], sides_vex[0], color=ply_color, thickness=4)
                        if draw_marker:
                            cv2.drawMarker(I, sides_vex[sid], markerType=cv2.MARKER_TRIANGLE_UP, color=ply_color, thickness=4)
                        #     cv2.drawMarker(I, sides_vex[0], markerType=cv2.MARKER_CROSS, color=ply_color, thickness=4)
                if draw_centers:
                    all_sides = np.array(ann['points'], dtype=np.float32)
                    centers = all_sides.mean(axis=0)
                    I = image_draw_circles(I, [centers.astype(np.int32).tolist()], ply_color)

        if ann['shape_type'] == 'point':
            pnt = np.array(ann['points'][0], dtype=np.int32)
            cv2.circle(I, pnt, radius=6, color=(0, 255, 0), thickness=-1)

    if oper_type == 'vis_crop' and crop_shapes[0]['shape_type'] == 'rectangle':
        sides_vex = np.array(crop_shapes[0]['points'], dtype=np.float32)
        # assert len(sides_vex) == 4, 'anno:{},sides_vex:{}'.format(idx, str(sides_vex))
        sides_vex = np.floor(sides_vex.reshape(-1, 2)).astype(np.int32)
        start_column = sides_vex[0, 0]
        end_column = sides_vex[1, 0]
        start_row = sides_vex[0, 1]
        end_row = sides_vex[1, 1]

        I_CROP = I[start_row:end_row+1, start_column:end_column+1]
        print('sides_vex', sides_vex)
        print(I.shape, I_CROP.shape)
        I = I_CROP




    if show:
        # cv2.imshow('img_lined', I)
        # if cv2.waitKeyEx() == 27:
        #     cv2.destroyAllWindows()
        
        # plt.title('completed_depth')
        # plt.imshow(I)
        # plt.show()
        fname = os.path.basename(img_file)
        save_new_name = fname.split('.')[0] + '-showlabel.png'
        save_file = '{}/{}'.format(save_dir, save_new_name)
        print(save_file)
        cv2.imwrite(save_file, I)

def image_draw_circles(image, centers, color, radius=10, thickness=-1):
    for i, c in enumerate(centers):
        cv2.circle(image, c, radius=radius, color=color, thickness=thickness)
    return image

def show_frames_from_json(img_file, json_label, show=True, save_dir='saved_vis', label_color=False):
    # I = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    I = cv2.imread(img_file, cv2.IMREAD_COLOR)
    print(img_file)
    height, width = I.shape[0:2]
    print(img_file)

    poly_shapes = read_json_label(json_label, key='shapes')
    for idx, ann in enumerate(poly_shapes):
        if ann['shape_type'] in ['rectangle'] and ann['label'] == 'delete':
            continue
        label_list = ann['label'].split('-')
        assert len(label_list) > 0, '{},{}'.format(json_label, str(label_list))

        sides_vex = np.array(ann['frames'], dtype=np.float32)
        sides_vex = np.floor(sides_vex).astype(np.int32)

        sides_color = [(255, 0, 0),(0, 0, 255),(255, 0, 255), (0, 255, 255)]
        
        # connect end to head when meets end points
        if not label_color:
            for sid in range(len(sides_vex)):
                cv2.line(I, (sides_vex[sid][0], sides_vex[sid][1]), (sides_vex[sid][2], sides_vex[sid][3]), 
                    color=sides_color[sid%4], thickness=4)
        else:
            label = label_list[0]
            for sid in range(len(sides_vex)):
                cv2.line(I, (sides_vex[sid][0], sides_vex[sid][1]), (sides_vex[sid][2], sides_vex[sid][3]), 
                    color=label_color_map[label], thickness=4)


    if show:
        # cv2.imshow('img_lined', I)
        # if cv2.waitKeyEx() == 27:
        #     cv2.destroyAllWindows()
        
        # plt.title('completed_depth')
        # plt.imshow(I)
        # plt.show()
        fname = os.path.basename(img_file)
        save_new_name = fname.split('.')[0] + '-showlabel.png'
        save_file = '{}/{}'.format(save_dir, save_new_name)
        print(save_file)
        cv2.imwrite(save_file, I)

def gen_random_points(height, width, point_num):
    if point_num <= 3:
        y_values = np.random.rand(point_num) * height
        x_values = np.random.rand(point_num) * width
        points = np.concatenate([x_values[:, np.newaxis], y_values[:, np.newaxis]], axis=1)
    else:
        y_values = np.random.rand(3) * height
        x_values = np.random.rand(3) * width

        remain_num = point_num - 3
        points = np.concatenate([x_values[:, np.newaxis], y_values[:, np.newaxis]], axis=1)
        while remain_num > 0:
            y = np.random.rand(1) * height
            x = np.random.rand(1) * width
            new_points = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)
            known_points = np.concatenate([points[0].reshape(1, -1), points[-2:]])
            in_poly, within_info = within_poly(known_points, new_points)
            crsval = within_info[0]
            if (crsval[0] * crsval[1] > 0) and \
                (crsval[0] * crsval[2] < 0) and \
                (crsval[1] * crsval[2] < 0):
                remain_num -= 1
                points = np.concatenate([points, new_points])

    return points

def draw_lines_from_points(points, I=None, height=100, width=100, check_point=None):
    if I is None:
        assert height > 0 and width > 0
        I = np.ones((height, width, 3)) * 255

    sides_vex = points.reshape(-1, 2).astype(np.int32)
    sides_color = [(255, 0, 0),(0, 0, 255),(255, 0, 255), (125,125,0)]
    for sid in range(len(sides_vex)):
        sp = sides_vex[sid]
        if sid != len(sides_vex) - 1:
            ep = sides_vex[sid+1]
        else:
            ep = sides_vex[0]
        color = sides_color[sid % len(sides_color)]
        cv2.line(I, sp, ep, color=color, thickness=2)
        cv2.putText(I, str(sid), (sp[0], sp[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=2)
    if check_point is not None:
        circle_color = (0, 255, 255)
        check_point = check_point.astype(np.int32)
        cv2.circle(I, check_point, 7, circle_color, -1)
    cv2.imshow("color_map", I)
    key = cv2.waitKey()
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()

def cross_value_2D(s, e, c):
    s_3 = np.zeros((1, 3))
    s_3[:, 0:2] = s
    e_3 = np.zeros((1, 3))
    e_3[:, 0:2] = e
    c_3 = np.zeros((1, 3))
    c_3[:, 0:2] = c
    vec_se = e_3 - s_3
    vec_sc = c_3 - s_3
    return np.cross(vec_se, vec_sc)

def within_poly(points, check_points):
    pnts_pairs = gen_pairs(points)
    is_in = []
    within_info = []
    for c in check_points:
        lines_crs = []
        for ps in pnts_pairs:
            s = ps[0]
            e = ps[1]
            crs = cross_value_2D(s, e, c)
            lines_crs.append(crs[0][-1])
        # print('lines_crs', lines_crs)
        if np.sum(np.array(lines_crs) < 0) == len(lines_crs) or np.sum(np.array(lines_crs) > 0) == len(lines_crs):
            in_poly = True
        else:
            in_poly = False
        is_in.append(in_poly)
        within_info.append(lines_crs)
    return is_in, within_info

def within_poly_test():
    poly_vertex = gen_random_points(720, 1280, 5)
    check_point = gen_random_points(720, 1280, 1)

    is_in, whithin_info = within_poly(poly_vertex, check_point)
    print(is_in)
    draw_lines_from_points(poly_vertex, height=720, width=1280, check_point=check_point[0])

#show polygon label
def init_show_label(label_color=False, folder_name=''):
    assert len(sys.argv) > 1, 'need png/json name'
    fname_ext = sys.argv[-1]

    if len(sys.argv) > 2:
        oper_type = sys.argv[1]
    else:
        oper_type = 'init'
    fname_split = fname_ext.split('_')
    image_folder = fname_split[0]
    if oper_type == 'crop':
        json_folder = 'crop_vis'
        save_dir = './saved_vis/{}/crop_label'.format(image_folder)
    elif oper_type == 'frame':
        json_folder = 'frame_vis'
        save_dir = './saved_vis/{}/frame_label'.format(image_folder)
    elif oper_type == 'vis_crop':
        json_folder_crop = 'crop_vis'
        json_folder_line = 'depth_vis'
        save_dir = './vis_and_crop/{}/frame_label'.format(image_folder)
    else:
        json_folder = 'depth_vis'
        save_dir = './saved_vis/{}/initial_label'.format(image_folder)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # for single file
    if len(fname_split) > 1:
        fname = fname_ext.split('.')[0]
        color_file = '/home/ly/data/datasets/trans-depth/{}/{}/aligned_depth_color/{}.png'.format(folder_name, image_folder, fname)
        if oper_type == 'vis_crop':
            json_label_crp = '/home/ly/data/datasets/trans-depth/{}/{}/aligned_depth_color/{}/{}.json'.format(folder_name, image_folder, json_folder_crop, fname)
            json_label_line = '/home/ly/data/datasets/trans-depth/{}/{}/aligned_depth_color/{}/{}.json'.format(folder_name, image_folder, json_folder_line, fname)
            json_label = [json_label_line, json_label_crp]
        else:
            json_label = '/home/ly/data/datasets/trans-depth/{}/{}/aligned_depth_color/{}/{}.json'.format(folder_name, image_folder, json_folder, fname)
      
            
        print('color_file', color_file)
        print('json_label', json_label)
        # 'wall':(128,0,128), 'window':(0,128,0), 'door':(0,128,128), 'guardrail':(0,0,128), 
        show_polygon_from_json(color_file, json_label, show=True, save_dir=save_dir, label_color=label_color, 
                    oper_type=oper_type, draw_marker=True, draw_centers=True, show_labels=['wall', 'window', 'door', 'guardrail'])
    else:
    # for files in a folder
        color_files_str = '/home/ly/data/datasets/trans-depth/{}/{}/aligned_depth_color'.format(folder_name, image_folder)
        json_labels_str = '/home/ly/data/datasets/trans-depth/{}/{}/aligned_depth_color/{}/*.json'.format(folder_name, image_folder, json_folder)
        json_labels = glob(json_labels_str)
        json_labels.sort()
        for jsn in json_labels:
            img = color_files_str + '/' + os.path.basename(jsn).split('.')[0]+'.png'
            print(img, jsn)
            show_polygon_from_json(img, jsn, show=True, save_dir=save_dir, label_color=label_color, oper_type=oper_type)


# show crop label
def cropped_show_label():
    assert len(sys.argv) > 1, 'need png/json name'
    fname_ext = sys.argv[-1]
    
    image_folder = fname_ext.split('_')[0]
    fname = fname_ext.split('.')[0]
    save_dir = './saved_vis/{}/cropped_label'.format(image_folder)
    color_file = '/home/ly/data/datasets/trans-depth/Glass-RGBD/images/{}.png'.format(fname)
    json_label = '/home/ly/data/datasets/trans-depth/Glass-RGBD/polygon_json/{}.json'.format(fname)
    print('color_file', color_file)
    print('json_label', json_label)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print('color_file', color_file)
    print('json_label', json_label)
        
    show_polygon_from_json(color_file, json_label, show=True, save_dir = save_dir)

def cropped_show_frames():
    save_dir = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Test/frames_vis'
    color_file = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/images'
    json_label = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/polygon_json'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print('color_file', color_file)
    print('json_label', json_label)
    
    rgb_pngs = glob(color_file+'/*.png')
    pnt_jsn = glob(json_label+'/*.json')
    rgb_pngs.sort()
    pnt_jsn.sort()
    for img_png, jsn in zip(rgb_pngs, pnt_jsn):
        print(img_png, jsn)
        show_frames_from_json(img_png, jsn, show=True, save_dir=save_dir)

def check_depth_with_different_line_sample_ratio(date_folder_s3, date_folder_s6):
    depth_s3 = date_folder_s3+'/completed_depth_npy'
    depth_s6 = date_folder_s6+'/aligned_depth_color/completed_depth_npy'
    ds3_npys = os.listdir(depth_s3)
    for d3ny in ds3_npys:
        ds3 = os.path.join(depth_s3, d3ny)
        ds6 = os.path.join(depth_s6, d3ny)
        ds3_mat = np.load(ds3).astype(np.int64)
        ds6_mat = np.load(ds6).astype(np.int64)
        diff = ds3_mat - ds6_mat
        dm = np.mean(diff)
        print(diff.shape)
        print('ds3_mat', np.unique(ds3_mat, return_counts=True))
        print('ds6_mat', np.unique(ds6_mat, return_counts=True))
        diff_value, diff_count = np.unique(diff, return_counts=True)

        print('diff', diff_value, 'mean', dm)
        numel = ds3_mat.shape[0]*ds3_mat.shape[1]
        dvr_list = {}
        for dv in diff_value:
            d_ratio = diff_count[diff_value == dv] / numel
            dvr_list[str(dv)] = str(d_ratio)
        print(dvr_list)
        print()

def copy_crop_png(from_dir, to_dir):
    depth_npys = glob(from_dir+'/*_filled.npy')
    print('depth_npys')
    for dth_npy in depth_npys:
        dpth_mat = np.load(dth_npy)
        dpth_color = cv2.applyColorMap(
            cv2.convertScaleAbs(dpth_mat, alpha=0.03), cv2.COLORMAP_JET)

        names = os.path.basename(dth_npy).split('_filled')[0]+'.png'
        write_png = to_dir+'/'+names
        print(write_png)
        cv2.imwrite(write_png, dpth_color)


if __name__ == '__main__':
    init_show_label(label_color=True, folder_name='')
    # check_depth_with_different_line_sample_ratio( '/home/ly/data/datasets/Glass-RGBD-Dense/20210927', 
    #                                     '/home/ly/data/datasets/trans-depth/20210927')
    # date_name = '20210928'
    # copy_crop_png(from_dir='/home/ly/data/datasets/trans-depth/{}/aligned_depth_color/completed_depth_npy'.format(date_name),
    #             to_dir='/home/ly/data/datasets/trans-depth/{}/aligned_depth_color/crop_vis'.format(date_name))

    # cropped_show_label()
    # cropped_show_frames()