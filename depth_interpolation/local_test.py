import enum
from itertools import count
import os
import glob
import random
from secrets import choice
from PIL import Image
from turtle import color, width
import numpy as np
import json
import cv2
import matplotlib
matplotlib.use('TKAgg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, mapping
import shutil
from tqdm import tqdm
import torch
from utils.sne_model import SNE

from commons import line_intersection, read_depth_npy, read_json_label, vis_depth_mat, decode_parsing_numpy, \
    gen_pairs, point_side_of_line, draw_polys, COLORS, intersect_remap, draw_points
from check_in_polygon import within_poly
from depth_interpolation import sample_points

def line_crop(crp_point_size, lines):
    i, j, w, h = crp_point_size
    lines = np.array(lines)
    cropped_lines = lines - np.array([j, i, j, i])
    
    eps = 1e-12

    # In dataset, we assume the left point has smaller x coord
    remove_x_min = cropped_lines[:, 2] < 0
    remove_x_max = cropped_lines[:, 0] > w
    remove_x = np.logical_or(remove_x_min, remove_x_max)
    keep_x = ~remove_x

    # there is no assumption on y, so remove lines that have both y coord out of bound
    remove_y_min = np.logical_and(cropped_lines[:, 1] < 0, cropped_lines[:, 3] < 0)
    remove_y_max = np.logical_and(cropped_lines[:, 1] > h, cropped_lines[:, 3] > h)
    remove_y = np.logical_or(remove_y_min, remove_y_max)
    keep_y = ~remove_y

    keep = np.logical_and(keep_x, keep_y)
    cropped_lines = cropped_lines[keep]
    clamped_lines = np.zeros_like(cropped_lines)

    for i,line in enumerate(cropped_lines):
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1 + eps)
        if x1 < 0:
            x1 = 0
            y1 = y2 + (x1 - x2) * slope
        if y1 < 0:
            y1 = 0
            x1 = x2 - (y2 - y1) / slope
        if x2 > w:
            x2 = w
            y2 = y1 + (x2 - x1) * slope
        if y2 > h:
            y2 = h
            x2 = x1 + (y2 - y1) / slope

        clamped_lines[i, :] = np.array([x1, y1, x2, y2])

    return clamped_lines


def main_readjust_coors():
    # lt_coor = [50, 100]
    # rb_coor = [800, 600]
    # poly_coors = np.array([[600, 300],[600, 800], [1000, 900], [1000, 200]])
    
    lt_coor = [490.2773109243698, 7.621848739495803]
    rb_coor = [1212.1260504201682, 698.3781512605042]
    # poly_coors = [ [ 489.436974789916, 142.91596638655463 ], [ 471.78991596638656, 705.1008403361345 ], 
    #                 [ 759.1848739495798, 705.9411764705883 ], [ 1249.9411764705883, 660.5630252100841 ], 
    #                 [ 1248.8986784140968, 215.41850220264317 ] ]
    poly_coors = [[1212.1260504201682, 698.3781512605042], [1215.1260504201682, 710.3781512605042], [1220.1260504201682, 720.3781512605042]]
    crop_1 = [lt_coor[0], rb_coor[1]]
    crop_2 = [rb_coor[0], lt_coor[1]]
    crop_coors = np.array([lt_coor, crop_1, rb_coor, crop_2])

    p1 = Polygon([lt_coor, crop_1, rb_coor, crop_2])
    p2 = Polygon(poly_coors)
    new_coors = p1.intersection(p2)
    new_coors_mapping = mapping(new_coors)
    print(new_coors)
    print(new_coors_mapping)
    print(len(new_coors_mapping['coordinates']))
    x, y = new_coors.exterior.coords.xy
    print(len(x), len(y))
    # new_coors = p2.intersection(p1)
    # new_coors = readjust_coors(lt_coor, rb_coor, poly_coors)

    crop_lines = gen_pairs(np.array(crop_coors))
    poly_lines = gen_pairs(np.array(poly_coors))
    a = draw_polys(crop_lines, width=1280, height=1000, color=COLORS[3], show=True)
    b = draw_polys(poly_lines, mat=a,  color=COLORS[4], show=True)
    # draw_polys(new_lines, mat=b, color=COLORS[6], show=True)
    # new_x, new_y = new_coors.exterior.coords.xy
    new_coors = list(new_coors.exterior.coords)

    new_coors = np.array(new_coors, dtype=np.uint16)
    new_lines = gen_pairs(new_coors)
    width = int(rb_coor[0] - lt_coor[0] + 1)
    height = int(rb_coor[1] - lt_coor[1] + 1)
    draw_polys(new_lines, width=1280, height=1000, color=COLORS[6], show=True)

def main_line_crop():
    lt_coor = [490.2773109243698, 7.621848739495803]
    rb_coor = [1212.1260504201682, 698.3781512605042]
    poly_coors = [ [ 489.436974789916, 142.91596638655463 ], [ 471.78991596638656, 705.1008403361345 ], 
                    [ 759.1848739495798, 705.9411764705883 ], [ 1249.9411764705883, 660.5630252100841 ], 
                    [ 1248.8986784140968, 215.41850220264317 ] ]
    # poly_coors = [[1212.1260504201682, 698.3781512605042], [1215.1260504201682, 710.3781512605042], [1220.1260504201682, 720.3781512605042]]

    crop_1 = [lt_coor[0], rb_coor[1]]
    crop_2 = [rb_coor[0], lt_coor[1]]
    crop_coors = np.array([lt_coor, crop_1, rb_coor, crop_2])

    crop_w = rb_coor[0] - lt_coor[0]
    crop_h = rb_coor[1] - lt_coor[1]
    crop_point_size = [lt_coor[0], lt_coor[1], crop_w, crop_h]
    # def line_crop(crp_point_size, lines):
    poly_lines = gen_pairs(np.array(poly_coors))
    poly_lines_resh = poly_lines.reshape(-1, 4)
    cropped_lines = line_crop(crop_point_size, poly_lines_resh)

    crop_lines = gen_pairs(np.array(crop_coors))
    origin_mat = draw_polys(crop_lines, width=1280, height=720, color=COLORS[3], show=False)
    draw_polys(poly_lines, mat=origin_mat, color=COLORS[5], show=True)

    cropped_lines = cropped_lines.reshape(-1, 2, 2)
    print(cropped_lines)
    draw_polys(cropped_lines, width=1280, height=720, color=COLORS[8], show=True)

def find_split_diff(all_names_dir, train_names_file, val_names_file):
    all_names = os.listdir(all_names_dir)
    all_names = [nm.split('.')[0] for nm in all_names]
    with open(train_names_file) as tf:
        train_names = tf.readlines()
    with open(val_names_file) as tf:
        val_names = tf.readlines()
    train_names = [tn.strip() for tn in  train_names]
    val_names = [vn.strip() for vn in  val_names]
    all_set = set(all_names)
    train_set = set(train_names)
    val_set = set(val_names)
    tv_set = train_set.union(val_set)
    print(all_set - tv_set)

def generate_id_name_map():
    ply_jsons = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/polygon_json'
    save_map_json = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/glassrgbd_images_1200.json'
    assert not os.path.isfile(save_map_json), save_map_json+' exists!'

    jnames = sorted(glob.glob(ply_jsons+"/*.json"))
    ids = []
    inames = []
    id_name_maps = {'images':[]}
    for jn in jnames:
        with open(jn, 'r') as f:
            lgt = json.load(f)
            ids.append(lgt['imageId'])
            inames.append(lgt['imagePath'])
    print('ids', ids)
    ids_set = set(ids)
    inames_set = set(inames)
    # confirm that no duplicate id or names
    assert len(ids) == len(ids_set), (len(ids), len(ids_set))
    assert len(inames) == len(inames_set), (len(inames), len(inames_set))

    for i, jn in enumerate(jnames):
        with open(jn, 'r') as f:
            lgt = json.load(f)
            lj = {'id':lgt['imageId'], 'file_name':lgt['imagePath'].split('.')[0]}
            print(i, lj)
            id_name_maps['images'].append(lj)
    with open(save_map_json, 'w') as f:
        json.dump(id_name_maps, f)

def centroid(vertexes):
     _x_list = [vertex [0] for vertex in vertexes]
     _y_list = [vertex [1] for vertex in vertexes]
     _len = len(vertexes)
     _x = sum(_x_list) / _len
     _y = sum(_y_list) / _len
     return(_x, _y)

def gen_poly_centers():
    fname = '20211104_173241_0'
    ply_jsons = '/home/ly/data/datasets/trans-depth/Glass-RGBD/polygon_json'
    images = '/home/ly/data/datasets/trans-depth/Glass-RGBD/images/'
    poly_shapes = read_json_label(ply_jsons+'/{}.json'.format(fname), key='shapes')
    I = cv2.imread(images+'/{}.png'.format(fname))
    from polylabel import polylabel

    for pid, ply in enumerate(poly_shapes):
        line_points = gen_pairs(np.array(ply['points']))
        j_points = []
        for i, line in enumerate(line_points):
            if i == 0:
                j_points = line.tolist()
            else:
                j_points.append(line[-1].tolist())
            
        # print(line_points)
        print(j_points)
        # center = polylabel([j_points])
        center = centroid(j_points)
        center_round = np.around(center).astype(np.uint16).tolist()
        print(center , center_round)
        I = draw_polys(line_points, color=COLORS[3], show=False, mat=I)
        cv2.circle(I, center_round, radius=10, color=COLORS[4], thickness=2)
    cv2.imshow("lin map", I)
    if cv2.waitKeyEx() == 27:
        cv2.destroyAllWindows()

def image_draw_circles(image, centers, color=None):
    if color is None:
        color = (120, 0, 215)
    for c in centers:
        print('circle center:', c)
        cv2.circle(image, c, radius=4, color=color, thickness=4)
    return image

def draw_poly_joints():
    # w, h = 600, 722
    h, w = 1024, 1024
    # crp_point = [[147, 48], [147, 478], [540, 478], [540, 48]]
    # src_py_points =   [[595.6534423828125, 142.5537567138672], [203.59317016601562, 226.8690948486328], 
    #                     [203.85147094726562, 149.95127868652344], [592.3250122070312, 27.705894470214844], 
    #                     [595.6534423828125, 142.5537567138672]]

    crp_point = [[144, 16], [144, 461], [652, 461], [652, 16]]
    src_py_points = [[273.7486572265625, 456.5132141113281], [273.7486572265625, 500.0000305175781], 
            [402.96783447265625, 458.9632263183594], [403.9447937011719, 500.0000305175781], [273.7486572265625, 500.0000305175781]]

    crp_p = gen_pairs(np.array(crp_point))
    src_p = gen_pairs(np.array(src_py_points))
    
    # jnt_p_l = intersect_remap(crp_point, src_py_points)
    # jnt_p = gen_pairs(np.array(jnt_p_l))

    # center = centroid(jnt_p_l)
    I = draw_polys(crp_p, show=False, width=w, height=h)
    I = draw_polys(src_p, mat=I, show=False, color=(255,0,0)) #  blue
    # I = draw_polys(jnt_p, mat=I, show=False, color=(0,0,255)) # red
    # I = image_draw_circles(I, [[int(center[0]), int(center[1])]], color=COLORS[-1])
    cv2.imshow("lin map", I)
    if cv2.waitKeyEx() == 27:
        cv2.destroyAllWindows()

def plt_imshow(im):
    plt.close()
    sizes = im.shape
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, sizes[1] - 0.5])
    plt.ylim([sizes[0] - 0.5, -0.5])
    plt.imshow(im)

def draw_by_matplot():
    imfile = '/home/ly/data/datasets/trans-depth/Glass-RGBD/images/20211104_173241_0.png'
    linejsn = '/home/ly/data/datasets/trans-depth/Glass-RGBD/polygon_json/20211104_173241_0.json'
    im = cv2.imread(imfile)
    jsn = read_json_label(linejsn, key='shapes')
    plt_imshow(im)
    npz_name = '20211104_173241_0'
    for i, ginst in enumerate(jsn):
        p = ginst['points']
        pp = gen_pairs(np.array(p))
        for line in pp:
            s, e = line
            plt.plot([s[0], e[0]], [s[1], e[1]], c="orange", linewidth=0.5)
        cen = centroid(p)
        plt.plot([s[0], e[0]], [s[1], e[1]], c="orange", linewidth=0.5)
        plt.plot([int(cen[0])], [int(cen[1])], "ro")


        # plt.scatter(a[1], a[0], **PLTOPTS)
        # plt.scatter(b[1], b[0], **PLTOPTS)
    plt.savefig(npz_name+"_gt.png", dpi=500, bbox_inches=0)

def vis_img_copy():
    date_str = '20211107'
    im_dir = '/home/ly/data/datasets/trans-depth/zhouyang_rgbd/{}/images'.format(date_str)
    d_vis_dir = '/home/ly/data/datasets/trans-depth/zhouyang_rgbd/{}/depth_vis'.format(date_str)
    from_dir = '/home/ly/data/datasets/trans-depth/{}/aligned_depth_color/depth_vis'.format(date_str)
    ilist = os.listdir(im_dir)
    for im in ilist:
        src_file = from_dir+'/'+im
        to_file = d_vis_dir+'/'+im
        print(im)
        shutil.copy(src_file, to_file)
        
def rename_files():
    folder = '/home/ly/data/datasets/trans-depth/20220430/aligned_depth_color/depth_vis'
    fs = os.listdir(folder)
    for fname in fs:
        fsplits = fname.split('_')
        dsplits = fsplits[0].split('-')
        ds = dsplits[:3]
        ts = dsplits[3:]
        d = ''.join(ds)
        t = ''.join(ts)
        print(d, t)
        new_name = "{}_{}_0.{}".format(d, t, fname.split('.')[-1])
        oldfile = os.path.join(folder, fname)
        newfile = os.path.join(folder, new_name)
        print('oldfile', oldfile)
        print('newfile', newfile)
        os.rename(oldfile, newfile)

def crop_first():
    rgb_png = '/home/ly/data/datasets/trans-depth/20210927-copy/aligned_depth_color/20210927_172444-angle2-fullcover_0.png'
    rgb_nocover_png = '/home/ly/data/datasets/trans-depth/20210927-copy/aligned_depth_color/20210927_175219-angle2-nocover_0.png'
    depth_coarse_npy = '/home/ly/data/datasets/trans-depth/20210927-copy/aligned_depth_color/20210927_172444-angle2-fullcover_0.npy' 
    depth_vis_png = '/home/ly/data/datasets/trans-depth/20210927-copy/aligned_depth_color/depth_vis/20210927_172444-angle2-fullcover_0.png'
    crop_json = '/home/ly/data/datasets/trans-depth/20210927-copy/aligned_depth_color/20210927_172444-angle2-fullcover_0_precrop.json'
    

    save_dir = '/home/ly/data/datasets/trans-depth/20210927-copy/aligned_depth_color/precrop'
    points = read_json_label(crop_json, 'shapes')[0]['points']
    depth_mat = read_depth_npy(depth_coarse_npy)
    I_rgb = cv2.imread(rgb_png)
    I_rgb_nocv = cv2.imread(rgb_nocover_png)
    I_dep = cv2.imread(depth_vis_png)

    points = np.array(points, dtype=np.int32)
    left_up_corner = points[0]
    right_lower_corner = points[1]
    col_left, row_top = left_up_corner
    col_right, row_botm = right_lower_corner
    crop_dmat = depth_mat[row_top:row_botm+1, col_left:col_right+1]
    crop_rgb = I_rgb[row_top:row_botm+1, col_left:col_right+1]
    crop_rgb_nocv = I_rgb_nocv[row_top:row_botm+1, col_left:col_right+1]
    crop_dvis = I_dep[row_top:row_botm+1, col_left:col_right+1]

    name = os.path.basename(rgb_png).split('.')[0]
    name_nocv = os.path.basename(rgb_nocover_png).split('.')[0]
    cv2.imwrite(save_dir+'/{}.png'.format(name), crop_rgb)
    cv2.imwrite(save_dir+'/{}.png'.format(name_nocv), crop_rgb_nocv)
    cv2.imwrite(save_dir+'/{}_depvis.png'.format(name), crop_dvis)
    np.save(save_dir+'/{}'.format(name), crop_dmat)

    # cv2.imshow('1', crop_rgb)
    # if cv2.waitKeyEx() == 27:
    #     cv2.destroyAllWindows()

def crop_image_by_label(folder, json_label, save_dir):
    points = read_json_label(json_label, 'shapes')[0]['points']
    points = np.array(points, dtype=np.int32)
    left_up_corner = points[0]
    right_lower_corner = points[1]
    col_left, row_top = left_up_corner
    col_right, row_botm = right_lower_corner
    # img_files = os.listdir(folder)

    img_files = glob.glob(folder+"/*.png")
    for im in img_files:
        # fname = os.path.join(folder, im)
        fname = im
        print(fname)
        I_rgb = cv2.imread(fname)
        cropped_I = I_rgb[row_top:row_botm+1, col_left:col_right+1]
        name_I = os.path.basename(fname)
        save_file = os.path.join(save_dir, name_I)
        print('save_file', save_file)
        cv2.imwrite(save_file, cropped_I)


def depth_interpolation_sample_vis():
    img_dir = '/home/ly/data/datasets/trans-depth/20210927copy/aligned_depth_color/precrop/'
    init_json = '/home/ly/data/datasets/trans-depth/20210927copy/aligned_depth_color/depth_vis/20210927copy_172444-angle2-fullcover_0_precrop.json'
    img = cv2.imread(img_dir+'20210927_172444-angle2-fullcover_0_precrop.png', cv2.IMREAD_COLOR)
    ply_labels = read_json_label(init_json, key='shapes')
    
    sides_pixels = [
    [[401, 157], [400.99395751953125, 157.22894287109375], [399.81842041015625, 202.91427612304688], [398.7027893066406, 248.38546752929688], 
    [397.6685485839844, 293.6189270019531], [396.72930908203125, 338.57501220703125], [395.8905029296875, 383.20599365234375], [395.1495056152344, 427.4638977050781], 
    [394.4960021972656, 471.30767822265625], [393.91314697265625, 514.7093505859375], [393.378662109375, 557.6585693359375], [392.8663330078125, 600.1649169921875], 
    [392.34796142578125, 642.2590942382812], [392, 669]], 
    
    [[392, 669], [392.21356201171875, 669.0064697265625], [433.36407470703125, 670.26171875], [471.6427307128906, 671.422607421875], 
    [507.34539794921875, 672.4814453125], [540.7205810546875, 673.4381103515625], [561, 674]], 
    
    [[561, 674], [561.1729736328125, 673.9600830078125], [594.87548828125, 666.1585083007812], [626.8765869140625, 658.7119750976562], 
    [657.2904663085938, 651.5928955078125], [686.2203369140625, 644.7791748046875], [713.7608642578125, 638.2529296875], [740.0, 631.9988403320312], 
    [765.019775390625, 626.0033569335938], [788.89697265625, 620.25439453125], [811.7037353515625, 614.7406005859375], [833.5074462890625, 609.451171875], 
    [854.3715209960938, 604.3761596679688], [874.355224609375, 599.5055541992188], [893.513671875, 594.8300170898438], [911.8984375, 590.34033203125], 
    [929.557373046875, 586.027587890625], [946.5352783203125, 581.8834228515625], [962.8732299804688, 577.8994140625], [978.609619140625, 574.0677490234375], [983, 573]], 
    
    [[983, 573], [982.9996337890625, 572.87744140625], [982.9300537109375, 548.3744506835938], 
    [982.8609619140625, 523.8731689453125], [982.7870483398438, 499.37127685546875], [982.7032470703125, 474.8681945800781], [982.605224609375, 450.36474609375], 
    [982.4892578125, 425.86328125], [982.3525390625, 401.3670959472656], [982.193115234375, 376.8805236816406], [982.0096435546875, 352.40838623046875], 
    [981.802001953125, 327.95574951171875], [981.5706787109375, 303.52764892578125], [981.3172607421875, 279.1287841796875], [981.0438232421875, 254.76312255859375], [981, 251]], 


    [[981, 251], [980.9318237304688, 250.98863220214844], [967.0130615234375, 248.6640625], 
    [952.4965209960938, 246.2356414794922], [937.3399658203125, 243.69674682617188], [921.4970703125, 241.0403594970703], [904.9171752929688, 238.2589111328125], 
    [887.5449829101562, 235.34445190429688], [869.3199462890625, 232.28839111328125], [850.176025390625, 229.08175659179688], [830.0408935546875, 225.71495056152344], 
    [808.8358764648438, 222.17787170410156], [786.47509765625, 218.45982360839844], [762.865234375, 214.5496063232422], [737.90478515625, 210.43527221679688], 
    [711.4838256835938, 206.1042938232422], [683.4830932617188, 201.5431671142578], [653.7734375, 196.7373504638672], [622.2146606445312, 191.67059326171875], 
    [588.65380859375, 186.32447814941406], [552.9224853515625, 180.67681884765625], [514.8319702148438, 174.70005798339844], [474.1658020019531, 168.35816955566406], 
    [430.6668701171875, 161.60220336914062], [401, 157]]
    
    ]

    start_pixels = sample_points(sides_pixels, sample_ratio=0.05, min_side_inter_point_num=3)
    end_pixels = sample_points(sides_pixels, sample_ratio=0.05, min_side_inter_point_num=3)
    # end_pixels = []
    # for i, px in enumerate(sides_pixels):
    #     end_pixels += px
    
    for start_px in tqdm(start_pixels, desc='start points:{}, end points:{}'.format(len(start_pixels), len(end_pixels))):
        for end_px in end_pixels:
            one_line = np.array([[start_px, end_px]])
            img = draw_polys(one_line, mat=img, show=False, thickness=1)
    
    first_door = ply_labels[0]['points']
    pairs = gen_pairs(np.array(first_door))
    img = draw_polys(pairs, mat=img, show=False, color=(255, 0, 0), thickness=6) # blue line


    def in_sublists(l, lol):
        l_int = list(map(int, l))
        for sublist in lol:
            sub_int = list(map(round, sublist))
            if abs(sub_int[0] - l_int[0]) <= 2  and abs(sub_int[1] - l_int[1]) <= 2:
                return True
        return False

    start_pixels_show = []
    for px in start_pixels:
        if not in_sublists(px, first_door):
            start_pixels_show.append(px)
    
    end_pixels_show = []
    for px in end_pixels:
        if not in_sublists(px, first_door) and not in_sublists(px, start_pixels_show):
            end_pixels_show.append(px)

    # start_pixels_show += end_pixels_show
    print('end_pixels_show', end_pixels_show)
    draw_points(start_pixels_show, mat=img, show=False, color=(0, 255,255))
    draw_points(end_pixels_show, mat=img, show=False, color=(255, 0,255))
    draw_points(first_door, mat=img, show=False, color=(0, 0, 255)) # red junction points

    save_png = '/home/ly/workspace/my_linux_lib/realsense_libs/depth_generation/saved_vis/20210927copy/initial_label/interpolation_expample_172444_s0.05.png'

    cv2.imwrite(save_png, img)
    cv2.imshow('1', img)
    if cv2.waitKeyEx() == 27:
        cv2.destroyAllWindows()

def gen_val_split(val_num, ):
    merge_val_files = ['/home/ly/data/datasets/trans-depth/Glass-RGBD/val.txt',
    '/home/ly/data/datasets/trans-depth/Glass-RGBD-Test/val_unseen.txt']
    merge_train_file = '/home/ly/data/datasets/trans-depth/Glass-RGBD/train.txt'
    save_dir = '/home/ly/data/datasets/trans-depth/Glass-RGBD/'
    all_files = '/home/ly/data/datasets/trans-depth/Glass-RGBD/images'
    fname_val_list = []
    for fil in merge_val_files:
        with open(fil, 'r') as f:
            filenames = f.readlines()
            fname_val_list += filenames
    
    with open(merge_train_file, 'r') as f:
        fname_train_list = f.readlines()

   
    fname_list = os.listdir(all_files)
    fname_list = [fn.split('.')[0] for fn in fname_list]
    fname_val_list = [fn.strip() for fn in fname_val_list]
    fname_train_list = [fn.strip() for fn in fname_train_list]
    set_all = set(fname_list)
    set_val = set(fname_val_list)
    set_train = set(fname_train_list) - set_val
    gen_num = val_num - len(set_val)

    set_remain = (set_all - set_train - set_val)
    remain_names = list(set_remain)

    order_range = list(range(len(remain_names)))
    random.shuffle(order_range)
    val_ids = order_range[0:gen_num]
    new_val_names = [remain_names[idx] for idx in val_ids]

    train_ids = order_range[gen_num:]
    new_train_names = [remain_names[idx] for idx in train_ids]
    
    new_train_names += list(set_train)
    new_val_names += list(set_val)

    assert  len(new_train_names) + len(new_val_names) == len(fname_list), (len(new_train_names),len(new_val_names),len(fname_list))
    assert set(new_train_names) - set(new_val_names) == set(new_train_names)

    new_train_names.sort()
    new_val_names.sort()
    with open(save_dir+'/train.txt', 'w') as f:
        for fnm in new_train_names:
            f.write(fnm+'\n')
    with open(save_dir+'/val.txt', 'w') as f:
        for fnm in new_val_names:
            f.write(fnm+'\n')

def random_sample_val_by_scenes():
    scene_info = '/home/ly/data/datasets/trans-depth/scene_info.txt'
    with open(scene_info, 'r') as f:
        a = f.readlines()
        sc_counts = []
        for sc in a:
            sc_cnt = sc.strip()
            sc_cnt_num = sc_cnt.split(',')[1]
            sc_counts.append(int(sc_cnt_num.strip()))
        print(sc_counts)
    sc_arr = np.array(sc_counts)

    total = np.sum(sc_arr)
    print('total:', total)
    choose_num = [1, 1, 1, 3, 3]
    index_all = np.array(list(range(total)))
    
    idx40 = np.where(sc_arr > 40)[0]
    b40 = sc_arr[idx40]
    print('>40', 'idx', index_all[idx40], 'num', b40, np.sum(b40), np.sum(b40) / total)
    idx_all = np.copy(index_all)
    choice_pool = idx_all[idx40]
    random.shuffle(choice_pool)
    ch_b40 = choice_pool[:choose_num[0]]
    
    
    b30 = np.where(sc_arr > 30)[0]
    le40 = np.where(sc_arr <= 40)[0]
    idx30_40 = np.intersect1d(b30, le40)
    inter30_40 = sc_arr[idx30_40]
    print('>30,<=40', 'idx', index_all[idx30_40], 'num', inter30_40, np.sum(inter30_40), np.sum(inter30_40) / total)
    idx_all = np.copy(index_all)
    choice_pool = idx_all[idx30_40]
    random.shuffle(choice_pool)
    ch_b30_le40 = choice_pool[:choose_num[1]]
    
    
    b20 = np.where(sc_arr > 20)[0]
    le30 = np.where(sc_arr <= 30)[0]
    idx20_30 = np.intersect1d(b20, le30)
    inter20_30 = sc_arr[idx20_30]
    print('>20,<=30', 'idx', index_all[idx20_30], 'num', inter20_30, np.sum(inter20_30), np.sum(inter20_30) / total)
    idx_all = np.copy(index_all)
    choice_pool = idx_all[idx20_30]
    random.shuffle(choice_pool)
    ch_b20_le30 = choice_pool[:choose_num[2]]
    

    b10 = np.where(sc_arr > 10)[0]
    le20 = np.where(sc_arr <= 20)[0]
    idx10_20 = np.intersect1d(b10, le20)
    inter10_20 = sc_arr[idx10_20]
    print('>10,<=20', 'idx', index_all[idx10_20], 'num', inter10_20, np.sum(inter10_20), np.sum(inter10_20) / total)
    idx_all = np.copy(index_all)
    choice_pool = idx_all[idx10_20]
    random.shuffle(choice_pool)
    ch_b10_le20 = choice_pool[:choose_num[3]]
    
    
    idx10 = np.where(sc_arr <= 10)[0]
    le10 = sc_arr[idx10]
    print('<=10', 'idx', index_all[idx10], 'num', le10, np.sum(le10), np.sum(le10) / total)
    idx_all = np.copy(index_all)
    choice_pool = idx_all[idx10]
    random.shuffle(choice_pool)
    ch_le10 = choice_pool[:choose_num[4]]
    
    print()
    print('chossen ids for > 40', ch_b40)
    print('chossen ids for >30,<=40', ch_b30_le40)
    print('chossen ids for >20,<=30', ch_b20_le30)
    print('chossen ids for >10,<=20', ch_b10_le20)
    print('chossen ids for <=10', ch_le10)
    total_choose_num = np.sum(sc_arr[ch_b40]) + np.sum(sc_arr[ch_b30_le40]) + np.sum(sc_arr[ch_b20_le30]) + np.sum(sc_arr[ch_b10_le20]) + np.sum(sc_arr[ch_le10])
    print('total_choose_num', total_choose_num)

def gen_train_split():
    val_new_file = '/home/ly/data/datasets/trans-depth/val_new_182.txt'
    all_files = '/home/ly/data/datasets/trans-depth/Glass-RGBD/images'
    save_val_file = '/home/ly/data/datasets/trans-depth/val_182.txt'
    save_train_file = '/home/ly/data/datasets/trans-depth/train_182.txt'

    with open(val_new_file, 'r') as f:
        fname_val_list = f.readlines()

    fname_list = os.listdir(all_files)
    fname_list = [fn.split('.')[0] for fn in fname_list]
    val_list = [fn.strip() for fn in fname_val_list]

    set_all = set(fname_list)
    set_val = set(val_list)
    set_train = set_all - set_val
    train_list = list(set_train)
    val_list.sort()
    train_list.sort()

    with open(save_val_file, 'w') as f:
        for fnm in val_list:
            f.write(fnm+'\n')
    print(save_val_file)
    
    with open(save_train_file, 'w') as f:
        for fnm in train_list:
            f.write(fnm+'\n')
    print(save_train_file)

def statistic_capturing_time():
    fname = '/home/ly/data/datasets/trans-depth/scene_capaturing_time.txt'
    with open(fname, 'r') as f:
        fname_list = f.readlines()
    f_list = [fn.strip() for fn in fname_list]
    day_scene_num, sun_scene_num, night_scene_num, indoor_scene_num = 0, 0, 0, 0
    day_num, sun_num, night_num, indoor_num = 0, 0, 0, 0
    for finfo in f_list:
        fin_list = finfo.split(',')
        fi_list = [fi.strip() for fi in fin_list]
        if fi_list[-1] == 'd':
            day_scene_num+=1
            day_num += int(fi_list[1])
        elif fi_list[-1] == 's':
            sun_scene_num += 1
            sun_num += int(fi_list[1])
        elif fi_list[-1] == 'n':
            night_scene_num += 1
            night_num += int(fi_list[1])
        elif fi_list[-1] == 'ind':
            indoor_scene_num += 1
            indoor_num += int(fi_list[1])
    print('scene: day_num:{}, sun_num:{}, night_num:{}, indoor_num:{}'.format(day_num, sun_num, night_num, indoor_num))
    print('images: day_scene_num:{}, sun_scene_num:{}, night_scene_num:{}, indoor_scene_num:{}'.format(day_scene_num, 
        sun_scene_num, night_scene_num, indoor_scene_num))

def glass_instance_num():
    from raw_preprocess import GLASS_LABELS
    dates_dir = [ '20210927', '20210928', '20210929', '20211002', '20211004', '20211016', 
    '20211017', '20211018', '20211019', '20211020', '20211021', '20211027', '20211028', 
    '20211031', '20211104', '20211105', '20211107', '20211111', '20220430', '20220514',]
    json_dir_str = '/home/ly/data/datasets/trans-depth/{}/aligned_depth_color/depth_vis'
    json_dirs = [json_dir_str.format(dds) for dds in dates_dir]

    instance_num_dict = {key:0 for key in GLASS_LABELS}
    for jdr in json_dirs:
        d_all_jsn = glob.glob(jdr+'/*.json')
        for dj in d_all_jsn:
            img_polys = read_json_label(dj, 'shapes')
            for poly in img_polys:
                if poly['label'] in GLASS_LABELS:
                    instance_num_dict[poly['label']] += 1
    print(instance_num_dict)

def pixel_ratio():
    imgs = '/home/ly/data/datasets/trans-depth/Glass-RGBD/images'
    jsons = '/home/ly/data/datasets/trans-depth/Glass-RGBD/polygon_json'
    segs = '/home/ly/data/datasets/trans-depth/Glass-RGBD/segmentation'
    from raw_preprocess import LABELS_ID, LABELS_ID_MAP
    seg_pngs = glob.glob(segs+'/*.png')

    cls_ratio = [[] for _ in LABELS_ID]
    ratio_maps = {str(r):0 for r in range(1, 11)}
    total_glass_ratio = 0
    for sp in seg_pngs:
        s_png = cv2.imread(sp, cv2.IMREAD_UNCHANGED)
        H, W = s_png.shape
        total_num = H * W
        for id in LABELS_ID:
            clss_p_num = np.sum(s_png == id)
            cls_ratio[id - 1].append(clss_p_num / total_num)
        
        glass_image_ratio = np.sum(s_png > 0) / (H*W)
        gir_10 = glass_image_ratio * 10
        print(glass_image_ratio, gir_10)
        r_key = str(int(np.floor(gir_10) + 1))
        ratio_maps[r_key] += 1
        total_glass_ratio += glass_image_ratio

    for i, r in enumerate(cls_ratio):
        total_ratio = np.sum(r) / len(r)
        print(i, total_ratio)
    
    total_glass_ratio = total_glass_ratio / len(seg_pngs)
    print('glss to image ratio:', ratio_maps)
    print('total_glass_ratio', total_glass_ratio)

def depth_range():
    min_depth_mm = 1
    max_depth_mm = 10000
    imgs = '/home/ly/data/datasets/trans-depth/Glass-RGBD/images'
    depts = '/home/ly/data/datasets/trans-depth/Glass-RGBD/depth'
    segs = '/home/ly/data/datasets/trans-depth/Glass-RGBD/segmentation'
    from raw_preprocess import LABELS_ID, LABELS_ID_MAP
    seg_pngs = glob.glob(segs+'/*.png')
    dep_pngs = glob.glob(depts+'/*.png')
    assert len(seg_pngs) == len(dep_pngs)
    seg_pngs.sort()
    dep_pngs.sort()

    cls_ratio = [[] for _ in LABELS_ID]
    total_glass_ratio = 0
    i = 0
    total_min, total_max, total_avg = 0, 0, 0
    depth_range_maps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dpeth_ranges = [
        [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10]],
        [[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10]],
        [[2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10]],
        [[3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10]],
        [[4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10]],
        [[5, 6], [5, 7], [5, 8], [5, 9], [5, 10]],
        [[6, 7], [6, 8], [6, 9], [6, 10]], 
        [[7, 8], [7, 9], [7, 10]],
        [[8, 9], [8, 10]],
        [[9, 10]]
    ]
    range_keys = []
    for dr in dpeth_ranges:
        for rng in dr:
            range_keys.append(str(rng[0])+'-'+str(rng[1]))
    range_counts = {k:0 for k in range_keys}
    range_min_max = {k:[max_depth_mm, 0] for k in range_keys}

    for i, (sp, dp) in enumerate(zip(seg_pngs, dep_pngs)):
        assert os.path.basename(sp) == os.path.basename(dp), (os.path.basename(sp), os.path.basename(dp))
        s_mat = cv2.imread(sp, cv2.IMREAD_UNCHANGED)
        d_mat = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if s_mat.shape != d_mat.shape:
            s_mat = cv2.resize(s_mat, dsize=(d_mat.shape[1], d_mat.shape[0]), interpolation=cv2.INTER_NEAREST)
            print('resized', (os.path.basename(sp), os.path.basename(dp)), s_mat.shape, d_mat.shape)

        glass_depth = d_mat[s_mat > 0]
        glass_depth_min = glass_depth[glass_depth >= min_depth_mm]
        glass_depth_valid = glass_depth_min[glass_depth_min <= max_depth_mm]
        dvalues = np.unique(glass_depth_valid)
        min = np.min(dvalues) / 1000.0
        max = np.max(dvalues) / 1000.0
        avg = np.mean(dvalues) / 1000.0
        min_floor = np.floor(min)
        max_ceil = np.ceil(max)
        r_key = str(int(min_floor))+'-'+str(int(max_ceil))
        assert r_key in range_counts, r_key
        range_counts[r_key] += 1
        if min < range_min_max[r_key][0]:
            range_min_max[r_key][0] = min
        if max > range_min_max[r_key][1]:
            range_min_max[r_key][1] = max

        print(os.path.basename(sp), ' ', min , max, avg, ' ', (min_floor, max_ceil))
    print(range_counts)
    for k, v in range_counts.items():
        if v > 0:
            print(k, v, ' min_max', range_min_max[k])

def get_camera_intrisic(type='intelD455'):
    assert type in ['intelD455', 'kinectV2'], type
    intel_camParam = torch.tensor([[6.360779e+02, 0.000000e+00, 6.348217e+02],
                             [0.000000e+00, 6.352265e+02, 3.570233e+02],
                             [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=torch.float32)  # camera parameters
    intel_size = (728, 1280)

    azure_camParam = torch.tensor([[504.4602356,  0.000000e+00, 321.77764893],
                                [0.000000e+00,   504.88800049, 318.09918213],
                                [0.000000e+00,   0.000000e+00, 1.000000e+00]], dtype=torch.float32)  # camera parameters
    azure_size = (728, 1280)
    if type == 'intelD455':
        return intel_camParam, intel_size
    else:
        return azure_camParam, azure_size


def depth_2_xy():
    camParam, size = get_camera_intrisic(type='intelD455')
    h = size[0]
    w = size[1]

    v_map, u_map = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    v_map = v_map.type(torch.float32)
    u_map = u_map.type(torch.float32)

    depth_png = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/depth/20210927_164622-angle1-origin_0.png'
    door_points = np.array([ [39.34929023323336, 690.6],
			[910.674449339207, 657.9898678414097],
			[907.4052631578948, 180.94688995215313],
			[38.86828193832599, 132.43920704845814],
			[16.053395733770714, 690.6] ])
    im = Image.open(depth_png)
    depth_mat = np.array(im)
    door_points = np.floor(door_points)
    

    print(depth_mat.shape, np.unique(depth_mat))
    exit()

    Z = depth   # h, w
    Y = Z.mul((v_map - camParam[1,2])) / camParam[0,0]  # h, w
    X = Z.mul((u_map - camParam[0,2])) / camParam[0,0]  # h, w
    Z[Y <= 0] = 0
    Y[Y <= 0] = 0
    Z[torch.isnan(Z)] = 0

def filter_image():
    depth_pngs = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/images/'
    pngs = glob.glob(depth_pngs+'/*.png')
    num = 0
    for i, sp in enumerate(pngs):
        img = cv2.imread(sp, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        if h * w < 256**2:
            print(os.path.basename(sp))
            print(img.shape)
            num+=1
    print(num)

if __name__ == '__main__':
    # vis_img_copy()
    # rename_files()
    # depth_interpolation_sample_vis()
    # crop_image_by_label('/home/ly/Documents/papers/my_pappers/trans-depth/figures', 
    #     '/home/ly/Documents/papers/my_pappers/trans-depth/figures/20210927_172444-cover_depthvis.json', 
    #     '/home/ly/Documents/papers/my_pappers/trans-depth/figures/cropped')
    # random_sample_val()
    # statistic_capturing_time()
    # glass_instance_num()
    # depth_range()
    # pixel_ratio()
    # generate_id_name_map()
    # depth_2_xy()
    # filter_image()
    # gen_train_split()
    pass