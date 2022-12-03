import json
import os
import struct
import numpy as np
import cv2
from PIL import Image
from numpy.lib import save

# from pycocotools.coco import COCO

# bin_file = "/home/ly/workspace/my_linux_lib/realsense_libs/pyrealsense_exp/out/converted/depth_bin/_Depth_1630316334593.71362304687500.bin"
color_file = "/home/ly/data/datasets/trans-depth/navigation/window-439-1-4/png/_Color_1631160511667.82128906250000.png"
raw_file = "/home/ly/data/datasets/trans-depth/navigation/window-439-1-4/raw/_Depth_1631160511669.74804687500000.raw"
bin_file = "/home/ly/data/datasets/trans-depth/navigation/window-439-1-1/bin/_Depth_1631160228630.56762695312500.bin"
save_dir = "/home/ly/data/datasets/trans-depth/navigation/realsense-viewer/20210914_150348/dpth_png"
def read_raw_depth(width, height, raw_depth_file):
    dpth_mat = np.zeros(width*height, dtype=np.uint16)
    with open(raw_depth_file, "rb") as f:
        num = 0
        byte = f.read(2)
        while byte != b"":
            # Do stuff with byte.
            v = struct.unpack('H', byte)
            dpth_mat[num] = v[0]
            num+=1
            byte = f.read(2)
        dpth_mat = dpth_mat.reshape(height, width)
        # vis_depth_mat(dpth_mat, color_file, height, width)
        return dpth_mat

def read_json_label(json_label, key=None):
    with open(json_label) as f:
        annos = json.load(f)
        if key is not None:
            assert key in annos, 'key {} not exists in keys:{}'.format(key, str(annos.keys()))
            return annos[key]
        else:
            return annos
        
def vis_depth_mat(dpth_mat, height=None, width=None, plt_show=False,):
    # min, max, min_loc, max_loc = cv2.minMaxLoc(dpth_mat)
    if height is None or width is None:
        height, width = dpth_mat.shape[-2:]
    # dpth_arr = np.zeros((height, width), dtype=np.uint8)
    # max = 6000
    # cv2.convertScaleAbs(dpth_mat, dpth_arr, 255/max)
    # dpth_color = cv2.applyColorMap(dpth_arr, cv2.COLORMAP_JET)

    dpth_color = cv2.applyColorMap(
            cv2.convertScaleAbs(dpth_mat, alpha=0.03), cv2.COLORMAP_JET)

    if plt_show:
        cv2.imshow("dpth_arr", dpth_color)
        if cv2.waitKeyEx() == 27:
            cv2.destroyAllWindows()
    else:
        return dpth_color
    

def read_depth_npy(depth_file, allow_pickle=False):
    with open(depth_file, 'rb') as f:
        dpth_mat = np.load(f, allow_pickle=allow_pickle)
        return dpth_mat

def vis_img_polygon_coco(img_file, json_file, raw_depth_file=None, depth_npy_file=None, height=None, width=None, save=False):
    assert raw_depth_file is not None or depth_npy_file is not None

    I = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    if height is None or width is None:
        height, width = I.shape[0:2]

    if raw_depth_file is not None:
        raw_depth = read_raw_depth(width, height, raw_depth_file)
    else:
        raw_depth = read_depth_npy(depth_npy_file)
        raw_depth = raw_depth.astype(np.uint16)

    raw_depth_colored = vis_depth_mat(raw_depth, plt_show=False)

    coco = COCO(json_file)

    for imid, iminfo in coco.imgs.items():
        print('imid:', imid, ' iminfo:', iminfo)
        annids = coco.getAnnIds(imgIds=[imid])
        anns = coco.loadAnns(annids)
        for idx, ann in enumerate(anns):
            sides_vex = np.array(ann['segmentation'], dtype=np.float32)
            sides_vex = np.floor(sides_vex.reshape(-1, 2)).astype(np.int32)
            # sides_vex = sides_vex.reshape(-1, 2)
            print(idx, 'c1', sides_vex)
            for sid in range(len(sides_vex)):
                if sid % 2 == 0:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)

                if sid != len(sides_vex) - 1:
                    cv2.line(I, sides_vex[sid], sides_vex[sid+1], color=color, thickness=2)
                    cv2.line(raw_depth_colored, sides_vex[sid], sides_vex[sid+1], color=color, thickness=2)
                else:
                    cv2.line(I, sides_vex[sid], sides_vex[0], color=color, thickness=2)
                    cv2.line(raw_depth_colored, sides_vex[sid], sides_vex[0], color=color, thickness=2)
    cv2.imshow('image', I)
    cv2.imshow('raw_depth', raw_depth_colored)
    if save:
        json_dir = os.path.dirname(json_file)
        save_dir = json_dir + '/vis_png'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_img_vis_name = save_dir+'/img_sides_vis.png'
        save_dpth_vis_name = save_dir+'/depth_sides_vis.png'
        cv2.imwrite(save_img_vis_name, I)
        cv2.imwrite(save_dpth_vis_name, raw_depth_colored)
    if cv2.waitKeyEx() == 27:
        cv2.destroyAllWindows()

def pil_frombyte(w, h):
    f = open(raw_file, 'rb')
    I = Image.frombytes("I;16", (w, h), f.read())
    f.close()
    I.show()

    file_name = os.path.splitext(os.path.basename(raw_file))[0]
    # save_file = save_dir+"/"+file_name+".png"
    save_file = save_dir+"/"+file_name+".jpeg"
    I.save(save_file, format='')
    print('save to '+save_file)

def npy_to_color(npy_dir, save_dir):
    for files in os.listdir(npy_dir):
        if files.endswith(".npy"):
            # print(os.path.join(npy_dir, files))
            depth_mat = read_depth_npy(os.path.join(npy_dir, files))
            depth_mat = depth_mat.astype(np.uint16)
            depth_color = vis_depth_mat(depth_mat, plt_show=False)
            save_file = save_dir+"/"+files.split('.')[0]+'_depth_color.png'
            cv2.imwrite(save_file, depth_color)
            print(save_file)

def readjust_coordinates(vertex_p, vertical_line=False):
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

# npy_mat_rc: matrix with shape of (rol_num, col_num), points_list_cr: four points,e.g. [[col_no, row_no],,,]
def depth_from_mat(npy_mat_rc, points_list_cr):
    assert len(points_list_cr) == 4
    tl, bl, br, tr = points_list_cr
    region_mat = npy_mat_rc[tl[1]:br[1], bl[0]:tr[0]]
    return region_mat
    
def region_depth_error(depth_npy1, depth_npy2, json_file=None, points_numpy=None):
    mat1 = read_depth_npy(depth_npy1)
    mat2 = read_depth_npy(depth_npy2)
    
    if json_file is not None:
        poly_shapes = read_json_label(json_file, key='shapes')
        test_regions = []
        for i, ann in enumerate(poly_shapes):
            label_list = ann['label'].split('-')
            if label_list[0] == 'gtest':
                p = readjust_coordinates(np.array(ann['points']), vertical_line=False)
                m1 = depth_from_mat(mat1, p)        
                m2 = depth_from_mat(mat2, p)        
                print(i, 'm1', m1)
                print(i, 'm2', m2)
                s1, s2 = m1.shape
                area = s1*s2
                mse_error = np.abs(m1-m2).mean()
                # avg_error = np.sqrt(np.sum((m1-m2)**2)/area)
                print(i, 'mse_error', mse_error)

    if points_numpy is not None:
        pd = read_depth_npy(points_numpy, allow_pickle=True).item()
        points = pd['points']
        depths = pd['depths']

        def find_dp(p, mat):
           pr = [np.round(p[0]), np.round(p[1])]
        #    mat1

        #point, depths:{'left':[], 'bottom':[], 'right':[], 'top':[]}
        for p, d in zip(points, depths):
             p['left']


    


if __name__ == '__main__':
    color_file = '/home/ly/data/datasets/trans-depth/navigation/20210915_154303-416/20210915_161422-416-1/png/_Color_1631693667692.64135742187500.png'
    depth_raw_file = '/home/ly/data/datasets/trans-depth/navigation/20210915_154303-416/20210915_161422-416-1/raw/_Depth_1631693667625.77148437500000.raw'
    npy_file = '/home/ly/data/datasets/trans-depth/navigation/20210915_154303-416/20210915_161422-416-1/compl_npy/_Depth_1631693667625_anno-2.npy'
    sides_json = '/home/ly/data/datasets/trans-depth/navigation/20210915_154303-416/20210915_161422-416-1/json/labels_transdepth_2021-09-24-05-36-07.json'

    color_file= '/home/ly/data/datasets/trans-depth/navigation/20210915_154303-416/aligned/2021_09_25_12_11_53/color/2.png'
    npy_file = '/home/ly/data/datasets/trans-depth/navigation/20210915_154303-vis_completed_depthpth_color/20210927_171255-angle1-fullcover-maxvis'
    # npy_to_color(npy_dir, npy_color_save)

    # depth_npy1 = '/home/ly/data/datasets/trans-depth/20210927/aligned_depth_color/20210927_171255-angle1-fullcover_1.npy'
    # depth_npy2 = '/home/ly/data/datasets/trans-depth/20210927/aligned_depth_color/completed_depth_npy/20210927_171255-angle1-fullcover_1_filled.npy'
    # test_json = '/home/ly/data/datasets/trans-depth/20210927/aligned_depth_color/20210927_171255-angle1-fullcover_1-gtest2.json'
    # points_numpy = '/home/ly/data/datasets/trans-depth/20210927/aligned_depth_color/completed_depth_npy/20210927_171255-angle1-fullcover_1_sides_depth.npy'

    depth_npy1 = '/home/ly/data/datasets/trans-depth/20210915_154303-416/aligned_depth_color/20210915_163731-416-mask_1.npy'
    depth_npy2 = '/home/ly/data/datasets/trans-depth/20210915_154303-416/aligned_depth_color/completed_depth_npy/20210915_161422-416-1_1_filled.npy'
    test_json = '/home/ly/data/datasets/trans-depth/20210915_154303-416/aligned_depth_color/20210915_163731-416-mask_1-gtest1.json'

    depth_npy1 = '/home/ly/data/datasets/trans-depth/temp_test/aligned_depth_color/20211006_105012-439test1_0.npy'
    depth_npy2 = '/home/ly/data/datasets/trans-depth/temp_test/aligned_depth_color/completed_depth_npy/20211006_105012-439test1_0_filled.npy'
    test_json = '/home/ly/data/datasets/trans-depth/temp_test/aligned_depth_color/20211006_105012-439test1_0.json'

    region_depth_error(depth_npy1, depth_npy2, json_file=test_json)