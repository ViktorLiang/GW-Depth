import enum
import math
import os
import cv2
import numpy as np
import torch
from commons import draw_points, draw_polys, read_depth_npy
from check_in_polygon import draw_lines_from_points
import matplotlib

from read_binfile import depth_from_mat
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import colors as mcolors
import colorsys

from depth_pred_comparison import resize_by_torch


def sample_points(ratio_sample=False):
    sample_num_seg = 4
    num_line = 3
    H, W = 512, 1024
    random_lines = np.random.rand(num_line, 2, 2)
    random_lines[:, 0, 0] *= W
    random_lines[:, 0, 1] *= H
    random_lines[:, 1, 0] *= W
    random_lines[:, 1, 1] *= H
    img = np.ones((H, W, 3)) * 255
    print(random_lines)
    random_lines = torch.from_numpy(random_lines)
    if ratio_sample:
        max_dist = math.sqrt(H ** 2 + W ** 2)
        line_length = torch.sqrt(torch.sum((random_lines[:, 0] - random_lines[:, 1]) ** 2, dim=-1))
        sample_num = []
        for i, len in enumerate(line_length):
            sample_num.append((len / max_dist) * sample_num_seg)

    st = torch.argmin(random_lines[:, :, 0], axis=1)
    end = torch.argmax(random_lines[:, :, 0], axis=1)
    st_ids = st[:, None, None].expand(-1, -1, 2)
    end_ids = end[:, None, None].expand(-1, -1, 2)
    st_points = torch.gather(random_lines, 1, index=st_ids).squeeze(1)
    end_points = torch.gather(random_lines, 1, index=end_ids).squeeze(1)

    dist = np.sqrt((st_points[:, 0] - end_points[:, 0]) ** 2 + (st_points[:, 1] - end_points[:, 1]) ** 2)
    x_dist =  np.sqrt((st_points[:, 0] - end_points[:, 0]) ** 2)
    y_dist =  np.sqrt((st_points[:, 1] - end_points[:, 1]) ** 2)
    cosin = x_dist / dist
    sin = y_dist / dist
    dist_seg = dist / sample_num_seg
    seg_x = dist_seg * cosin
    seg_y = dist_seg * sin
    new_points_x = []
    is_row_ascent = st_points[:, 1] < end_points[:, 1]
    row_oper = is_row_ascent.type(torch.int8)
    row_oper = row_oper * 2 -1

    for seg_i in range(1, sample_num_seg):
        p_x = st_points[:, 0] + seg_x * seg_i
        p_y = st_points[:, 1] + seg_y * seg_i * row_oper
        seg_p = torch.cat([p_x[:, None], p_y[:, None]], dim=1)
        new_points_x.append(seg_p)
    
   
    all_seg_points = torch.cat(new_points_x, dim=0)
    draw_points(random_lines.reshape(-1, 2), mat=img, show=True)
    draw_polys(random_lines, mat=img, show=True)
    draw_points(all_seg_points, mat=img, color=(0, 256, 128), show=True)
    print('is_row_ascent', is_row_ascent)
    print(new_points_x)
    exit()

    cv2.imshow('img', img)

    # for i, seg in enumerate(random_lines):
    #     print(seg)
    #     st = seg[0]
    #     end = seg[1]
    #     # hori_lin_st = min(st[0], end[0])
    #     # hori_lin_end = max(st[0], end[0])

    #     # hori_coors = np.linspace(hori_lin_st, hori_lin_end, num=sample_num_perseg, endpoint=False)
    #     # hori_rela_dist = hori_coors - st[0]
    #     # samp_hori_coors = hori_coors[1:]
    #     # tanAngle = (st[1] - end[1]) / (st[0] - end[0])
    #     # vert_rela_dist = hori_rela_dist * tanAngle
    #     # vert_coors = vert_rela_dist + st[1]
    


    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()

def vis_depth_mat(dpth_mat, height=None, width=None, show=False, raw_depth_file=None, save_file=None):
    # min, max, min_loc, max_loc = cv2.minMaxLoc(dpth_mat)
    if height is None or width is None:
        height, width = dpth_mat.shape[-2:]

    # dpth_color = cv2.applyColorMap(
    #         cv2.convertScaleAbs(dpth_mat, alpha=0.03), cv2.COLORMAP_JET)
    # plt.imsave(depth_error_png, pred_error_emap, vmin=0.0, vmax=e_max, cmap='Reds')
    # plt.imshow(dpth_mat, vmin=0, vmax=np.max(dpth_mat), cmap='jet')
    plt.imshow(dpth_mat, vmin=0, vmax=np.max(dpth_mat), cmap='jet')
    plt.show()

    

    # if show:
    #     # plt.title('completed_depth')
    #     # plt.imshow(dpth_color)
    #     # if save_file is not None:
    #     #     save_new_name = save_file.split('.')[0] + '.png'
    #     #     plt.savefig(save_new_name)
    #     #     print('saved to '+save_new_name)
    #     # plt.show()

    #     cv2.imshow("dpth_arr", dpth_color)
    #     if cv2.waitKeyEx() == 27:
    #         cv2.destroyAllWindows()
    # else:
    #     return dpth_color

def error_vis_test():
    # e_max = 0.5
    e_max = 2
    

    pred_dir = '/home/ly/workspace/git/segm/line_segm/ablation/reliable_points/PGT_RPT/exp/PGT_RPT_win9-11-11/eval_log/dense_pred/checkpoint0135'
    depth_pred = pred_dir+'/20211111_213536_0.npy'
    pred_mat = read_depth_npy(depth_pred)
    # vis_depth_mat(pred_mat, show=True)

    depth_png = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/depth/20211111_213536_0.png'
    depth_mat = cv2.imread(depth_png, -1)
    # vis_depth_mat(depth_mat, show=True)
    h, w = pred_mat.shape[-2:]
    depth_mat = resize_by_torch(depth_mat.astype(np.float32), h, w)
    depth_mat = np.array(depth_mat) / 1000.0
    # vis_depth_mat(depth_mat, show=True)
    print('depth_mat', depth_mat.shape, 'pred_mat', pred_mat.shape)
    # pred emap
    pred_error_emap = np.abs(pred_mat - depth_mat)
    pred_error_emap[depth_mat < 0.001] = 0.0 
    pred_error_emap[depth_mat > 10] = 0.0 
    # target_path = '%s/%08d_pred_emap_iter%02d.jpg' % (args.exp_vis_dir, total_iter, i)
    print('np.unique(pred_mat)', np.unique(pred_mat))
    print('np.unique(depth_mat)', np.unique(depth_mat))
    print(np.unique(pred_error_emap))
    depth_error_png = './saved_vis/20211111_213536_0-error-max2.png'
    plt.imsave(depth_error_png, pred_error_emap, vmin=0.0, vmax=e_max, cmap='Reds')
    plt.show(pred_error_emap)



__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
def unnormalize(img_in):
    img_out = np.zeros(img_in.shape)
    for ich in range(3):
        img_out[:, :, ich] = img_in[:, :, ich] * __imagenet_stats['std'][ich]
        img_out[:, :, ich] += __imagenet_stats['mean'][ich]
    img_out = (img_out * 255).astype(np.uint8)
    return img_out

def vis_test(pred_npy, gt_png):
    pred_dir = '/home/ly/workspace/git/segm/line_segm/ablation/reliable_points/PGT_RPT/exp/PGT_RPT_win9-11-11/eval_log/dense_pred/checkpoint0135'
    depth_png = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/depth/20211111_213536_0.png'

    depth_pred = pred_dir+'/20211111_213536_0.npy'
    pred_mat = read_depth_npy(depth_pred) * 1000
    vis_depth_mat(pred_mat, show=True)

    depth_mat = cv2.imread(depth_png, -1)
    h, w = pred_mat.shape[-2:]
    depth_mat = resize_by_torch(depth_mat.astype(np.float32), h, w)
    gt_mat = np.array(depth_mat)
    vis_depth_mat(gt_mat, show=True)
    print('depth_mat', depth_mat.shape, 'pred_mat', pred_mat.shape)

def save_vis_mat(save_file, depth_mat, vmax=5000):
    # print('save_file', save_file)
    # print('depth_mat', depth_mat.shape)
    # print('vmax', np.max(depth_mat))
    # exit()
    if vmax > 0:
        plt.imsave(save_file, depth_mat, vmin=0.0, vmax=vmax, cmap='jet')
    else:
        plt.imsave(save_file, depth_mat, vmin=0.0, vmax=np.max(depth_mat), cmap='jet')

    # plt.imshow(dpth_mat, vmin=0, vmax=np.max(dpth_mat), cmap='jet')
    # plt.imshow(depth_mat, vmin=0, vmax=3500, cmap='hsv')

def example_depth_vis():
    ours_dense = '/home/ly/workspace/git/segm/line_segm/ablation/reliable_points/PGT_RPT/exp/PGT_RPT_win9-11-11/eval_log/dense_pred/checkpoint0135'
    p3dpth_dense = '/home/ly/workspace/git/segm/line_segm/ablation/other_mono_depth/P3Depth/output/StructRankNetPQRS_resnet101_15_08_2022-175456_exp1_GLASSRGBD/checkpoints/eval/pred_depth'
    newcrfs_dense = '/home/ly/workspace/git/segm/line_segm/ablation/other_mono_depth/NeWCRFs/output/model-9-best_d1_0.90031/'
    pred_dirs = [ours_dense, newcrfs_dense, p3dpth_dense, ]
    save_root = '/home/ly/Documents/papers/my_pappers/glass-depth-ieee/IEEEtran/figures/experiments/otherworks/'
    cmap = 'jet-test'
    save_folders = ['{}/our', '{}/newcrfs', '{}/p3depth']

    for i, sf in enumerate(save_folders):
        rsf = save_root + '/' + sf.format(cmap)
        save_folders[i] = sf.format(cmap)
        os.makedirs(rsf, exist_ok=True)


    gt_dir = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/depth/'
    pred_imgs = ['20211018_165254_0', '20211111_212137_0', '20211111_213536_0', '20220514_102529_0']

    vmax = 0
    for im in pred_imgs:
        gt_png = os.path.join(gt_dir, im)
        our_dir = pred_dirs[0]
        our_depth = our_dir+'/{}.npy'.format(im)
        our_pred_mat = read_depth_npy(our_depth) * 1000

        save_file = save_root+'/{}/{}.png'.format(save_folders[0], im)
        h, w = our_pred_mat.shape[-2:]

        gt_depth = gt_dir+'/{}.png'.format(im)
        gt_mat = cv2.imread(gt_depth, -1)
        save_gt_file = save_root+'/{}/{}.png'.format(cmap, im)
        gt_mat = gt_mat.astype(np.float32)
        gt_mat = resize_by_torch(gt_mat, h, w)
        gt_mat = np.array(gt_mat)
        gtmax = np.max(gt_mat)
        vmax = gtmax * 1.1
        save_vis_mat(save_gt_file, gt_mat, vmax=vmax)
        save_vis_mat(save_file, our_pred_mat, vmax=vmax)
        # print(im, vmax, np.min(gt_mat))

        for pred_dir, save_folder in zip(pred_dirs[1:], save_folders[1:]):
            depth_npy = pred_dir+'/{}.npy'.format(im)
            save_file = save_root+'/{}/{}.png'.format(save_folder, im)
            print('depth_npy', depth_npy)
            print('save_file', save_file)
            pred_mat = read_depth_npy(depth_npy) * 1000
            pred_mat = resize_by_torch(pred_mat, h, w)
            pred_mat = np.array(pred_mat)
            save_vis_mat(save_file, pred_mat, vmax=vmax)
            print()

def draw_color_bar():
    from matplotlib.colors import LinearSegmentedColormap

    def grayscale_cmap(cmap):
        """Return a grayscale version of the given colormap"""
        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))
        
        # convert RGBA to perceived grayscale luminance
        # cf. http://alienryderflex.com/hsp.html
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]
            
        return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
    

    def view_colormap(cmap):
        """Plot a colormap with its grayscale equivalent"""
        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))

        # cmap = grayscale_cmap(cmap)
        # grayscale = cmap(np.arange(cmap.N))
        print('colors', colors.shape)

        fig, ax = plt.subplots(subplot_kw=dict(xticks=[], yticks=[]))
        print('colors', colors.shape)
        ax.imshow([colors], extent=[0, 10, 0, 1])
        # ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
    
    view_colormap('jet')
    plt.show()

def show_color_bar():
    d_maxd = {'20211018_165254_0': [2383., 5401.0], 
                '20211111_212137_0': [1288., 2851.2000000000003],
                '20211111_213536_0': [1850., 3713.6000000000004], 
                '20220514_102529_0': [2680., 7123.6]}
    
    for im, dv in d_maxd.items():
        dmin, dmax = dv
        n1 = np.arange(dmin, dmax)
        n1 = n1[np.newaxis]
        dpth_mat = np.repeat(n1, 500, axis=0)
        save_file = '/home/ly/Documents/papers/my_pappers/glass-depth-ieee/IEEEtran/figures/experiments/otherworks/range_bars/{}-bar-flip.png'.format(im)
        dpth_mat = np.flip(dpth_mat)
        dpth_mat = dpth_mat.transpose()
        plt.imsave(save_file, dpth_mat, vmin=0.0, vmax=dmax, cmap='jet')
        print(save_file)
        # plt.imshow(dpth_mat, vmin=0.0, vmax=dmax, cmap='jet')
        # plt.show()

if __name__ == '__main__':
    # sample_points(ratio_sample=True)
    # vis_test()
    # example_depth_vis()
    # draw_color_bar()
    show_color_bar()