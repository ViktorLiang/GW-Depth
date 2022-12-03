import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage
import cv2
import torch
import torchvision.transforms.functional as F

def vis_depth_pred(depth_pred):
    dpth_mat = depth_pred * 1000
    # alpha = 0.03 #[0-8.5meters]
    alpha = 0.0255 #[0-10meters]
    dpth_color = cv2.applyColorMap(
            cv2.convertScaleAbs(dpth_mat, alpha=alpha), cv2.COLORMAP_JET)
    return dpth_color

def compute_rmse(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    return rmse

def resize_by_torch(npy_depth, h, w):
    ncf_d_t = torch.tensor(npy_depth, dtype=torch.float32)
    ncf_d_t = torch.nn.functional.interpolate(ncf_d_t[None, None], size=[h, w], mode='nearest')
    ncf_d_t = ncf_d_t[0][0]
    return ncf_d_t


def rms_test():
    f1 = '/home/ly/workspace/git/segm/line_segm/ablation/other_mono_depth/P3Depth/output/StructRankNetPQRS_resnet101_15_08_2022-175456_exp1_GLASSRGBD/checkpoints/eval/pred/20211111_211515_0.png'
    f2 = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/depth/20211111_211515_0.png'
    pred = cv2.imread(f1, cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(f2, cv2.IMREAD_UNCHANGED)
    h, w = pred.shape[-2], pred.shape[-1]
    gt = resize_by_torch(gt.astype(np.float32), h, w)
    print(np.unique(gt/1000))
    print(np.unique(pred))
    print(compute_rmse(gt/1000, pred))


def all_works_depth():
    ours_dense = '/home/ly/workspace/git/segm/line_segm/ablation/reliable_points/PGT_RPT/exp/PGT_RPT_win9-11-11/eval_log/dense_pred/checkpoint0135'
    p3dpth_dense = '/home/ly/workspace/git/segm/line_segm/ablation/other_mono_depth/P3Depth/output/StructRankNetPQRS_resnet101_15_08_2022-175456_exp1_GLASSRGBD/checkpoints/eval/pred_depth'
    newcrfs_dense = '/home/ly/workspace/git/segm/line_segm/ablation/other_mono_depth/NeWCRFs/output/model-9-best_d1_0.90031/'
    
    # newcrfs_dense = '/home/ly/workspace/git/segm/line_segm/ablation/other_mono_depth/NeWCRFs/output/model_nyu.ckpt/'
    gt_depth_dir = '/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/depth'

    # save_dir = '/home/ly/workspace/git/segm/line_segm/ablation/other_mono_depth/all_works_depth'
    save_dir = '/home/ly/workspace/git/segm/line_segm/ablation/other_mono_depth/all_works_depth_error'
    save_dir_depth = ['ours', 'p3depth', 'newcrf', 'gt']
    works_depths_dir = []
    for sd in save_dir_depth:
        work_dir = os.path.join(save_dir, sd)
        os.makedirs(work_dir, exist_ok=True)
        works_depths_dir.append(work_dir)
        
    our_save_dir = works_depths_dir[0]
    p3d_save_dir = works_depths_dir[1]
    ncrf_save_dir = works_depths_dir[2]
    gt_save_dir = works_depths_dir[3]
    # e_max = 1.0
    # e_max = 2.0
    e_max = 3.0
    cmaps = ['viridis', 'inferno', 'magma', 'cividis', 'Purples', 'Greens', 'gist_yarg', 'PiYG', 'coolwarm', 'twilight', 'twilight_shifted', 'hsv', 'Pastel1', 'Paired', 'Accent',
    'Dark2', 'Set1', 'tab10', 'jet']
    color_map = cmaps[-1]

    depth_error_dir_our = our_save_dir+'/error-em{}-{}/'.format(e_max, color_map)
    depth_error_dir_p3depth = p3d_save_dir+'/error-em{}-{}/'.format(e_max, color_map)
    depth_error_dir_newcrf = ncrf_save_dir+'/error-em{}-{}/'.format(e_max, color_map)
    os.makedirs(depth_error_dir_our, exist_ok=True)
    os.makedirs(depth_error_dir_p3depth, exist_ok=True)
    os.makedirs(depth_error_dir_newcrf, exist_ok=True)

    ours_npy = glob.glob(ours_dense+"/*.npy")
    for our_npy in ours_npy:
        npy_name = os.path.basename(our_npy)
        file_name = npy_name.split('.')[0]
        our_depth = np.load(our_npy)
        # gt_depth = Image.open(ours_dense+'/'+file_name+'_gt-depth.png')
        gt_depth = cv2.imread(gt_depth_dir+'/'+file_name+'.png', cv2.IMREAD_UNCHANGED)
        gt_depth = np.array(gt_depth, dtype=np.float32)

        p3_npy = p3dpth_dense+'/'+npy_name
        p3_depth = np.load(p3_npy) 
        ncrf_npy = newcrfs_dense+'/'+npy_name
        ncrf_depth = np.load(ncrf_npy)

        h, w = our_depth.shape
        
        gt_depth = resize_by_torch(gt_depth, h, w)
        p3_depth = resize_by_torch(p3_depth, h, w)
        ncrf_depth = resize_by_torch(ncrf_depth, h, w)

        gt_depth = np.array(gt_depth)
        p3_depth = np.array(p3_depth)
        ncrf_depth = np.array(ncrf_depth)
        gt_depth_me = gt_depth/1000
        rmse_ncrf = compute_rmse(gt_depth/1000, ncrf_depth)
        rmse_p3d = compute_rmse(gt_depth/1000, p3_depth)
        rmse_our = compute_rmse(gt_depth/1000, our_depth)
        rmse_log = "{}, ncrf:{:2.3f}, p3d:{:2.3f}, our:{:2.3f}\n".format(file_name, rmse_ncrf, rmse_p3d, rmse_our)
        with open(save_dir+'/rmse_log.txt', 'a+') as f:
            f.write(rmse_log)

        gt_depth_vis = vis_depth_pred(gt_depth_me)
        our_depth_vis = vis_depth_pred(our_depth)
        p3_depth_vis = vis_depth_pred(p3_depth)
        ncrf_depth_vis = vis_depth_pred(ncrf_depth)
        cv2.imwrite(our_save_dir+'/'+file_name+'.png', our_depth_vis,)
        cv2.imwrite(gt_save_dir+'/'+file_name+'.png', gt_depth_vis)
        cv2.imwrite(p3d_save_dir+'/'+file_name+'.png', p3_depth_vis)
        cv2.imwrite(ncrf_save_dir+'/'+file_name+'.png', ncrf_depth_vis)

        # our depth error
        def save_depth_error(pred_depth_meter, gt_depth_meter, save_file):
            error_emap = np.abs(pred_depth_meter - gt_depth_meter)
            error_emap[gt_depth_me < 0.001] = 0.0 
            error_emap[gt_depth_me > 10] = 0.0 
            plt.imsave(save_file, error_emap, vmin=0.0, vmax=e_max, cmap=color_map)
        
        our_depth_error_png = depth_error_dir_our + '/' + file_name+'.png'
        p3_depth_error_png = depth_error_dir_p3depth + '/' + file_name+'.png'
        ncrf_depth_error_png = depth_error_dir_newcrf + '/' + file_name+'.png'
        save_depth_error(our_depth, gt_depth_me, our_depth_error_png)
        save_depth_error(p3_depth, gt_depth_me, p3_depth_error_png)
        save_depth_error(ncrf_depth, gt_depth_me, ncrf_depth_error_png)

        print(file_name)



if __name__ == '__main__':
    # all_works_depth()
    all_works_depth()