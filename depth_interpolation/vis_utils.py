import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from commons import read_depth_npy
from local_test import image_draw_circles

def get_top(variance):
        b = 0
        sample_inter_d_num = 30
        H, W = variance.shape

        val, indx = torch.topk(variance.flatten(0), sample_inter_d_num)
        indx, _ = indx.sort()
        row = torch.div(indx, W, rounding_mode='floor')
        col = indx % W
        coors = torch.stack([col, row])
       
        return coors

def show_points(img, coors):
        im = image_draw_circles(img, np.ceil(coors).astype(np.int64),)
        return im

def show_depth_pair():
        npyfolders = '/home/ly/workspace/git/segm/line_segm/ablation/reliable_points/PGT_RPT/exp/val187_LowResDepthDenorm/eval_log/dense_pred/checkpoint0132'
        npy_file1 = '20220821_153433_0_1.npy'
        npy_file2 = '20220821_153433_0_2.npy'
        imgname = '20220821_153433_0_rgb.png'
        save_dir = './vis_test/'
        imgfile = npyfolders+'/'+imgname
        save_file1 = save_dir+'/'+ os.path.basename(npy_file1).split('.')[0]+'-cv.png'
        save_file2 = save_dir+'/'+ os.path.basename(npy_file2).split('.')[0]+'-cv.png'
        save_var = save_dir+'/'+os.path.basename(npy_file2).split('.')[0]+'-var.png'

        npy1 = npyfolders+'/'+npy_file1
        npy2 = npyfolders+'/'+npy_file2
        depth1 = read_depth_npy(npy1)
        depth2 = read_depth_npy(npy2)
        H, W = depth1.shape
        print(np.unique(depth1), depth1.shape)
        print(np.unique(depth2), depth2.shape)
        vars = np.abs(depth1 - depth2)
        
        colormap = 'Greens'
        cat1 = np.concatenate([depth1, depth2])
        max_depth = np.max(cat1)*100
        d1 = depth1 * 100
        d2 = depth2 * 100
        
        t_vars = torch.from_numpy(vars)
        topcoors = get_top(t_vars)
        topcoors = np.array(topcoors).astype(np.uint16)
        topcoors = topcoors.reshape(-1, 2)
        im = cv2.imread(imgfile)
        im = cv2.resize(im, (W, H), interpolation = cv2.INTER_AREA)

        # im = show_points(im, topcoors)
        plt.imsave(save_file1, d1, vmin=0.0, vmax=max_depth, cmap=colormap)
        plt.imsave(save_file2, d2, vmin=0.0, vmax=max_depth, cmap=colormap)
        # plt.imsave(save_var, vars, vmin=0.0, vmax=np.max(vars), cmap='Greens')
        # plt.imsave(save_dir+'/'+imgname, im)

        # scale_value = 255 / np.max(depth1)
        # depth1_color = cv2.applyColorMap( cv2.convertScaleAbs(depth1 * scale_value, alpha=1), cv2.COLORMAP_MAGMA)
        # depth2_color = cv2.applyColorMap( cv2.convertScaleAbs(depth2 * scale_value, alpha=1), cv2.COLORMAP_MAGMA)

        # cv2.imwrite(save_file1, depth1_color)
        # cv2.imwrite(save_file2, depth2_color)



if __name__ == '__main__':
        show_depth_pair()