import enum
import os
import random
from turtle import color
import cv2
from cv2 import IMREAD_UNCHANGED
from matplotlib.pyplot import axis
import json
from numpy import int16, save, size
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def find_diff(image1_origin, image2_refrmove):
    im1_org = cv2.imread(image1_origin, IMREAD_UNCHANGED)
    im2_rr = cv2.imread(image2_refrmove, IMREAD_UNCHANGED)
    im_diff = np.abs(im1_org.astype(np.int16) - im2_rr.astype(np.int16)).astype(np.uint8)
    diff_color = cv2.applyColorMap(im_diff, cv2.COLORMAP_HSV)

    # save_heat_dir = '/home/ly/workspace/git/det/irr/Location-aware-SIRR/results/rattn_heatmap'
    save_rdiff_dir = '/home/ly/workspace/git/det/irr/Location-aware-SIRR/results/rattn'
    save_heat_dir = '/home/ly/workspace/git/det/irr/Location-aware-SIRR/results/rr_results'
    # save_rdiff_dir = '/home/ly/workspace/git/det/irr/Location-aware-SIRR/results/rr_results'
    im_org_name = os.path.basename(image1_origin)
    im_name= im_org_name.split('.')[0]
    #save reflection region highlight matrix
    save_npy_name = os.path.join(save_rdiff_dir, im_name)
    np.save(save_npy_name, im_diff)

    # im_remain = im_diff * image1_origin
    cv2.imshow('im_diff', diff_color)

    save_name = im_name + '_2highlight.png'
    save_name = os.path.join(save_heat_dir, save_name)
    cv2.imwrite(save_name, diff_color)

    #save reflection remove image
    save_rr_name = im_name + '_1remove.png'
    rr_save_name = os.path.join(save_heat_dir, save_rr_name)
    cv2.imwrite(rr_save_name, im2_rr)

    #save origin image
    im_save_name = os.path.join(save_heat_dir, im_org_name)
    cv2.imwrite(im_save_name, im1_org)


    if cv2.waitKeyEx() == 27:
        cv2.destroyAllWindows()

def main_find_diff():
    imnames = ['20210927_164622-angle1-origin_0', '20211019_113128_0', '20211019_132912_1', '20211019_141428_0', '20211016_165043_0', '20211016_164932_0']
    for inm in imnames:
        im_rr = '/home/ly/data/datasets/trans-depth/glass-rgbd-sirr/{}_fake_Ts_03.png'.format(inm)
        im_org= '/home/ly/data/datasets/trans-depth/glass-rgbd-sirr/{}_I_00.png'.format(inm)
        find_diff(im_org, im_rr)

def gen_reflection_map():
    org_img = '/home/ly/workspace/git/det/irr/Location-aware-SIRR/results/rattn_imgs/20211019_143938_0.png'
    rr_diff_file = '/home/ly/workspace/git/det/irr/Location-aware-SIRR/results/rattn/20211019_143938_0_I_00.npy'
    rrd_map = np.load(rr_diff_file)
    print(rrd_map.dtype)
    rm = np.max(rrd_map)
    dmap_copy = rrd_map.copy()
    dmap_copy[dmap_copy <= 7] = 0
    ref_mean_diff = dmap_copy.sum() / (dmap_copy > 0).sum()
    
    rrd_uni = np.unique(rrd_map, return_counts=True)
    print("np.unique(rrd_map)", rrd_uni)
    print("np.max(rrd_map)", rm)
    print("np.mean(rrd_map)", np.mean(rrd_map))
    print("ref_mean_diff", ref_mean_diff)

def gen_diff_map(image_dir, rr_dir, save_dir):
    im_names = os.listdir(image_dir)
    # to_tensor = transforms.ToTensor()
    #20210927_164622-angle1-origin_0_fake_Ts_03.png
    for im in im_names:
        org_image = image_dir+'/'+im
        iname = im.split('.')[0]
        rr_image = rr_dir+'/'+iname+'_fake_Ts_03.png'
        im_org = cv2.imread(org_image, IMREAD_UNCHANGED)
        im_rr = cv2.imread(rr_image, IMREAD_UNCHANGED)
        im_rr = torch.from_numpy(im_rr[np.newaxis].astype(np.float32))
        im_rr = im_rr.permute(0, 3, 1, 2)
        im_rr = torch.nn.functional.interpolate(im_rr, size=(im_org.shape[0], im_org.shape[1]), mode='bilinear')
        im_rr = im_rr.permute(0, 2, 3, 1)
        im_rr = np.array(im_rr[0])

        # im_rr_pil = Image.open(rr_image).convert('RGB')
        # new_h,  new_w = im_org.shape[:2]
        # resize_operation = transforms.Resize([new_h, new_w])
        # im_rr_pil = resize_operation(im_rr_pil)
        # im_rr_tensor = to_tensor(im_rr_pil) * 255
        # im_rr_np = np.array(im_rr_tensor.permute(1,2,0)) 
        # im_rr_np = im_rr_np.astype(np.uint8)
        # im_diff_np = np.abs(im_org.astype(np.int16) - im_rr_np.astype(np.int16)).astype(np.uint8)
        # print(np.unique(im_diff_np, return_counts=True))
        im_diff = im_org - im_rr

        rhint = np.abs(im_diff)
        rhint = rhint * (255 / np.max(rhint)) # norm to 0-255
        rhint = rhint.astype(np.uint8)
        print('max', np.max(rhint), 'min', np.min(rhint), 'avg', np.mean(rhint))
        save_file = save_dir+'/'+iname
        # print(save_file+'.npy')
        # np.save(save_file, im_diff)
        # print(save_file+'.png')
        cv2.imwrite(save_file+'.png', rhint)

def random_choice_within(choice_within, choice_num):
    return [random.choice(choice_within) for _ in range(choice_num)]

def imshow(im):
    plt.close()
    sizes = im.shape
    height = float(sizes[0])
    width = float(sizes[1])
    plt.imshow(im)
    print(plt.gcf().get_size_inches())
    exit()

    fig = plt.figure()
    fig.set_size_inches(width, height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, sizes[1] - 0.5])
    plt.ylim([sizes[0] - 0.5, -0.5])
    plt.imshow(im)

def reflection_sample(rgb_file, rh_file, hint_center_num=20, sample_max_ratio=1/3, save_vis_file=None):
    # rhint_dir = '/home/ly/data/datasets/glass-rgbd-rhint'
    # image_dir = '/home/ly/data/datasets/trans-depth/Glass-RGBD/images'
    # rh_fullglass_names = ['20211019_133156_1','20211021_154259_0','20211031_214510_0']
    # rh_wallglass_names = ['20211027_211312_0', '20210928_172025-angle6-origin_0', '20211002_153349-angle6_0', 
    #     '20211016_163026_0', '20211016_165043_0']
    # rh_name = '20211031_214510_0'
    # rh_file = rhint_dir+'/'+rh_name+'.png'
    # rgb_file = image_dir+'/'+rh_name+'.png'

    rhint = cv2.imread(rh_file)
    rhint = np.abs(rhint)
    rh_max = np.max(rhint)
    sample_coords = np.where(rhint > (rh_max * sample_max_ratio))
    x_list, y_list, c_list =sample_coords
    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    xy_arr = np.stack([x_arr, y_arr]).transpose(1, 0)
    xy_uni = np.unique(xy_arr, axis=0)
    km = KMeans(n_clusters=min(xy_uni.shape[0], hint_center_num), random_state=0).fit(xy_uni)
    hint_centers = km.cluster_centers_ #(n,2)[[row_number, column_number],...,]
    # tri = Delaunay(hint_centers)
    # tri_groups = tri.simplices #(n,3)[three points index from hint_centers]

    # print('xy_uni', xy_uni.shape, 'hint_centers', hint_centers.shape)
    # tri_idx = random_choice_within(range(tri_groups.shape[0]), choice_num=50)
    # tri_choosen = tri_groups[tri_idx]
    # tri_choosen = tri_groups
    im = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.axis('off')
    # plt.triplot(hint_centers[:, 1], hint_centers[:, 0], tri_choosen, zorder=1)
    plt.scatter(hint_centers[:, 1], hint_centers[:, 0], s=15, c='r', zorder=2)
    # plt.show()
    sz_inch = plt.gcf().get_size_inches()
    if save_vis_file is not None:
        plt.savefig(save_vis_file, dpi=im.shape[1]/sz_inch[0])
    plt.close()

    # for i, (x, y) in enumerate(xy_uni):
    #     cv2.circle(im, (y, x), 3, (0,0,255), -1)
    # hint_centers = hint_centers.astype(np.int64)
    # for i, (x, y) in enumerate(hint_centers):
    #     cv2.circle(im, (y, x), 3, (0,255,0), -1)
    # cv2.imshow('rhint', im)
    # if cv2.waitKeyEx() == 27:
    #     cv2.destroyAllWindows()
    save_hints = {'rhint_points':hint_centers.tolist()}
    return save_hints

def gen_reflection_points(image_dir, rhint_dir, save_dir, vis_save_dir):
    img_files = os.listdir(image_dir)
    for i, imf in enumerate(img_files):
        name = imf.split('.')[0]
        img_path = image_dir+'/'+imf
        hint_path = rhint_dir+'/'+name+'.png'
        save_vis_path = vis_save_dir+'/'+name+'.png'
        rhint_json = reflection_sample(img_path, hint_path, save_vis_file=save_vis_path)
        save_json_path = save_dir+'/'+name+'.json'
        with open(save_json_path, 'w') as scf:
            json.dump(rhint_json, scf)
        print('{}/{}'.format(i, len(img_files)), save_json_path)


if __name__ == '__main__':
    image_dir = '/home/ly/data/datasets/trans-depth/Glass-RGBD/images/'
    rhint_dir = '/home/ly/data/datasets/glass-rgbd-rhint'
    rpoints_save_dir = '/home/ly/data/datasets/glass-rgbd-rpoints-1_3'
    rpoints_vis_save_dir = '/home/ly/data/datasets/glass-rgbd-rpoints-vis-1_3'

    # rr_dir = '/home/ly/data/datasets/glass-rgbd-sirr'
    # rdiff_save_dir = '/home/ly/data/datasets/glass-rgbd-rhint'
    # gen_diff_map(image_dir, rr_dir, rdiff_save_dir)
    os.makedirs(rpoints_save_dir, exist_ok=False)
    os.makedirs(rpoints_vis_save_dir, exist_ok=False)
    # reflection_sample(1,2)
    # gen_reflection_points(image_dir, rhint_dir, rpoints_save_dir, rpoints_vis_save_dir)
