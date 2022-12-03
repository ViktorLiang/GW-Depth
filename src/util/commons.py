from ctypes.wintypes import PINT
from cv2 import imwrite, line
import torch
import torchvision
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

from evaluation.eval_post_online import imshow, pline, plambda
# colour map
COLORS = [(0,0,0)
        ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
        ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
        ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
        ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]

def read_json_label(json_label, key=None):
    with open(json_label) as f:
        annos = json.load(f)
        if key is not None:
            assert key in annos, 'key {} not exists in keys:{}'.format(key, str(annos.keys()))
            return annos[key]
        else:
            return annos

def gen_pairs(np_vector):
    d0 = np_vector[:, np.newaxis]
    d1 = np_vector[1:].tolist()
    d1.append(np_vector[0])
    d1 = np.array(d1)[:, np.newaxis]
    d_pairs = np.concatenate((d0, d1), axis=1)
    return d_pairs

def lines_reverse_to_image(image_width_height, lines_coors):
    w, h = image_width_height
    real_lines = lines_coors * np.array([[w, h, w, h]])
    # clamp
    real_lines = real_lines.astype(np.int)
    real_lines = np.where(real_lines >= 0, real_lines, 0)
    real_lines[:, :, 0::2] = np.where(real_lines[:, :, 0::2] > w - 1, w - 1, real_lines[:, :, 0::2])
    real_lines[:, :, 1::2] = np.where(real_lines[:, :, 1::2] > h - 1, h - 1, real_lines[:, :, 1::2])
    return real_lines

def image_draw_lines(image, lines, color=None):
    if color is None:
        color = (0, 0, 215)
    for l in lines:
        c1, r1 = l[0:2]
        c2, r2 = l[2:4]
        cv2.line(image, (c1, r1), (c2, r2), color=color, thickness=2)
    return image

def image_draw_circles(image, centers, color=None, radius=10):
    if color is None:
        color = (0, 0, 215)
    for c in centers:
        cv2.circle(image, c, radius=radius, color=color, thickness=4)
    return image

def draw_line_test():
    f = "/home/ly/data/datasets/trans-depth/20210929/aligned_depth_color/20210929_162104-angle2-origin_6.png"
    i = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    h,w,c = i.shape
    cv2.line(i, (0,0), (w*2, h*2), color=(0,255,0), thickness=3)
    cv2.imshow('gt lines', i)
    if cv2.waitKeyEx() == 27:
        cv2.destroyAllWindows()
    exit()

class NormalizeInverse(torchvision.transforms.Normalize):
    """ 
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

def inv_preprocess(imgs, num_images, toRGB=True):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    rev_imgs = imgs[:num_images].clone().cpu().data
    rev_normalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(num_images):
        rev_imgs[i] = rev_normalize(rev_imgs[i])

    if toRGB:
        rev_imgs = rev_imgs[:, [2,1,0]]
    return rev_imgs

def show_labels(image, targets, points=None, need_inv=True, tittle='image with line labels', save_dir=None, epoch=0, draw_color=None, mode='train'):
    if need_inv:
        img_mat = inv_preprocess(image.unsqueeze(0), num_images=1)
        im = np.array(img_mat.permute(0, 2, 3, 1))
        im = np.ascontiguousarray(im[0])
    else:
        im  = image
    tgt = targets
    h, w = targets['size']
    for k, v in tgt.items():
        if k == 'lines':
            v = np.array(v.cpu())
            if need_inv:
                # all_lines = v * np.array([w, h, w, h])
                # all_centers = tgt['poly_centers'] * np.array([w, h])
                all_points = v * np.array([w, h, w, h, w, h])
                all_lines = all_points[:, :4]
                all_centers = all_points[:, 4:]
            else:
                all_lines = v
                all_centers = tgt['poly_centers']
            all_pids = tgt['poly_ids']
            py_ids = torch.unique(all_pids)
            for pd in py_ids:
                choosen_idx = (all_pids == pd).tolist()
                c_lines = np.array(all_lines[choosen_idx])
                c_lines = c_lines.astype(np.uint16).tolist()
                cid = (pd+1) % len(COLORS)
                im = image_draw_lines(im, c_lines, color=COLORS[cid])
                c_centers = np.array(all_centers[choosen_idx]).astype(np.uint16)
                im = image_draw_circles(im, c_centers, color=COLORS[cid])
    if points is not None and len(points) > 0:
        assert points.shape[0] != 0, points
        points = np.array(points)
        if need_inv:
            points *= np.array([w, h])
        points = points.astype(np.uint16)
        im = image_draw_circles(im, points, color=COLORS[18] if draw_color is None else draw_color, radius=5)
            
    image_id = targets['image_id'].cpu().item()
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_png = "{}#ep{}#im{}#{}.png".format(mode, epoch, image_id, now)

    cv2.imwrite(save_dir+'/'+save_png, im)
    # imshow(im)
    # plt.savefig(save_dir+'/'+save_png, dpi=500, bbox_inches=0)
    # plt.close()
    # cv2.imshow(save_png, im)
    # if cv2.waitKeyEx() == 27:
    #     cv2.destroyAllWindows()

def save_input(samples, targets):
    img_mat = inv_preprocess(samples.tensors, num_images=1)
    img_mat = img_mat[0].permute(1, 2, 0)
    im = np.array(img_mat.cpu())
    im = np.ascontiguousarray(im)
    tgt = targets[0]
    h, w = tgt['size']
    for k, v in tgt.items():
        v = np.array(v.cpu())
        if k == 'lines' and 'poly_centers' in tgt:
            all_lines = v * np.array([w, h, w, h])
            all_centers = tgt['poly_centers'] * np.array([w, h])
            all_pids = tgt['poly_ids']
            py_ids = torch.unique(all_pids)
            for pd in py_ids:
                c_lines = np.array(all_lines[all_pids == pd])
                c_lines = c_lines.astype(np.uint16).tolist()
                im = image_draw_lines(im, c_lines, color=COLORS[pd+1])
                c_centers = np.array(all_centers[all_pids == pd]).astype(np.uint16)
                im = image_draw_circles(im, c_centers, color=COLORS[pd+1])
    cv2.imshow('im with  line', im)
    if cv2.waitKeyEx() == 27:
        cv2.destroyAllWindows()


def batch_isin_triangle(triangles, points):
    def area(p1x, p1y, p2x, p2y, p3x, p3y):
        return torch.abs((p1x * (p2y - p3y) + p2x * (p3y - p1y) + p3x * (p1y - p2y)) / 2.0)
    assert triangles.shape[0] == points.shape[0]
    assert triangles.dim() in [3, 4]
    B = triangles.shape[0]
    if triangles.dim() == 4:
        tri = triangles
    else:
        tri = triangles.reshape(B, -1, 3, 2)

    pnum = points.shape[1]
    tnum = tri.shape[1]
    tri_r = tri.repeat_interleave(pnum, dim=1)
    pnt_r = points[:, None].repeat_interleave(tnum, dim=1).reshape(B, -1, 2)

    tp1 = tri_r[:, :, 0]
    tp2 = tri_r[:, :, 1]
    tp3 = tri_r[:, :, 2]
    tp1_x, tp1_y = tp1[:, :, 0], tp1[:, :, 1]
    tp2_x, tp2_y = tp2[:, :, 0], tp2[:, :, 1]
    tp3_x, tp3_y = tp3[:, :, 0], tp3[:, :, 1]
    p_x, p_y = pnt_r[:, :, 0], pnt_r[:, :, 1]

    A = area(tp1_x, tp1_y, tp2_x, tp2_y, tp3_x, tp3_y)
    A1 = area(p_x, p_y, tp2_x, tp2_y, tp3_x, tp3_y)
    A2 = area(tp1_x, tp1_y, p_x, p_y, tp3_x, tp3_y)
    A3 = area(tp1_x, tp1_y, tp2_x, tp2_y, p_x, p_y)
    isin = (A == (A1+A2+A3))
    return isin

def sample_lines(lines, scores, height, width, base_threshold=0.01, tol=1e9, do_clip=False):
    lines[:, :, 0] *= height
    lines[:, :, 1] *= width
    diag = (height ** 2 + width ** 2) ** 0.5
    threshold = diag * base_threshold
    lines_glass = lines[:, :2]

    nlines, nscores, ncenters = [], [], []
    for id, ((p, q), score) in enumerate(zip(lines_glass, scores)):
        start, end = 0, 1
        for a, b in nlines:  # nlines: Selected lines.
            if (
                min(
                    max(pline(*p, *q, *a), pline(*p, *q, *b)),
                    max(pline(*a, *b, *p), pline(*a, *b, *q)),
                )
                > threshold ** 2
            ):
                continue
            lambda_a = plambda(*p, *q, *a)
            lambda_b = plambda(*p, *q, *b)
            if lambda_a > lambda_b:
                lambda_a, lambda_b = lambda_b, lambda_a
            lambda_a -= tol
            lambda_b += tol

            # case 1: skip (if not do_clip)
            if start < lambda_a and lambda_b < end:
                continue

            # not intersect
            if lambda_b < start or lambda_a > end:
                continue

            # cover
            if lambda_a <= start and end <= lambda_b:
                start = 10
                break

            # case 2 & 3:
            if lambda_a <= start and start <= lambda_b:
                start = lambda_b
            if lambda_a <= end and end <= lambda_b:
                end = lambda_a

            if start >= end:
                break

        if start >= end:
            continue
        nlines.append(np.array([p + (q - p) * start, p + (q - p) * end]))
        nscores.append(score)
        # choosen_ids.append(id)
        ncenters.append(lines[id, 2])
    # return torch.tensor(nlines, dtype=torch.float), torch.tensor(nscores, dtype=torch.float), torch.tensor(choosen_ids, dtype=torch.int16)
    return torch.tensor(nlines, dtype=torch.float), torch.tensor(nscores, dtype=torch.float), torch.tensor(ncenters, dtype=torch.int16)

def show_smapled_lines(line_centers, image_mats, is_normed=False, with_center=False, title='', stop_show=True):
    bs, c, h, w = image_mats.shape
    img_mat = inv_preprocess(image_mats, num_images=image_mats.shape[0])
    img_mat = img_mat[0].permute(1, 2, 0)
    im = np.array(img_mat.cpu())
    im = np.ascontiguousarray(im)
    line_centers = np.array(line_centers)
    # print('im', im.shape)
    # print('line_centers', line_centers)
    for i in range(bs):
        lcen = line_centers[i]
        lines = lcen[:, :2]
        if with_center:
            centers = lcen[:, 2]
        lines = lines.reshape(-1, 4)
        if not is_normed:
            lines *= np.array([w, h, w, h])
            if with_center:
                centers *= np.array([w, h])
        # print(i, lines, lines.shape, lines.dtype)
        im = image_draw_lines(im, np.ceil(lines).astype(np.int64))
        if with_center:
            im = image_draw_circles(im, np.ceil(centers).astype(np.int64), (123, 255, 5))
        cv2.imshow('show_smapled_lines:'+title, im)
        if stop_show:
            if cv2.waitKeyEx() == 27:
                cv2.destroyAllWindows()

def show_sampled_points(points, image_mats, height=None, width=None, is_normed=False):
    bs, c, h, w = image_mats.shape
    img_mat = inv_preprocess(image_mats, num_images=image_mats.shape[0])
    img_mat = img_mat[0].permute(1, 2, 0)
    im = np.array(img_mat.cpu())
    im = np.ascontiguousarray(im)
    print('points', points.shape)

    points = np.array(points)
    print('im', im.shape)
    print('points', points.shape)
    for i in range(bs):
        pnts = points[i]
        pnts = pnts.reshape(-1, 2)
        if not is_normed:
            pnts *= np.array([w, h])
        print(i, pnts.shape, pnts.dtype)
        im = image_draw_circles(im, np.ceil(pnts).astype(np.int64), (0, 255, 0))
        cv2.imshow('show_sampled_points', im)
        if cv2.waitKeyEx() == 27:
            cv2.destroyAllWindows()

def decode_parsing_numpy(labels, is_pred=False):
    """Decode batch of segmentation masks.

    Args:
      labels: input matrix to show

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    pred_labels = labels.copy()
    if is_pred:
        pred_labels = np.argmax(pred_labels, dim=1)
    assert pred_labels.ndim >= 2
    if pred_labels.ndim == 3:
        n, h, w = pred_labels.shape
    else:
        h, w = pred_labels.shape
        n = 1 
        pred_labels = pred_labels[None]

    labels_color = np.zeros([n, 3, h, w], dtype=np.uint8)
    for i, c in enumerate(COLORS):
        c0 = labels_color[:, 0, :, :]
        c1 = labels_color[:, 1, :, :]
        c2 = labels_color[:, 2, :, :]

        c0[pred_labels == i] = c[0]
        c1[pred_labels == i] = c[1]
        c2[pred_labels == i] = c[2]
    if n == 1:
        labels_color = np.transpose(labels_color[0], (1,2,0))

    return labels_color

def vis_depth_pred(depth_pred):
    dpth_mat = depth_pred * 1000
    dpth_color = cv2.applyColorMap(
            cv2.convertScaleAbs(dpth_mat, alpha=0.03), cv2.COLORMAP_JET)
    return dpth_color

def save_dense_pred(depth_pred, gt_depth, seg_pred, gt_seg, img, save_file):
    vis_depth = vis_depth_pred(depth_pred)
    vis_gt_depth = vis_depth_pred(gt_depth)
    vis_seg = decode_parsing_numpy(seg_pred)
    vis_gt_seg = decode_parsing_numpy(gt_seg)
    cv2.imwrite(save_file+'_depth.png', vis_depth)
    cv2.imwrite(save_file+'_seg.png', vis_seg)
    cv2.imwrite(save_file+'_gt-depth.png', vis_gt_depth)
    cv2.imwrite(save_file+'_gt-seg.png', vis_gt_seg)
    cv2.imwrite(save_file+'.png', img)

if __name__ == '__main__':
    # if (isInside(0.1,0.1, 20.9,0.1, 10.01,15.5, 10.01,15.5)):
    #     print('Inside')
    # else:
    #     print('Not Inside')

    triangles = torch.tensor([
        [[0.0,0.0, 20.,0.0, 10.0,15],
        [0.0,0.0, -10.,-20, 10.0,-20],
        [0,-20, 15,-20, 17,5]],
        
        [[-20,0.0, -20.,10.0, -10.0,20],
        [0.0,0.0, -10.,-20, 10.0,-20],
        [0,-20, 15,-20, 17,5]]
    ]) #(2, 3,6)
    points = torch.tensor([
        [[10,15], [0.0,0.0], [-15, 11], [2., -10], [10, -8]],
        [[-15,10], [0.0,0.0], [-15, 11], [2., -10], [10, -8]]
        ]
        #1,1,0,0,0
        #0,1,0,1,0
        #0,0,0,0,1

        #1,0,1,0,0
        #0,1,0,1,0
        #0,0,0,0,1
    ) #(2, 5,2)
    trng = triangles.reshape(2,3,3,2)
    trng[:,:,:,0] = (trng[:,:,:, 0] / (20-1)) * 2 - 1
    trng[:,:,:,1] = (trng[:,:,:, 1] / (20-1)) * 2 - 1
    points[:,:,0] = (points[:,:,0] / (20 -1)) * 2 - 1
    points[:,:,1] = (points[:,:,1] / (20 -1)) * 2 - 1
    isin = batch_isin_triangle(triangles, points)
    print(isin)
    print(isin.shape)
    isinnn = isin.reshape(2, 3, 5)
    print(isinnn)
