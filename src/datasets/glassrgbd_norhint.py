# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from hashlib import new
from matplotlib.pyplot import axis
import numpy as np
import torch
import torch.utils.data.distributed
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import os

from datasets.coco import make_coco_transforms
from util.commons import gen_pairs, read_json_label

import argparse
import sys

import json
# import cv2
from util.commons import show_labels

def get_args_parser():
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
    parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen_v2')
    parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                        'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                                default='densenet161_bts')
    # Dataset
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True, default='/home/ly/data/datasets/trans-depth/Glass-RGBD/images/')
    parser.add_argument('--gt_depth_path',                   type=str,   help='path to the groundtruth data', required=True, default='/home/ly/data/datasets/trans-depth/Glass-RGBD/depth/')
    parser.add_argument('--gt_seg_path',               type=str,   help='path to the coarse depth map', required=True,  default='/home/ly/data/datasets/trans-depth/Glass-RGBD/segmentation/')
    parser.add_argument('--gt_line_path',               type=str,   help='path to the coarse depth map', required=True,  default='/home/ly/data/datasets/trans-depth/Glass-RGBD/polygon_json/')
    parser.add_argument('--filenames_file_train',      type=str,   help='path to train filenames text file', required=True, default='/home/ly/data/datasets/trans-depth/Glass-RGBD/train.txt')
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to eval filenames text file', required=True, default='/home/ly/data/datasets/trans-depth/Glass-RGBD/eval.txt')
    parser.add_argument('--glassrgbd_images_json',     type=str,   help='path to id-imagename map json file', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=480)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
    parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

    # Log and save
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
    parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
    parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)

    # Training
    parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
    parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
    parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
    parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
    parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
    parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
    parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
    parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
    parser.add_argument('--att_rank',                  type=int,   help='initial rank in attention structure', default=1)
    # Preprocessing
    parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
    parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

    # Multi-gpu training
    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
    parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
    parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
    parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
    parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
    parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
    parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                        'N processes per node, which has N GPUs. This is the '
                                                                        'fastest way to use PyTorch for either single node or '
                                                                        'multi node data parallel training', action='store_true',)
    # Online eval
    parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
    parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
    parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
    parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
    parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
    parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                        'if empty outputs to checkpoint folder', default='')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--exp', type=str, default='noname', help='experiment name')

    return parser


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class ConvertLinePolysToMask(object):
    def __init__(self) -> None:
        self.label_map = {'wall':0, 'window':0, 'door':0, 'guardrail':0}

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        lines = torch.as_tensor(target["lines"], dtype=torch.float32)
        poly_centers = torch.as_tensor(target["poly_centers"], dtype=torch.float32)
        if len(lines) > 0:
            lines[:, 0::2].clamp_(min=0, max=w)
            lines[:, 1::2].clamp_(min=0, max=h)
            poly_centers[:, 0].clamp_(min=0, max=w)
            poly_centers[:, 1].clamp_(min=0, max=h)
            
        classes = torch.tensor(target["labels"], dtype=torch.int64)
        poly_ids = torch.tensor(target["poly_ids"], dtype=torch.int64)

        target = {}
        target["lines"] = lines

        target["labels"] = classes
        
        target["poly_ids"] = poly_ids

        target["poly_centers"] = poly_centers

        target["image_id"] = image_id
        area = torch.tensor([1 for _ in lines])
        iscrowd = torch.tensor([0 for _ in lines])
        target["area"] = area
        target["iscrowd"] = iscrowd

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target 

def centroid(vertexes):
    _x_list = [vertex [0] for vertex in vertexes]
    _y_list = [vertex [1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return(_x, _y)

def generate_line_labels(lines_gt_dict):
    poly_ids = []
    lines = []
    poly_centers = []
    for poly in lines_gt_dict['shapes']:
        if len(poly['points']) == 0:
            continue
        poly_lines = gen_pairs(np.array(poly['points']))
        poly_lines = poly_lines.reshape(-1, 4)
        lines += poly_lines.tolist()

        # center point of one polygon
        all_points = poly_lines[0].reshape(-1, 2).tolist() + poly_lines[1:, 2:].tolist()
        ply_center = centroid(all_points)
        # polygon id is unique for each polygon wihtin one image
        for _ in poly_lines:
            poly_ids.append(poly['poly_id'])
            poly_centers.append(ply_center)


    lines_class = np.full((len(lines),), 0) # (n,)
    lines = np.array(lines) # (n, 4)
    poly_ids = np.array(poly_ids) # (n,)
    lines_origin = {'lines': lines, 
                    'labels':lines_class,
                    'poly_ids': poly_ids,
                    'poly_centers': poly_centers,
                    'image_size':(lines_gt_dict['imageWidth'], lines_gt_dict['imageHeight']),
                    'image_id':lines_gt_dict['imageId']}
    return lines_origin          
            
class DataLoadPreprocess(torchvision.datasets.CocoDetection):
    def __init__(self, args, mode, transforms=None, log_dir=None):
        self.args = args
        if mode == 'train':
            with open(args.filenames_file_train, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
    
        self.mode = mode
        self._transforms = transforms
        self.log_dir = log_dir
        self.prepare = ConvertLinePolysToMask()
        self.id_to_img = {}
        self.norm_mean = [0.538, 0.494, 0.453]
        self.norm_std = [0.257, 0.263, 0.273]
        with open(args.glassrgbd_images_json, 'r') as f:
            data = json.load(f)
            for d in data['images']:
                self.id_to_img[d['id']] = d['file_name'].split('.')[0]
        print('Datloader transform:', self._transforms)
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # train, val 
        img_path=sample_path.split()[0]
        image_path = os.path.join(self.args.data_path,  img_path+'.png')
        # line labels
        line_labels_path = os.path.realpath(self.args.gt_line_path)
        line_path = os.path.join(line_labels_path, img_path+'.json')
        lines_gt_dict = read_json_label(line_path)
        lines_gt = generate_line_labels(lines_gt_dict)
        if len(lines_gt['lines']) == 0:
            print(sample_path, ' has no lines')

        #dense lables
        depth_path = os.path.join(self.args.gt_depth_path, img_path+'.png')
        seg_path = os.path.join(self.args.gt_seg_path, img_path+'.png')
        #rhint_jsn_path = os.path.join(self.args.glassrgbd_rhint_points_path, img_path+'.json')
        #rhint_png_path = os.path.join(self.args.glassrgbd_rhint_path, img_path+'.png')
    
        image = Image.open(image_path)
        depth_gt = Image.open(depth_path)
        seg_gt = Image.open(seg_path)
        #refl_png = Image.open(rhint_png_path)
        image, targets = self.prepare(image, lines_gt)

        #rhint = np.array(read_json_label(rhint_jsn_path, key='rhint_points'))
        #reflection_points = torch.as_tensor(rhint, dtype=torch.float32)
        #targets['reflection_points'] = reflection_points.flip(1) # [row, column] to [column, row]
        
        # np_image = np.asarray(image)
        # show_labels(np_image, targets, points=targets['reflection_points'], need_inv=False, tittle='no trans', 
        #     save_dir='/home/ly/workspace/git/segm/line_segm/letr-depth-2/exp/local_test_log/local_vis_unormed')

        if self._transforms is not None:
            image, targets, aux_mats = self._transforms(image, targets, aux_mats = [depth_gt, seg_gt])
            depth_gt, seg_gt = aux_mats
        
        depth_gt = depth_gt / 1000.0
        # all glass like segmentation are treated as one class
        seg_gt = torch.where(seg_gt > 0, 1, 0)
        seg_gt = seg_gt.type(torch.long)

        # merge line coords and glass center coors
        if self.args.with_center:
            targets['lines'] = torch.cat([targets['lines'], targets['poly_centers']], dim=1)
        # show_labels(image, targets, points=targets['reflection_points'], need_inv=True, tittle='transed', 
        #     save_dir='/home/ly/workspace/git/segm/line_segm/letr-depth-2/exp/local_test_log/local_vis')
       
        # write reflection points log when zero reflection points found.
        #reflec_points = targets['reflection_points']
        #if targets['reflection_points'].shape[0] == 0:
        #    show_labels(image, targets, points=reflec_points, need_inv=True, tittle='transed', save_dir=self.log_dir, mode=self.mode)
        # normalize reflection png
        #refl_png = refl_png.squeeze(0).permute(2, 0, 1).contiguous()
        #refl_png = refl_png.to(dtype=torch.get_default_dtype()).div(255)
        #refl_png = F.normalize(refl_png.type(torch.float), mean=self.norm_mean, std=self.norm_std)
        
        del targets['poly_centers']
        del targets['area']
        del targets['iscrowd']

        # check_keys = ['lines', 'labels', 'poly_ids', 'lines_depth']
        assert targets['lines'].size(0) == targets['labels'].size(0) == targets['poly_ids'].size(0), targets
        return image, depth_gt, seg_gt, targets, img_path
    
    def __len__(self):
        return len(self.filenames)



def loadertest(args):
    D = DataLoadPreprocess(args, mode='train', transforms=make_coco_transforms('val', args))
    save_base_dir = os.path.dirname(args.filenames_file_eval)
    save_dir  = save_base_dir+'/lines_npz/eval'
    save_img_dir  = save_base_dir+'/split_images/eval'
    import cv2
    import shutil
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print('making dir:', save_dir)
    if not os.path.isdir(save_img_dir):
        os.makedirs(save_img_dir)
        print('making dir:', save_img_dir)
    org_im_folder = save_base_dir + '/images'
    for im, gt, imname in D:
        print('im shape:', im.shape, im.dtype, imname)
        org_im_file = org_im_folder +'/' + imname+'.png'
        h, w = gt['size']
        lines = gt['lines']
        lines *= 128
        lines = lines.reshape(-1, 2, 2) # xyxy
        lines = lines.flip(-1) #yxyx
        label_dicts = {"lpos": lines, "file_name": imname, "image_id": gt['image_id'].item()}
        save_name = save_dir+'/'+imname
        np.savez(save_name, **label_dicts)

        save_im_name = save_img_dir+'/'+imname
        shutil.copy(org_im_file, save_im_name)
        print(save_name)


def build(image_set,  args, log_dir=None):
    dataset = DataLoadPreprocess(args, mode=image_set, transforms=make_coco_transforms(image_set, args), log_dir=log_dir)
    return dataset

if __name__ == '__main__':
    parser = get_args_parser()
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    loadertest(args)
