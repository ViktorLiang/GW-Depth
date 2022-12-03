"""
Train and eval functions used in main.py

modified based on https://github.com/facebookresearch/detr/blob/master/engine.py
"""
import math
import os
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import util.misc as utils
from util.commons import inv_preprocess, show_labels, save_dense_pred
from util.metrics import compute_mean_ioU, compute_depth_errors
# import cv2
import random

from evaluation.eval_post_online import vis_pred_lines

def train_one_epoch(model, criterions, postprocessors, data_loader, optimizer, device, epoch, max_norm, args, save_dir=None):
    model.train()
    criterion, criterion_depth, criterion_seg, criterion_plane = criterions
    if args.with_line:
        criterion.train()
    if args.with_dense:
        criterion_depth.train()
        criterion_seg.train()
    if args.with_plane_norm_loss:
        criterion_plane.train()
        # if sum(args.points_double) > 0:
        #     criterion_point.train()
        #     points_super = True

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    # epoch_iter = 0
    input_log_saved = False
    torch.cuda.empty_cache()
    # for samples, depth_gt, seg_gt, reflc_png, reflc_points, targets in metric_logger.log_every(data_loader, print_freq, header):
    for samples, depth_gt, seg_gt, targets, img_name in metric_logger.log_every(data_loader, print_freq, header):
        #reflc_points = torch.stack(reflc_points)
        if random.random() > args.input_log_freq and not input_log_saved:
            #show_labels(samples.tensors[0], targets[0], points=reflc_points[0], need_inv=True, save_dir=save_dir, epoch=epoch)
            show_labels(samples.tensors[0], targets[0], need_inv=True, save_dir=save_dir, epoch=epoch)
            input_log_saved = True
        samples = samples.to(device)
        depth_gt = depth_gt.to(device)
        seg_gt = seg_gt.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # reflc_mat = F.interpolate(reflc_png.tensors, scale_factor=0.5, mode='nearest')
        # reflc_mat = reflc_mat.cuda(device)
        try:
            #outputs = model(samples, reflc_points=reflc_points, reflc_mat=None)
            img_name = img_name[0].strip()
            outputs = model(samples, reflc_mat=None, img_name=img_name)
            if args.with_line:
                loss_point_dict = criterion(outputs, targets, depth_gt=depth_gt.tensors)
            
            if args.with_dense:
                mask = (depth_gt.tensors >= 0.2) & (depth_gt.tensors < 10.0)
                depth_loss_log = []
                seg_loss_log = []
                if isinstance(outputs['pred_depth'], torch.Tensor):
                    loss_depth = criterion_depth(outputs['pred_depth'], depth_gt.tensors, mask.to(torch.bool))
                elif isinstance(outputs['pred_depth'], list):
                    depth_loss_weights = args.depth_loss_weights
                    loss_depth = 0.0
                    d_gt_list = []
                    for i, pd in enumerate(outputs['pred_depth']):
                        pH, pW = pd.shape[-2:]
                        d_gt = F.interpolate(depth_gt.tensors, size=(pH, pW), mode='nearest')
                        m_rs = F.interpolate(mask.to(torch.uint8), size=(pH, pW), mode='nearest')
                        loss_d = criterion_depth(pd, d_gt, m_rs.to(torch.bool))
                        loss_d = loss_d * depth_loss_weights[i]
                        loss_depth = loss_depth + loss_d
                        depth_loss_log.append(loss_d)
                        d_gt_list.append(d_gt)
                    
                    # def forward(self, depth_pred, depth_gt, line_pred, line_score, valid_mask, depth_normal_pred=None):
                    if args.with_plane_norm_loss:
                        loss_plane = criterion_plane(outputs['pred_depth'][-1], d_gt_list[-1], outputs['pred_lines'], outputs['pred_logits'], mask)

                if isinstance(outputs['pred_seg'], torch.Tensor):
                    loss_seg = criterion_seg(outputs['pred_seg'], seg_gt.tensors.squeeze(1))
                    loss_seg = loss_seg * args.seg_loss_weight
                elif isinstance(outputs['pred_seg'], list):
                    seg_loss_weights = args.seg_loss_weights
                    loss_seg = 0.0
                    for i, ps in enumerate(outputs['pred_seg']):
                        pH, pW = ps.shape[-2:]
                        s_gt = F.interpolate(seg_gt.tensors.type(torch.float), size=(pH, pW), mode='nearest')
                        loss_seg = criterion_seg(ps, s_gt.squeeze(1).type(torch.long))
                        loss_s = loss_seg * seg_loss_weights[i]
                        loss_seg = loss_seg + loss_s
                        seg_loss_log.append(loss_s)

        except RuntimeError as e:
            if "out of memory" in str(e):
                sys.exit('Out Of Memory')   
            else:
                raise e
        
        if args.with_line:
            weight_dict = criterion.weight_dict
            losses = sum(loss_point_dict[k] * weight_dict[k] for k in loss_point_dict.keys() if k in weight_dict)
        if args.with_dense:
            if args.with_line:
                losses = losses + loss_depth + loss_seg
            else:
                losses = loss_depth + loss_seg

            loss_dense = {}
            ld_names = ['1/16', '1/8',  '1/4', '1']
            if len(depth_loss_log) > 0:
                for i, lss in enumerate(depth_loss_log):
                    loss_dense["loss_depth_"+ld_names[i]] = lss.item()
            else:
                loss_dense['loss_depth'] = loss_depth.item()
            
            if len(seg_loss_log) > 0:
                for i, lss in enumerate(seg_loss_log):
                    loss_dense["loss_seg_"+ld_names[i]] = lss.item()
            else:
                loss_dense['loss_seg'] = loss_seg.item()

        else:
            loss_dense = {}
        
        if args.with_plane_norm_loss:
            loss_dense['loss_plane'] = loss_plane * args.plane_norm_loss_coef
        # reduce losses over all GPUs for logging purposes
        #loss_dict_reduced = utils.reduce_dict(loss_point_dict)
        #loss_dict_reduced_unscaled = {f'{k}_unscaled': v   for k, v in loss_dict_reduced.items()}
        #loss_dict_reduced_scaled = {k: v * weight_dict[k]  for k, v in loss_dict_reduced.items() if k in weight_dict}
        #losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        #loss_value = losses_reduced_scaled.item()

        loss_value = losses.detach().clone().item() 

        if args.with_line:
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v.detach().clone().cpu()   for k, v in loss_point_dict.items()}
            loss_dict_reduced_scaled = {k: v.detach().clone().cpu() * weight_dict[k]  for k, v in loss_point_dict.items() if k in weight_dict}

        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value, 'loss_dense', loss_dense)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        if args.with_line:
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled, **loss_dense)
        else:
            metric_logger.update(loss=loss_value, **loss_dense)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # epoch_iter += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterions, postprocessors, data_loader, base_ds, device, output_dir, args, save_dir=None, epoch=0, save_dense=False, save_line=False):
    model.eval()
    criterion, _, _, _ = criterions
    if args.with_line:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if args.coco_path is not None:
        id_to_img = {}
        f = open(os.path.join(args.coco_path, "annotations", "lines_{}2017.json".format(args.dataset)))
        data = json.load(f)
        for d in data['images']:
            id_to_img[d['id']] = d['file_name'].split('.')[0]
    else:
        id_to_img = data_loader.dataset.id_to_img
    # print('id_to_img', id_to_img)
    vis_save_dir = save_dir
    if save_dense:
        checkpoint_name = os.path.basename(args.resume)
        checkpoint_name = checkpoint_name.split('.')[0]
        dense_pred_log = vis_save_dir + '/dense_pred/'+checkpoint_name
        os.makedirs(dense_pred_log, exist_ok=True)

    processing_id = 0
    seg_preds = []
    seg_gts = []
    depth_eval_measures = torch.zeros(10).cuda()
    metric_names = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
    d1_less = []
    # for samples, depth_gt, seg_gt, reflc_png, reflc_points, targets in metric_logger.log_every(data_loader, 1, header):
    for samples, depth_gt, seg_gt, targets, img_name in metric_logger.log_every(data_loader, 1, header):
        curr_img_id = targets[0]['image_id'].item()
        imname = id_to_img[curr_img_id]
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # reflc_mat = F.interpolate(reflc_png.tensors, scale_factor=0.5, mode='nearest')
        # reflc_mat = reflc_mat.cuda(device)

        # reflc_points = torch.stack(reflc_points)
        #outputs = model(samples, reflc_points=reflc_points, reflc_mat=None)
        img_name = img_name[0].strip()
        outputs = model(samples, reflc_mat=None, img_name=img_name)

        if args.with_line:
            loss_point_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_point_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
            metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                                **loss_dict_reduced_scaled,
                                **loss_dict_reduced_unscaled)
        
        if args.with_dense:
            gt_seg_cpu = seg_gt.tensors.cpu().squeeze(1)
            seg_gts.append(gt_seg_cpu)
            if isinstance(outputs['pred_seg'], list):
                pred_seg_cpu = outputs['pred_seg'][-1].cpu().argmax(1)
                seg_preds.append(pred_seg_cpu)
            else:
                pred_seg_cpu = outputs['pred_seg'].cpu().argmax(1)
                seg_preds.append(pred_seg_cpu)
                

            if isinstance(outputs['pred_depth'], list):
                pred_depth = outputs['pred_depth'][-1].clone().cpu().numpy().squeeze()
            else:
                pred_depth = outputs['pred_depth'].clone().cpu().numpy().squeeze()

            gt_depth = depth_gt.tensors.clone().cpu().numpy().squeeze()
            # depth error
            pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
            pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
            pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
            pred_depth[np.isnan(pred_depth)] = args.min_depth_eval
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            measures = compute_depth_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            
            single_metric = {}
            for m, d in zip(metric_names, measures):
                single_metric[m] = d
            # print(img_name, single_metric)
            # # if single_metric['d1'] < 0.6:
            if single_metric['rms'] > 0.5:
                d1_less.append([img_name, single_metric])
            depth_eval_measures[:9] += torch.tensor(measures).cuda()
            depth_eval_measures[9] += 1

            if save_dense:
                im = samples.tensors
                img_mat = inv_preprocess(im, num_images=1)
                img_mat = img_mat[0].permute(1, 2, 0)
                img_mat_npy = np.array(img_mat * 255, dtype=np.uint8)
                save_file = dense_pred_log+'/'+imname
                pred_seg_cpu_npy = np.array(pred_seg_cpu)
                gt_seg_cpu_npy = np.array(gt_seg_cpu)
                save_dense_pred(pred_depth, gt_depth, pred_seg_cpu_npy, gt_seg_cpu_npy, img_mat_npy, save_file)
       
                             
        # outputs['pred_logits'] #[1, n, 2]
        # outputs['pred_lines'] #[1, n, 4]
        # targets['lines'] #[n, 4]
        # targets['labels'] #[n,]
        # targets['image_id'] #[1,]
        # targets['orig_size'] #[2,]
        # targets['size'] #[2,]

        ## save visualized pred line
        if args.with_line and save_line:
            pred_scores = F.softmax(outputs['pred_logits'][0], -1)
            pred_lines = outputs['pred_lines'][0].reshape(-1, 3, 2).flip(-1)#yxyx
            gt_lines = targets[0]['lines'].reshape(-1, 3, 2).flip(-1)
            # pred_lines = outputs['pred_lines'][0][:, :4].reshape(-1, 2, 2).flip(-1)#yxyx
            # gt_lines = targets[0]['lines'][:, :4].reshape(-1, 2, 2).flip(-1)
            curr_img_id = targets[0]['image_id'].tolist()[0]
            im = samples.tensors
            img_mat = inv_preprocess(im, num_images=1)
            img_mat = img_mat[0].permute(1, 2, 0)
            imname = id_to_img[curr_img_id]+str(processing_id)
            pred_scores = pred_scores.cpu().numpy()[:,  0]
            pred_lines = pred_lines.cpu().numpy()
            gt_lines = gt_lines.cpu().numpy()

            checkpoint_name = os.path.basename(args.resume)
            checkpoint_name = checkpoint_name.split('.')[0]
            line_pred_log = vis_save_dir + '/line_pred/'+checkpoint_name
            os.makedirs(line_pred_log, exist_ok=True)
            vis_pred_lines(pred_lines, pred_scores, img_mat, gt_lines, imname, line_pred_log)
        processing_id += 1
    # print(d1_less)
    # print('len(d1_less)', len(d1_less))
    if args.with_dense:
        # segmentation metrics
        seg_iou_dict = compute_mean_ioU(seg_preds, seg_gts, num_classes=2)
        metric_logger.update(**seg_iou_dict)
        # average depth value
        depth_eval_measures_cpu  = depth_eval_measures.cpu()
        cnt = depth_eval_measures_cpu[9]
        depth_eval_measures_cpu = depth_eval_measures_cpu / cnt
        d_metric_names = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
        print('Computing errors for {} eval samples'.format(int(cnt)))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                    'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
        depth_metrics_dicts  = {}
        for i in range(9):
            print('{:7.3f}, '.format(depth_eval_measures_cpu[i]), end='')
            depth_metrics_dicts[d_metric_names[i]] = depth_eval_measures_cpu[i]
        metric_logger.update(**depth_metrics_dicts)

        #wirte eval results
        if args.append_word is not None:
            with open(vis_save_dir+'/eval_results.txt', 'a+') as f:
                f.write('epoch'+args.append_word+' depth:'+str(depth_metrics_dicts)+' segmentation:'+str(dict(seg_iou_dict))+'\n')
        else:
            with open(vis_save_dir+'/eval_results.txt', 'a+') as f:
                f.write('oneline eval epoch'+str(epoch)+' depth:'+str(depth_metrics_dicts)+' segmentation:'+str(dict(seg_iou_dict))+'\n')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats
