"""
This file provides coarse stage LETR definition
Modified based on https://github.com/facebookresearch/detr/blob/master/models/backbone.py
"""
import enum
from tkinter.messagebox import NO
from matplotlib import transforms
from sklearn.pipeline import FeatureUnion
import torch
import torch.nn.functional as F
from torch import nn
from matplotlib.path import Path
import numpy as np

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .dense_upsample import build_depth_decoder
# from .dense_multitask_upsample import build_depth_decoder

#from .geometry_transformer import build_dense_transformer
from .multiscale_transformerr import build_dense_transformer

from src.models.losses.sobel import Sobel

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class GlassRGBD(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, args, aux_loss=False,
                dense_encoder=None, depth_decoder=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        channel = [256, 512, 1024, 2048]
        if args.with_line and transformer is not None:
            hidden_dim = transformer.d_model
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            self.input_proj = nn.Conv2d(channel[args.layer1_num], hidden_dim, kernel_size=1)
            if args.with_center:
                # line points plus one center point
                self.lines_embed  =  MLP(hidden_dim, hidden_dim, 4 + 2, 3)
                # depth value for two points of line
                # self.lines_depth_embed  =  MLP(hidden_dim, hidden_dim, 2, 3)
            else:
                self.lines_embed  =  MLP(hidden_dim, hidden_dim, 4, 3)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.args = args

        if args.with_dense: 
            hidden_dim = 256
            self.dense_input_proj = nn.Conv2d(channel[args.layer1_num], hidden_dim * 2, kernel_size=1)
            self.dense_encoder = dense_encoder
            self.depth_decoder = depth_decoder

    def forward(self, samples, reflc_points=None, reflc_mat=None, img_name=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        inp_h, inp_w = samples.tensors.shape[-2:]

        features, pos = self.backbone(samples)

        num = self.args.layer1_num
        src, mask = features[num].decompose()
        assert mask is not None
        
        out = {}
        if self.args.with_line:
            trans_input = self.input_proj(src)
            hs, line_encoder_feats = self.transformer(trans_input, mask, self.query_embed.weight, pos[num])
            outputs_class = self.class_embed(hs)
            outputs_coord = self.lines_embed(hs).sigmoid()
            out = {'pred_logits': outputs_class[-1], 'pred_lines': outputs_coord[-1]}

            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        else:
            out['pred_lines'] = None
            out['pred_logits'] = None

            
        if self.args.with_dense:
            dense_input = self.dense_input_proj(src)
            dense_input_nest = NestedTensor(dense_input, mask)
            _, C, h, w = dense_input.shape
            # d_enc_out, H, W, x, Wh, Ww = self.dense_encoder(dense_input.flatten(2).permute(0, 2, 1), h, w)
            layers_size = [(f.tensors.shape[2], f.tensors.shape[3]) for f in features[:-1]]
            layers_size.reverse()
            # d_enc_out_list, depth_token, seg_token, depth_preds, seg_preds = self.dense_encoder(dense_input_nest, features, 
            #         out['pred_lines'], out['pred_logits'], 
            #         layers_size=layers_size, reflc_points=reflc_points, reflc_mat=reflc_mat, 
            #         input_size=(inp_h, inp_w), input_images=samples, img_name=img_name)
            d_enc_out_list, depth_token, seg_token, gen_points, depth_preds = self.dense_encoder(dense_input_nest, features, 
                    out['pred_lines'], out['pred_logits'], 
                    layers_size=layers_size, reflc_points=reflc_points, reflc_mat=reflc_mat, 
                    input_size=(inp_h, inp_w), input_images=samples)

            # depth_pred, seg_pred = self.depth_decoder(d_enc_out_list[-1], depth_preds[-1], seg_preds[-1], depth_token, seg_token, (inp_h, inp_w))
            # seg_preds.append(seg_pred)
            depth_pred, seg_pred = self.depth_decoder(d_enc_out_list[-1], depth_preds[-1], depth_token, seg_token, (inp_h, inp_w))
            # depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled
            depth_preds.append(depth_pred)
            out['pred_depth'] = depth_preds
            out['pred_seg'] = seg_pred
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_lines': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    
    @torch.jit.unused
    def _set_aux_depth_loss(self, outputs_class, outputs_coord, outputs_center_depth):
        return [{'pred_logits': a, 'pred_lines': b, 'pred_lines_depth': c} for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_center_depth[:-1])]

class SetCriterion(nn.Module):

    def __init__(self, num_classes, weight_dict, eos_coef, losses, args, matcher=None):

        super().__init__()
        self.num_classes = num_classes

        self.matcher = matcher

        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.args = args
        try:
            self.args.label_loss_params = eval(self.args.label_loss_params)  # Convert the string to dict.
        except:
            pass

    def loss_lines_labels(self, outputs, targets,  num_items,  log=False, origin_indices=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_lines]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(origin_indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, origin_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if self.args.label_loss_func == 'cross_entropy':
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        elif self.args.label_loss_func == 'focal_loss':
            loss_ce = self.label_focal_loss(src_logits.transpose(1, 2), target_classes, self.empty_weight, **self.args.label_loss_params)
        else:
            raise ValueError()

        losses = {'loss_ce': loss_ce}
        return losses

    def label_focal_loss(self, input, target, weight, gamma=2.0):
        """ Focal loss for label prediction. """
        # In our case, target has 2 classes: 0 for foreground (i.e. line) and 1 for background.
        # The weight here can serve as the alpha hyperparameter in focal loss. However, in focal loss, 
        # 
        # Ref: https://github.com/facebookresearch/DETR/blob/699bf53f3e3ecd4f000007b8473eda6a08a8bed6/models/segmentation.py#L190
        # Ref: https://medium.com/visionwizard/understanding-focal-loss-a-quick-read-b914422913e7

        # input shape: [batch size, #classes, #queries]
        # target shape: [batch size, #queries]
        # weight shape: [#classes]

        prob = F.softmax(input, 1)                                          # Shape: [batch size, #classes, #queries].
        ce_loss = F.cross_entropy(input, target, weight, reduction='none')  # Shape: [batch size, #queries].
        p_t = prob[:,1,:] * target + prob[:,0,:] * (1 - target)             # Shape: [batch size, #queries]. Note: prob[:,0,:] + prob[:,1,:] should be 1.
        loss = ce_loss * ((1 - p_t) ** gamma)
        loss = loss.mean()    # Original label loss (i.e. cross entropy) does not consider the #lines, so we also do not consider that.
        return loss

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets,  num_items, origin_indices=None):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty lines
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_lines_POST(self, outputs, targets, num_items, origin_indices=None):
        assert 'POST_pred_lines' in outputs

        if outputs['POST_pred_lines'].shape[1] == 1000:
            idx = self._get_src_permutation_idx(origin_indices)

            src_lines = outputs['POST_pred_lines'][idx]
            
        else:
            src_lines = outputs['POST_pred_lines'].squeeze(0)


        target_lines = torch.cat([t['lines'][i] for t, (_, i) in zip(targets, origin_indices)], dim=0)

        loss_line = F.l1_loss(src_lines, target_lines, reduction='none')

        losses = {}
        losses['loss_line'] = loss_line.sum() / num_items

        return losses

    def loss_lines(self, outputs, targets, num_items, origin_indices=None):
        assert 'pred_lines' in outputs

        idx = self._get_src_permutation_idx(origin_indices)

        src_lines = outputs['pred_lines'][idx]
        target_lines = torch.cat([t['lines'][i] for t, (_, i) in zip(targets, origin_indices)], dim=0)

        loss_line = F.l1_loss(src_lines, target_lines, reduction='none')

        losses = {}
        losses['loss_line'] = loss_line.sum() / num_items

        return losses

    def loss_lines_depeth(self, outputs, targets, num_items, origin_indices=None, depth_gt=None):
        assert 'pred_lines_depth' in outputs, outputs.keys()
        idx = self._get_src_permutation_idx(origin_indices)

        pline_depth_preds = []
        pline_depth_gt = []

        bsz = depth_gt.shape[0]
        for b in range(bsz):
            h, w = targets[b]['size']
            bidx = (idx[0] == b).tolist()
            pidx = (idx[1][bidx]).tolist()
            plines = outputs['pred_lines'][b][pidx][:, :4]
            pdepth = outputs['pred_lines_depth'][b][pidx]
            pdepth_rec = pdepth * self.args.max_depth

            dgt = depth_gt[b].squeeze(0)
            denormed_lines = plines * torch.tensor([w, h, w, h], dtype=torch.float32).cuda(plines.device)
            denormed_lines = torch.round(denormed_lines)
            denormed_lines[:, 0::2].clamp_(min=0, max=w-1)
            denormed_lines[:, 1::2].clamp_(min=0, max=h-1)
            denormed_lines = denormed_lines.reshape(-1, 2).type(torch.long)
            lines_gt_depth = dgt[denormed_lines[:, 1], denormed_lines[:, 0]]
            pdepth_rec = pdepth_rec.reshape(-1,1).squeeze(1)
            pline_depth_preds.append(pdepth_rec)
            pline_depth_gt.append(lines_gt_depth)

        pline_depth_gt = torch.cat(pline_depth_gt)
        pline_depth_preds = torch.cat(pline_depth_preds)
        # d = torch.log(pline_depth_preds) - torch.log(pline_depth_gt)
        # dloss = torch.sqrt((d ** 2).mean() - self.args.variance_focus * (d.mean() ** 2)) * 10.0

        dloss = F.l1_loss(pline_depth_preds, pline_depth_gt, reduction='none')

        losses = {}
        losses['loss_line_depth'] = dloss
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, num_items, **kwargs):
        
        loss_map = {
            'POST_lines_labels': self.loss_lines_labels,
            'POST_lines': self.loss_lines,
            'lines_labels': self.loss_lines_labels,
            'cardinality': self.loss_cardinality,
            'lines': self.loss_lines,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_items, **kwargs)
    
    def forward(self, outputs, targets, origin_indices=None, depth_gt=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        
        origin_indices = self.matcher(outputs_without_aux, targets)


        num_items = sum(len(t["labels"]) for t in targets)

        num_items = torch.as_tensor([num_items], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_items)
        num_items = torch.clamp(num_items / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets,  num_items, origin_indices=origin_indices))
        # if self.args.with_center:
        #     losses.update(self.loss_lines_depeth(outputs, targets, num_items, origin_indices=origin_indices, depth_gt=depth_gt))
            
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        aux_name = 'aux_outputs'
        if aux_name in outputs:
            if 'loss_lines_depeth' in self.losses:
                aux_losses = [self.losses[0], self.losses[1]]
            else:
                aux_losses = self.losses
            for i, aux_outputs in enumerate(outputs[aux_name]):
                
                origin_indices = self.matcher(aux_outputs, targets)

                # for loss in self.losses:
                for loss in aux_losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, num_items, origin_indices=origin_indices, **kwargs)

                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class SilogLoss(nn.Module):
    def __init__(self, variance_focus=0.85, log_depth_error=True):
        super(SilogLoss, self).__init__()
        self.variance_focus = variance_focus
        self.log_depth_error = log_depth_error

    def forward(self, depth_est, depth_gt, mask):
        if self.log_depth_error:
            d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        else:
            valid_depth_est = depth_est[mask]
            valid_depth_gt = depth_gt[mask]
            d = (valid_depth_est + torch.log(valid_depth_est)) - (valid_depth_gt + torch.log(valid_depth_gt))

        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()
        self.seg_loss = nn.CrossEntropyLoss()

    def forward(self, seg_pred, seg_gt):
        segloss = self.seg_loss(seg_pred, seg_gt)
        return segloss

class PlaneLoss(nn.Module):
    def __init__(self, num_ref, line_score_thresh=0.6, min_plane_area=100):
        super(PlaneLoss, self).__init__()
        self.num_ref = num_ref
        self.line_score_thresh = line_score_thresh
        self.min_plane_area = min_plane_area
        self.sobel = Sobel().cuda()
    
    def forward(self, depth_pred, depth_gt, line_pred, line_score, valid_mask):
        #depth_pred, line_pred, line_score torch.Size([1, 3, 576, 666]) torch.Size([1, 100, 6]) torch.Size([1, 100, 2])
        B = line_score.shape[0]
        assert B == 1, 'one image each iter'

        score_logits = torch.softmax(line_score, dim=-1)
        keep_id = score_logits[:, :, 0] > self.line_score_thresh
        kept_lines = line_pred[keep_id]
        top_num = min(kept_lines.shape[0], self.num_ref)
        _, t_ids = torch.topk(line_score[:, :, 0], top_num, dim=-1)
        
        # generate plane norm
        ones = torch.ones(depth_gt.size(0), 1, depth_gt.size(2), depth_gt.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)
        output_grad = self.sobel(depth_pred)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth_gt)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth_gt)
        depth_normal_pred = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
            
        H, W = depth_normal_pred.shape[-2:]
        choosen_lines = torch.stack([line_pred[i][t_ids[i]] for i in range(B)])
        # choosen_points = choosen_points.reshape(B, self.num_ref, -1, 3)
        denormed_lines = choosen_lines * torch.tensor([W, H, W, H, W, H], dtype=torch.float32).cuda(line_pred.device)
        denormed_lines = torch.round(denormed_lines)
        denormed_lines[:, :, 0::2].clamp_(min=0, max=W-1)
        denormed_lines[:, :, 1::2].clamp_(min=0, max=H-1)
        denormed_lines = denormed_lines.reshape(B, -1, 3, 2).type(torch.long) # (B, num_line, 3, 2)
    
        x, y = np.mgrid[:H, :W]
        all_coors = np.hstack((y.reshape(-1, 1), x.reshape(-1, 1))) # (N, 2)
        valid_vector = torch.flatten(valid_mask[0][0])
        valid_vector = np.array(valid_vector.cpu())
        valid_coors = all_coors[valid_vector, :]

        depth_normal_pred_vec = depth_normal_pred[0].flatten(1)
        loss_var_sum = torch.tensor(0.0).cuda(line_pred.device)
        total_plane=0
        for j in range(top_num):
            p = denormed_lines[0, j]
            poly_path = Path(p.tolist())
            # mask = poly_path.contains_points(all_coors)
            mask = poly_path.contains_points(valid_coors)
            # mask = mask.reshape(H, W)
            mask_t = torch.tensor(mask).cuda(line_pred.device)
            plane_area = torch.sum(mask_t)
            # remove too small area
            if plane_area < self.min_plane_area:
                continue
            plane_normal_x = torch.masked_select(depth_normal_pred_vec[0][valid_vector], mask_t)
            plane_normal_y = torch.masked_select(depth_normal_pred_vec[1][valid_vector], mask_t)
            loss_plane_var_x = torch.var(plane_normal_x, unbiased=False)
            loss_plane_var_y = torch.var(plane_normal_y, unbiased=False)
            loss_plane_var = loss_plane_var_x + loss_plane_var_y
            loss_var_sum = loss_var_sum + loss_plane_var
            total_plane += 1
        total_plane = max(1, total_plane)
        loss_var = loss_var_sum / total_plane
        return loss_var

class PostProcess_Line(nn.Module):

    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes, output_type):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        if output_type == "prediction":
            out_logits, out_line = outputs['pred_logits'], outputs['pred_lines']

            assert len(out_logits) == len(target_sizes)
            assert target_sizes.shape[1] == 2

            prob = F.softmax(out_logits, -1)
            scores, labels = prob[..., :-1].max(-1)

            # convert to [x0, y0, x1, y1] format
            img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            lines = out_line * scale_fct[:, None, :]

            results = [{'scores': s, 'labels': l, 'lines': b} for s, l, b in zip(scores, labels, lines)]
        elif output_type == "prediction_POST":
            out_logits, out_line = outputs['pred_logits'], outputs['POST_pred_lines']

            assert len(out_logits) == len(target_sizes)
            assert target_sizes.shape[1] == 2

            prob = F.softmax(out_logits, -1)
            scores, labels = prob[..., :-1].max(-1)

            # convert to [x0, y0, x1, y1] format
            img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            lines = out_line * scale_fct[:, None, :]

            results = [{'scores': s, 'labels': l, 'lines': b} for s, l, b in zip(scores, labels, lines)]
        elif output_type == "ground_truth":
            results = []
            for dic in outputs:
                lines = dic['lines']
                img_h, img_w = target_sizes.unbind(1)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                scaled_lines = lines * scale_fct
                results.append({'labels': dic['labels'], 'lines': scaled_lines, 'image_id': dic['image_id']})
        else:
            assert False
        return results


def build(args):
    num_classes = 1

    device = torch.device(args.device)
    
    backbone = build_backbone(args)

    if args.with_line:
        transformer = build_transformer(args)
    else:
        transformer = None

    if args.with_dense:
        dense_encoder = build_dense_transformer(args)
        depth_decoder = build_depth_decoder(args)
    else:
        dense_encoder = None
        depth_decoder = None
    
    model = GlassRGBD(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        args=args,
        aux_loss=args.aux_loss,
        dense_encoder=dense_encoder,
        depth_decoder=depth_decoder
    )

    matcher = build_matcher(args, type='origin_line')
    
    losses = []
    weight_dict = {}
    
    losses.append('lines_labels')
    losses.append('lines')
    weight_dict['loss_ce'] = 1
    weight_dict['loss_line'] = args.line_loss_coef

    if args.aux_loss:
        aux_layer = args.dec_layers
        aux_weight_dict = {}
        for i in range(aux_layer - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.with_line:
        criterion = SetCriterion(num_classes, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses, args=args, matcher=matcher)
        criterion.to(device)
        postprocessors = {'line': PostProcess_Line()}
    else:
        criterion = None
        postprocessors = None

    if args.with_dense:
        criterion_depth = SilogLoss(variance_focus=args.variance_focus, log_depth_error=args.log_depth_error)
        criterion_depth.to(device)
        criterion_seg = SegLoss()
        criterion_seg.to(device)
    else:
        criterion_depth = None
        criterion_seg = None
    
    if args.with_plane_norm_loss:
        # def __init__(self, num_ref, line_score_thresh=0.6, min_plane_area=100):
        criterion_plane = PlaneLoss(28, line_score_thresh=0.6, min_plane_area=100)
    else:
        criterion_plane = None

    return model, [criterion, criterion_depth, criterion_seg, criterion_plane], postprocessors
