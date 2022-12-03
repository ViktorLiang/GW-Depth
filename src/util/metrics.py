import numpy as np
import cv2
import os
import json
from collections import OrderedDict
import argparse
from PIL import Image as PILImage
#from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support

GLASS_LABELS = ['Background', 'Glass']
labels_name_dicts = {"GLASS": GLASS_LABELS}

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def compute_mean_ioU(preds, gts, num_classes, dataset_name='GLASS'):
    assert len(preds) == len(gts), (len(preds),len(gts))

    confusion_matrix = np.zeros((num_classes, num_classes))

    for i in range(len(preds)):
        gt = gts[i]
        pred = preds[i]

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)

    name_value = []

    for i, (label, iou) in enumerate(zip(labels_name_dicts[dataset_name], IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value

def np_hard_max(np_array):
    for cat_id in range(np_array.shape[0]):
        cat_sum = np.sum(np_array[cat_id], dtype=np.float64)
        if cat_sum <= 0:
            cat_sum = 1
        cat_norm = np_array[cat_id]/cat_sum
        np_array[cat_id] = cat_norm
    return np_array


def count_pred_cat_num(top_box, gt, pred, num_classes):
    """
    calculate the top predication category
    :param pred_top_box:total top category
    :param gt:
    :param pred:
    :return:
    """
    for cat in range(top_box.shape[0]):
        gt_cat_idx = np.where(gt == cat)
        pred_cats = pred[gt_cat_idx]
        cat_count = np.bincount(pred_cats)
        cat_count = np.pad(cat_count, (0, num_classes - len(cat_count)), 'constant', constant_values=0)
        cat_count = cat_count.astype(np.float64)
        top_box[cat] += cat_count
    return top_box

def compute_mean_ioU_file(preds_dir, num_classes, datadir, dataset='val', dataset_name='CIHP'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]

    confusion_matrix = np.zeros((num_classes, num_classes))

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset + '_segmentations', im_name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        pred_path = os.path.join(preds_dir, im_name + '.png')
        pred = np.asarray(PILImage.open(pred_path))

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum())*100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean())*100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array*100
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    for i, (label, iou) in enumerate(zip(labels_name_dicts[dataset_name], IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value

def write_results(preds, scales, centers, datadir, dataset, result_dir, input_size=[473, 473]):
    palette = get_palette(20)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    json_file = os.path.join(datadir, 'annotations', dataset + '.json')
    with open(json_file) as data_file:
        data_list = json.load(data_file)
        data_list = data_list['root']
    for item, pred_out, s, c in zip(data_list, preds, scales, centers):
        im_name = item['im_name']
        w = item['img_width']
        h = item['img_height']
        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        #pred = pred_out
        save_path = os.path.join(result_dir, im_name[:-4]+'.png')

        output_im = PILImage.fromarray(np.asarray(pred, dtype=np.uint8))
        output_im.putpalette(palette)
        output_im.save(save_path)

# depth metrics
eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
def compute_depth_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV NetworkEv")
    parser.add_argument("--pred-path", type=str, default='',
                        help="Path to predicted segmentation.")
    parser.add_argument("--gt-path", type=str, default='',
                        help="Path to the groundtruth dir.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    palette = get_palette(20)
    # im_path = '/ssd1/liuting14/Dataset/LIP/val_segmentations/100034_483681.png'
    # #compute_mean_ioU_file(args.pred_path, 20, args.gt_path, 'val')
    # im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    # print(im.shape)
    # test = np.asarray( PILImage.open(im_path))
    # print(test.shape)
    # if im.all()!=test.all():
    #     print('different')
    # output_im = PILImage.fromarray(np.zeros((100,100), dtype=np.uint8))
    # output_im.putpalette(palette)
    # output_im.save('test.png')
    pred_dir = '/ssd1/liuting14/exps/lip/snapshots/results/epoch4/'
    num_classes = 20
    datadir = '/ssd1/liuting14/Dataset/LIP/'
    compute_mean_ioU_file(pred_dir, num_classes, datadir, dataset='val')
