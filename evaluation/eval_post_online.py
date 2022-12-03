import glob
import imp
import math
import os
import os.path as osp
import sys
from turtle import color

# import cv2
# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

def pline(x1, y1, x2, y2, x, y):
    px = x2 - x1
    py = y2 - y1
    dd = px * px + py * py
    u = ((x - x1) * px + (y - y1) * py) / max(1e-9, float(dd))
    dx = x1 + u * px - x
    dy = y1 + u * py - y
    return dx * dx + dy * dy


def psegment(x1, y1, x2, y2, x, y):
    px = x2 - x1
    py = y2 - y1
    dd = px * px + py * py
    u = max(min(((x - x1) * px + (y - y1) * py) / float(dd), 1), 0)
    dx = x1 + u * px - x
    dy = y1 + u * py - y
    return dx * dx + dy * dy


def plambda(x1, y1, x2, y2, x, y):
    px = x2 - x1
    py = y2 - y1
    dd = px * px + py * py
    return ((x - x1) * px + (y - y1) * py) / max(1e-9, float(dd))


def postprocess(lines, scores, threshold=0.01, tol=1e9, do_clip=False):
    nlines, nscores, choosen_ids = [], [], []
    for id, ((p, q), score) in enumerate(zip(lines, scores)):
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
        choosen_ids.append(id)
    return np.array(nlines), np.array(nscores), np.array(choosen_ids)

PLTOPTS = {"color": "#33FFFF", "s": 1.2, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.92, vmax=1.02)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

def c(x):
    return sm.to_rgba(x)


def imshow(im):
    plt.close()
    sizes = im.shape
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, sizes[1] - 0.5])
    plt.ylim([sizes[0] - 0.5, -0.5])
    plt.imshow(im)
  
def vis_pred_lines(pred_lines, pred_scores, img_mat, gt_lines, fname, save_dir):
    thresholds = [0.010, 0.015]
    print("Processing", fname)
    im = img_mat
    lines = pred_lines
    scores = pred_scores
    gtlines = gt_lines
    gtlines[:, :, 0] *= im.shape[0]
    gtlines[:, :, 1] *= im.shape[1]
    for i in range(1, len(lines)):
        if (lines[i] == lines[0]).all():
            lines = lines[:i]
            scores = scores[:i]
            break

    lines[:, :, 0] *= im.shape[0]
    lines[:, :, 1] *= im.shape[1]
    diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
    lines_glass = lines[:, :2]
    lines_center = lines[:, 2]

    gtlines_glass = gtlines[:, :2]
    gtlines_center = gtlines[:, 2]
    for threshold in thresholds:
        nlines, nscores, choosen_ids = postprocess(lines_glass, scores, diag * threshold, 0, False)

        outdir = osp.join(save_dir, f"{threshold:.3f}".replace(".", "_"))
        os.makedirs(outdir, exist_ok=True)
        npz_name = osp.join(outdir, osp.split(fname)[-1])
        
        # plot gt
        imshow(im)
        for (a, b), cen in zip(gtlines_glass, gtlines_center):
            plt.plot([a[1], b[1]], [a[0], b[0]], c="orange", linewidth=0.5)
            plt.scatter(a[1], a[0], **PLTOPTS)
            plt.scatter(b[1], b[0], **PLTOPTS)
            plt.plot([cen[1]], [cen[0]], marker='o',  markersize=2)

        plt.savefig(npz_name+"_gt.png", dpi=500, bbox_inches=0)

        pred_lines_center = lines_center[choosen_ids]

        thres = [0.96, 0.97, 0.98, 0.99]
        for i, t in enumerate(thres):
            imshow(im)
            for (a, b), s, cen in zip(nlines[nscores > t], nscores[nscores > t], pred_lines_center[nscores > t]):
                plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=0.5)
                plt.scatter(a[1], a[0], **PLTOPTS)
                plt.scatter(b[1], b[0], **PLTOPTS)

                plt.plot([cen[1]], [cen[0]], marker='o',  markersize=2, c=c(s))

            plt.savefig(
                npz_name+"_pred_{}.png".format(str(i)), dpi=500, bbox_inches=0
            )

        nlines[:, :, 0] *= 128 / im.shape[0]
        nlines[:, :, 1] *= 128 / im.shape[1]
        np.savez_compressed(npz_name, lines=nlines, score=nscores)
