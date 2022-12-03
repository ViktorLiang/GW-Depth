import json
import numpy as np
import cv2
from shapely.geometry import Polygon, mapping

# colour map
COLORS = [(0,0,0)
        ,(255,0,0),(0,255,0),(255,128,128),(0,0,255),(128,0,128)
        ,(200,150,10),(128,128,255),(64,0,255),(192,192,0),(200,200,64)
        ,(255,150,0),(64,0,255),(192,0,192),(0,128,128),(192,128,128)
        ,(0,64,192),(128,192,0),(0,192,255),(128,255,0),(64,64,255)]


def gen_pairs(np_vector):
    d0 = np_vector[:, np.newaxis]
    d1 = np_vector[1:].tolist()
    d1.append(np_vector[0])
    d1 = np.array(d1)[:, np.newaxis]
    d_pairs = np.concatenate((d0, d1), axis=1)
    return d_pairs

def read_json_label(json_label, key=None):
    with open(json_label) as f:
        annos = json.load(f)
        if key is not None:
            assert key in annos, 'key {} not exists in keys:{}'.format(key, str(annos.keys()))
            return annos[key]
        else:
            return annos

def cross_value_2D(s, e, c):
    s_3 = np.zeros((1, 3))
    s_3[:, 0:2] = s
    e_3 = np.zeros((1, 3))
    e_3[:, 0:2] = e
    c_3 = np.zeros((1, 3))
    c_3[:, 0:2] = c
    vec_se = e_3 - s_3
    vec_sc = c_3 - s_3
    return np.cross(vec_se, vec_sc)

def within_poly(points, check_points):
    pnts_pairs = gen_pairs(points)
    is_in = []
    within_info = []
    for c in check_points:
        # assert (isinstance(c, list)) and len(c) == 2, 'check point {} must be 2D value'.format(str(c))
        lines_crs = []
        for ps in pnts_pairs:
            s = ps[0]
            e = ps[1]
            crs = cross_value_2D(s, e, c)
            lines_crs.append(crs[0][-1])
        # print('lines_crs', lines_crs)
        if np.sum(np.array(lines_crs) < 0) == len(lines_crs) or np.sum(np.array(lines_crs) > 0) == len(lines_crs):
            in_poly = True
        else:
            in_poly = False
        is_in.append(in_poly)
        within_info.append(lines_crs)
    return is_in, within_info

def read_depth_npy(depth_file, allow_pickle=False):
    with open(depth_file, 'rb') as f:
        dpth_mat = np.load(f, allow_pickle=allow_pickle)
        return dpth_mat

def vis_depth_mat(dpth_mat, height=None, width=None, show=False, raw_depth_file=None, save_file=None):
    # min, max, min_loc, max_loc = cv2.minMaxLoc(dpth_mat)
    if height is None or width is None:
        height, width = dpth_mat.shape[-2:]
    # dpth_arr = np.zeros((height, width), dtype=np.uint8)
    # max = 6000
    # cv2.convertScaleAbs(dpth_mat, dpth_arr, 255/max)
    # dpth_color = cv2.applyColorMap(dpth_arr, cv2.COLORMAP_JET)

    dpth_color = cv2.applyColorMap(
            cv2.convertScaleAbs(dpth_mat, alpha=0.03), cv2.COLORMAP_JET)

    if show:
        # plt.title('completed_depth')
        # plt.imshow(dpth_color)
        # if save_file is not None:
        #     save_new_name = save_file.split('.')[0] + '.png'
        #     plt.savefig(save_new_name)
        #     print('saved to '+save_new_name)
        # plt.show()

        cv2.imshow("dpth_arr", dpth_color)
        if cv2.waitKeyEx() == 27:
            cv2.destroyAllWindows()
    else:
        return dpth_color

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
    assert pred_labels.size >= 2
    if pred_labels.size == 3:
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

def line_intersection(line1, line2, within_line1_segment=True):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    # if within_line1_segment:
    #     scol = line1[0][0]
    #     ecol = line1[1][0]
    #     if (x < scol and x < ecol) or (x > scol and x > ecol):
    #         print('lines interscet do not in segment of line1')
    #         return -1, -1
    return x, y

def point_side_of_line(line1, point):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x, y = point
    d = (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)
    return d

def draw_polys(poly_points, width=0, height=0, mat=None, color=(0, 255, 0), thickness=4, show=False):
    if mat is None:
        palette = np.ones((height, width, 3), dtype=np.uint8) * 255
    else:
        palette = mat
    for i, p in enumerate(poly_points):
        cv2.line(palette, (int(p[0][0]),int(p[0][1])), (int(p[1][0]),int(p[1][1])), color, thickness=thickness)
    if show:
        cv2.imshow("lin map", palette)
        if cv2.waitKeyEx() == 27:
            cv2.destroyAllWindows()
    return palette

def draw_points(points, width=0, height=0, mat=None, color=(0, 0, 255), show=False):
    if mat is None:
        palette = np.ones((height, width, 3), dtype=np.uint8) * 255
    else:
        palette = mat
    for i, p in enumerate(points):
        cv2.circle(palette, (int(p[0]),int(p[1])), radius=10, color=color, thickness=-1)

    if show:
        cv2.imshow("points map", palette)
        if cv2.waitKeyEx() == 27:
            cv2.destroyAllWindows()
    return palette


def intersect_remap(main_coors, poly_coors):
    p1 = Polygon(main_coors)
    p2 = Polygon(poly_coors)
    new_coors = p1.intersection(p2)
    new_coors_mapping = mapping(new_coors)

    if new_coors_mapping['type'] == 'Point' and len(new_coors_mapping['coordinates']) == 2:
        print('new_coors_mapping is point:', new_coors_mapping)
        return []

    assert new_coors_mapping['type'] in ['Polygon', 'GeometryCollection'], 'new_coors_mapping:'+str(new_coors_mapping)
    if new_coors_mapping['type'] == 'Polygon':
        coors_npy = np.array(new_coors_mapping['coordinates'])
    else:
        coors_list = []
        for geo in new_coors_mapping['geometries']:
            if geo['type'] == 'Polygon':
                coors_list.append(geo['coordinates'])
        coors_npy = np.array(coors_list)
    coors_npy = coors_npy.squeeze()
    
    if coors_npy.size <= 2:
        return []
    new_points = coors_npy.tolist()

    # new_points = list(new_coors.exterior.coords)
    # return new_points

    lt_coor = main_coors[0]
    rb_coor = main_coors[2]
    new_coors = []
    for pnt in new_points:
        # each line have two point
        col_no = max(pnt[0], lt_coor[0]) #column number
        row_no = max(pnt[1], lt_coor[1]) #row number
        col_no = min(col_no, rb_coor[0]) #column number
        row_no = min(row_no, rb_coor[1]) #row number
        col_no = col_no - lt_coor[0]
        row_no = row_no - lt_coor[1]
        new_coors.append([col_no, row_no])
    return new_coors

def clamp_lines(left_top_coor, right_bottom_coor, raw_lines):
    lt_coor = left_top_coor
    rb_coor = right_bottom_coor
    width = rb_coor[0] - lt_coor[0]
    height = rb_coor[1] - lt_coor[1]
    lines = raw_lines
    j = lt_coor[0]
    i = lt_coor[1]
    cropped_lines = lines - np.array([j, i, j, i])
    
    eps = 1e-12

    # In dataset, we assume the left point has smaller x coord
    remove_x_min = np.logical_and(cropped_lines[:, 0] < 0, cropped_lines[:, 2] < 0)
    remove_x_max = np.logical_and(cropped_lines[:, 0] > width, cropped_lines[:, 2] > width)
    remove_x = np.logical_or(remove_x_min, remove_x_max)
    keep_x = ~remove_x

    # there is no assumption on y, so remove lines that have both y coord out of bound
    remove_y_min = np.logical_and(cropped_lines[:, 1] < 0, cropped_lines[:, 3] < 0)
    remove_y_max = np.logical_and(cropped_lines[:, 1] > height, cropped_lines[:, 3] > height)
    remove_y = np.logical_or(remove_y_min, remove_y_max)
    keep_y = ~remove_y

    keep = np.logical_and(keep_x, keep_y)
    cropped_lines = cropped_lines[keep]
    clamped_lines = np.zeros_like(cropped_lines)

    for i, line in enumerate(cropped_lines):
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1 + eps)
        if x1 < 0:
            x1 = 0
            y1 = y2 + (x1 - x2) * slope
        if y1 < 0:
            y1 = 0
            x1 = x2 - (y2 - y1) / slope
        if x2 > width:
            x2 = width
            y2 = y1 + (x2 - x1) * slope
        if y2 > height:
            y2 = height
            x2 = x1 + (y2 - y1) / slope
        
        #
        if x2 < 0:
            x2 = 0
            y2 = y1 + (x2 - x1) * slope
        if y2 < 0:
            y2 = 0
            x2 = x1 - (y1 - y2) / slope
        if x1 > width:
            x1 = width
            y1 = y2 + (x1 - x2) * slope
        if y1 > height:
            y1 = height
            x1 = x2 + (y1 - y2) / slope

        clamped_lines[i, :] = np.array([x1, y1, x2, y2])

    # clamped_lines[:, 0::2].clamp_(min=0, max=width)
    # clamped_lines[:, 1::2].clamp_(min=0, max=height)
    clamped_lines[:, 0::2] = np.clip(clamped_lines[:, 0::2], 0, width)
    clamped_lines[:, 1::2] = np.clip(clamped_lines[:, 1::2], 0, height)

    return clamped_lines