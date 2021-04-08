from glob import glob
from math import exp
from os import path
from typing import List, Dict

import cv2
import numpy as np
import skimage.measure
from scipy.optimize import linear_sum_assignment

detection = ''
ground_truth = ''


def get_region_props(image: np.ndarray) -> List[skimage.measure._regionprops._RegionProperties]:
    image = image.copy()

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    label_image = skimage.measure.label(image)
    region_props = skimage.measure.regionprops(label_image)
    return region_props


def get_bounding_boxes(image: np.ndarray) -> List[Dict[str, int]]:
    props = get_region_props(image)

    coords = []
    for prop in props:
        min_row, min_col, max_row, max_col = prop.bbox
        coords.append({
            'x1': min_col,
            'x2': max_col,
            'y1': min_row,
            'y2': max_row,
        })
    return coords


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # print(iou)
    # return the intersection over union value
    return iou


def get_iou_score(y_true_reshaped, y_pred_reshaped):
    props_im = get_region_props(y_pred_reshaped)
    props_gt = get_region_props(y_true_reshaped)

    IOU_bbx_mul = np.zeros((props_gt.__len__(), props_im.__len__()))

    for g_b in range(0, props_gt.__len__()):
        for p_b in range(0, props_im.__len__()):
            IOU_bbx_mul[g_b, p_b] = bb_intersection_over_union(props_gt[g_b].bbox, props_im[p_b].bbox)

    row_ind, col_ind = linear_sum_assignment(1 - IOU_bbx_mul)

    calculated_IoU = 0
    for ir in range(0, len(row_ind)):
        IOU_bbx_s = IOU_bbx_mul[row_ind[ir], col_ind[ir]]

        calculated_IoU = IOU_bbx_s
        # because want to get for only one bounding box
        break

    return calculated_IoU


def compute_iou(y_true, y_pred):
    IoUs = []
    # iterating over batch
    for i in range(y_true.shape[0]):
        y_true_single = y_true[i, :, :].reshape(model.output_width, model.output_height, 2).argmax(axis=2)
        y_pred_single = y_pred[i, :, :].reshape(model.output_width, model.output_height, 2).argmax(axis=2)

        IoU = get_iou_score(y_true_single, y_pred_single)

        IoUs.append(IoU)

    return float(np.mean(IoUs))


def loss_from_iou(y_true, y_pred):
    average_iou = compute_iou(y_true, y_pred)

    if average_iou >= 0.8:
        loss = 0
    else:
        loss = exp(1 - average_iou)

    return float(loss)


def main_back():
    detection_files = glob(path.join(detection, '*'))
    for dt_f in detection_files:
        filename = path.basename(dt_f)

        ground_truth_file = path.join(ground_truth, filename)

        d = cv2.imread(dt_f)
        gt = cv2.imread(ground_truth_file)

        iou = get_iou_score(gt, d)
        pass


def main():
    det = '/home/hec/waseem/UAV/data/Videos/output_dataset_combined/test/ground_truth/Clip_042_000208.png'
    det = cv2.imread(det)
    gt = '/home/hec/waseem/UAV/data/Videos/output_dataset_combined/test/ground_truth/Clip_042_000208.png'
    gt = cv2.imread(gt)
    iou_value = get_iou_score(gt, det)
    pass


if __name__ == '__main__':
    main()
