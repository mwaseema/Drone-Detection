import argparse
import json
import os
from glob import glob
from os import path
from typing import Union, Tuple

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
from tqdm import tqdm

from tools.crf.IoU_test import get_bounding_boxes, get_region_props

patch_extra_pixels = 50
label_extra_pixels = 5


def get_args():
    parser = argparse.ArgumentParser(description="Perform CRF on images using labels")
    parser.add_argument('--frames_folder', type=str, required=True, help="Folder containing frames")
    parser.add_argument('--labels_mask_folder', type=str, required=True, help="Folder containing label binary masks")
    parser.add_argument('--output_folder', type=str, required=True, help="Folder where to output CRFed binary masks")
    parser.add_argument('--save_boxes_as_json', type=int, default=0, choices=[0, 1],
                        help="Whether to save boxes as json files or not")
    args = parser.parse_args()
    return args


def calculate_crf_on_labels(frame_path: Union[str, np.ndarray], detection_path: Union[str, np.ndarray]) -> np.ndarray:
    # if path of images is given
    if isinstance(frame_path, str):
        img = cv2.imread(frame_path)
    else:
        img = frame_path

    # if probability path is given
    if isinstance(detection_path, str):
        anno_rgb = cv2.imread(detection_path)
    else:
        anno_rgb = detection_path

    # convert to grayscale if it has 3 channels
    if len(anno_rgb.shape) == 3 and anno_rgb.shape[2] == 3:
        anno_rgb = cv2.cvtColor(anno_rgb, cv2.COLOR_BGR2GRAY)

    # expanding labels a little bit to accommodate any missing area of the object
    anno_rgb = expand_labels(anno_rgb, label_extra_pixels)

    anno_rgb[anno_rgb > 0] = 1
    anno_rgb = anno_rgb.astype(np.uint32)

    n_labels = 2

    labels = np.zeros((n_labels, img.shape[0], img.shape[1]), dtype=np.uint8)
    labels[0, :, :] = 1 - anno_rgb
    labels[1, :, :] = anno_rgb

    colors = [0, 255]
    colorize = np.empty((len(colors), 1), np.uint8)
    colorize[:, 0] = colors

    crf = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

    U = unary_from_labels(labels[1], n_labels, gt_prob=0.7, zero_unsure=False)
    crf.setUnaryEnergy(U)

    # feats = create_pairwise_gaussian(sdims=(2, 2), shape=img.shape[:2])
    # crf.addPairwiseEnergy(feats, compat=10,
    #                       kernel=dcrf.FULL_KERNEL,
    #                       normalization=dcrf.NORMALIZE_SYMMETRIC)

    # try different sdims 64*64 or 100*100 or larger on one single image and check which pramter gives better results
    feats = create_pairwise_bilateral(sdims=(10, 10), schan=(13, 13, 13),
                                      img=img, chdim=2)

    crf.addPairwiseEnergy(feats, compat=10,
                          kernel=dcrf.FULL_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = crf.inference(20)

    MAP = np.argmax(Q, axis=0)
    MAP = colorize[MAP]
    output_image = MAP.reshape(anno_rgb.shape)

    return output_image


def convert_segmented_area_to_bounding_box(segmented_mask: np.ndarray) -> np.ndarray:
    bounding_boxes = get_bounding_boxes(segmented_mask)
    new_segmented_mask = np.zeros(shape=segmented_mask.shape, dtype=np.uint8)

    for bounding_box in bounding_boxes:
        x1 = bounding_box['x1']
        x2 = bounding_box['x2']
        y1 = bounding_box['y1']
        y2 = bounding_box['y2']

        new_segmented_mask[y1:y2, x1:x2] = 255

    return new_segmented_mask


def get_safe_extended_pixels(height, width, extra_pixels, y1, x1, y2, x2) -> Tuple[int, int, int, int]:
    x1 -= extra_pixels
    x1 = x1 if x1 > 0 else 0

    y1 -= extra_pixels
    y1 = y1 if y1 > 0 else 0

    x2 += extra_pixels
    x2 = x2 if x2 < width else width

    y2 += extra_pixels
    y2 = y2 if y2 < height else height

    return y1, x1, y2, x2


def expand_labels(label_mask: np.ndarray, label_expand_pixels):
    boxes = get_bounding_boxes(label_mask)

    new_label_mask = np.zeros(label_mask.shape, dtype=np.uint8)
    for box in boxes:
        y1 = box['y1']
        x1 = box['x1']
        y2 = box['y2']
        x2 = box['x2']

        y1, x1, y2, x2 = get_safe_extended_pixels(label_mask.shape[0], label_mask.shape[1], label_expand_pixels, y1, x1,
                                                  y2, x2)

        new_label_mask[y1:y2, x1:x2] = 255
    return new_label_mask


def main():
    args = get_args()
    frames_folder = args.frames_folder
    labels_mask_folder = args.labels_mask_folder
    output_folder = args.output_folder
    save_boxes_as_json = args.save_boxes_as_json

    output_json_folder = output_folder + '_json'

    os.makedirs(output_folder, exist_ok=True)

    if save_boxes_as_json == 1:
        os.makedirs(output_json_folder, exist_ok=True)

    frame_paths = glob(path.join(frames_folder, '*.png'))
    frame_paths.sort()

    for frame_path in tqdm(frame_paths):
        filename = path.basename(frame_path)
        filename_wo_ext, _ = path.splitext(filename)

        labels_mask_path = path.join(labels_mask_folder, filename)

        frame = cv2.imread(frame_path)

        labels_mask = cv2.imread(labels_mask_path)
        labels_mask = cv2.cvtColor(labels_mask, cv2.COLOR_BGR2GRAY)

        crf_mask = np.zeros(labels_mask.shape, dtype=np.uint8)
        crf_boxes = []

        region_props = get_region_props(labels_mask)
        for region_prop in region_props:
            __labels_mask = np.zeros(labels_mask.shape, labels_mask.dtype)

            y1, x1, y2, x2 = region_prop.bbox

            __labels_mask[y1:y2, x1:x2] = 255

            y1, x1, y2, x2 = get_safe_extended_pixels(labels_mask.shape[0], labels_mask.shape[1], patch_extra_pixels,
                                                      y1, x1, y2, x2)

            crf_mask_patch = calculate_crf_on_labels(frame[y1:y2, x1:x2], __labels_mask[y1:y2, x1:x2])

            temp_mask = np.zeros(labels_mask.shape, dtype=np.uint8)
            temp_mask[y1:y2, x1:x2] = crf_mask_patch

            # extending foreground mask before saving
            temp_mask = expand_labels(temp_mask, 3)

            # to make sure only foreground pixels are copied and not the background one
            crf_mask[temp_mask > 0] = temp_mask[temp_mask > 0]

            # for saving as json file
            temp_rps = get_region_props(temp_mask)
            for temp_rp in temp_rps:
                temp_y1, temp_x1, temp_y2, temp_x2 = temp_rp.bbox
                crf_boxes.append([temp_y1, temp_x1, temp_y2, temp_x2])

        crf_mask = convert_segmented_area_to_bounding_box(crf_mask)

        output_file_path = path.join(output_folder, filename)
        cv2.imwrite(output_file_path, crf_mask)

        if save_boxes_as_json == 1:
            output_json_file = path.join(output_json_folder, f'{filename_wo_ext}.json')
            with open(output_json_file, 'w') as f:
                json.dump(crf_boxes, f)


if __name__ == '__main__':
    main()
