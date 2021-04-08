import argparse
import os
from glob import glob
from typing import List

import cv2
from tqdm import tqdm

from correlational_tracker.multi_tracker_with_merge_on_fly import calculate_distance
from crf.IoU_test import get_region_props
import skimage.measure
import numpy as np

from crf.crf_on_labels import get_safe_extended_pixels


def get_arguments():
    parser = argparse.ArgumentParser(description="Remove boxes which are not available in reference mask")
    parser.add_argument('--masks_folder', type=str, required=True, help="Folder containing source masks")
    parser.add_argument('--reference_masks_folder', type=str, required=True, help="Folder containing reference masks")
    parser.add_argument('--output_masks_folder', type=str, required=True,
                        help="Folder where to save masks after removing boxes from source masks")
    parser.add_argument('--distance_threshold', type=int, default=30,
                        help="Threshold value above which box is considered absent")
    args = parser.parse_args()
    return args


def get_distance_matrix(rps_1: List[skimage.measure._regionprops._RegionProperties],
                        rps_2: List[skimage.measure._regionprops._RegionProperties]):
    distance_matrix = np.zeros((len(rps_1), len(rps_2)), dtype=np.float32)

    for rp_1_index, rp_1 in enumerate(rps_1):
        y1, x1 = rp_1.centroid
        for rp_2_index, rp_2 in enumerate(rps_2):
            y2, x2 = rp_2.centroid
            distance = calculate_distance(y1, x1, y2, x2)
            distance_matrix[rp_1_index, rp_2_index] = distance

    return distance_matrix


def main():
    args = get_arguments()
    masks_folder = args.masks_folder
    reference_masks_folder = args.reference_masks_folder
    output_masks_folder = args.output_masks_folder
    distance_threshold = args.distance_threshold

    mask_paths = glob(os.path.join(masks_folder, '*'))
    mask_paths.sort()

    for mask_path in tqdm(mask_paths, desc="Processing masks"):
        filename = os.path.basename(mask_path)

        reference_mask_path = os.path.join(reference_masks_folder, filename)
        if os.path.exists(reference_mask_path):
            mask = cv2.imread(mask_path)
            mask[mask > 0] = 255

            reference_mask = cv2.imread(reference_mask_path)
            reference_mask[reference_mask > 0] = 255

            mask_region_props = get_region_props(mask)
            reference_mask_region_props = get_region_props(reference_mask)

            # if mask or reference mask have no bounding box in it
            if len(reference_mask_region_props) == 0 or len(mask_region_props) == 0:
                output_mask = reference_mask
            else:
                distance_matrix = get_distance_matrix(mask_region_props, reference_mask_region_props)

                for mask_rp_index, distance_matrix_row in enumerate(distance_matrix):
                    min_distance = float(np.min(distance_matrix_row))

                    # if minimum distance is above threshold, that means box is not present
                    if min_distance > distance_threshold:
                        mask_rp = mask_region_props[mask_rp_index]
                        y1, x1, y2, x2 = mask_rp.bbox
                        y1, x1, y2, x2 = get_safe_extended_pixels(mask.shape[0], mask.shape[1], 4, y1, x1, y2, x2)

                        mask[y1:y2, x1:x2] = 0
                output_mask = mask.copy()

            output_path = os.path.join(output_masks_folder, filename)
            cv2.imwrite(output_path, output_mask)


if __name__ == '__main__':
    main()
