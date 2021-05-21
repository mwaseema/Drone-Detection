import os
from glob import glob
from math import sqrt
from shutil import copyfile
from typing import List

import cv2 as cv
import numpy as np
import skimage.measure
from tqdm import tqdm

# folder containing epochs folder which contain detections folder
from tools.crf.IoU_test import get_region_props
from test_model.spatial.config import evaluation_script

# if output folder by network test code ~/output_folder/EPOCH_NUMBER/network_detections
# then here ~/output_folder (absolute folder path)
network_detections_outer_folder = ''
# uncomment corresponding line if using fixed sized cuboids. As they need to be transformed back for scores computation
# inverse_matrices_folder = ''
# folder containing patch information json files (absolute folder path)
volume_patch_information_folder = ''
# detections obtained from 2D network (absolute folder path)
detection_masks_2d_folder = ''
# folder containing ground truth binary masks (absolute folder path)
ground_truths_folder = ''
# folder containing files with scores of 2d detections (absolute folder path)
box_scores_2d_folder = ''
# file containing scores of 3d cuboid detection (absolute folder path)
box_scores_3d_file_path = ''
# folder containing cuboid patch information json file (absolute folder path)
cuboid_patch_information_folder = ''

# set this to folder containing stabilized masks if they need to be transformed back.
# If don't want to transform them back for score calculation then following is required.
stabilized_masks_folder = detection_masks_2d_folder


def remove_boxes_less_than_threshold(masks_folder: str, threshold=50):
    mask_paths = glob(os.path.join(masks_folder, '*'))
    for mask_path in tqdm(mask_paths, desc="Removing small boxes"):
        mask = cv.imread(mask_path)
        rps = get_region_props(mask)
        new_mask = np.zeros(mask.shape[0:2], dtype=mask.dtype)
        for rp in rps:
            y1, x1, y2, x2 = rp.bbox
            if rp.bbox_area >= threshold:
                new_mask[y1:y2, x1:x2] = 255
        cv.imwrite(mask_path, new_mask)


def calculate_distance(y1, x1, y2, x2):
    val = pow(x2 - x1, 2) + pow(y2 - y1, 2)
    val = sqrt(val)
    return val


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


def __add_missing_boxes_to_mask2(mask1: np.ndarray, mask2: np.ndarray, distance_threshold=10):
    mask1 = mask1.copy()
    mask2 = mask2.copy()

    mask1_rps = get_region_props(mask1)
    mask2_rps = get_region_props(mask2)

    missing_boxes = []

    # if there are boxes in mask 1 but there is no box in mask 2
    # add all the boxes of mask 1 to mask 2
    if len(mask1_rps) > 0 and len(mask2_rps) == 0:
        missing_boxes.extend(mask1_rps)
    # if there are some boxes in mask 1 and mask 2
    # If these are not present in mask2 these will be copied to that
    elif len(mask1_rps) > 0 and len(mask2_rps) > 0:
        distance_matrix = get_distance_matrix(mask1_rps, mask2_rps)
        for i, dm in enumerate(distance_matrix):
            min_distance = np.min(dm)
            # if current box of mask 1 doesn't exist in mask 2
            # add this mask in mask 2
            if min_distance > distance_threshold:
                missing_boxes.append(mask1_rps[i])

    for missing_box in missing_boxes:
        foreground_number = int(np.max(mask1))

        y1, x1, y2, x2 = missing_box.bbox
        mask2[y1:y2, x1:x2] = foreground_number

    return mask2


def add_missing_boxes_to_mask2(mask1_folder: str, mask2_folder: str, output_folder: str):
    mask1_paths = glob(os.path.join(mask1_folder, '*'))
    for mask1_path in tqdm(mask1_paths, desc="Adding missing boxes to mask 2"):
        filename = os.path.basename(mask1_path)
        mask2_path = os.path.join(mask2_folder, filename)

        mask1 = cv.imread(mask1_path)
        mask2 = cv.imread(mask2_path)

        new_mask = __add_missing_boxes_to_mask2(mask1, mask2, 30)

        output_path = os.path.join(output_folder, filename)
        cv.imwrite(output_path, new_mask)


def region_prop_masks(masks_folder: str):
    mask_paths = glob(os.path.join(masks_folder, '*'))
    for mask_path in tqdm(mask_paths, desc="Region proping masks"):
        mask = cv.imread(mask_path)
        rps = get_region_props(mask)
        new_mask = np.zeros(mask.shape, dtype=mask.dtype)
        foreground_pixel = np.max(mask)
        for rp in rps:
            y1, x1, y2, x2 = rp.bbox
            new_mask[y1:y2, x1:x2] = foreground_pixel

        cv.imwrite(mask_path, new_mask)


def main():
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    scores_list = []

    epoch_folders = glob(os.path.join(network_detections_outer_folder, '*'))
    epoch_folders.sort(key=lambda x: int(os.path.basename(x)), reverse=True)
    for epoch_folder in tqdm(epoch_folders, desc='Processing epochs folder'):
        print()

        network_detections_folder = os.path.join(epoch_folder, 'network_detections')
        merged_detections = os.path.join(epoch_folder, 'merged_with_stabilized_masks')

        os.makedirs(merged_detections, exist_ok=True)

        # merge with stabilized frames and transform them back to original form
        command = f'python {os.path.join(BASE_PATH, "tools/merge_detection_mask_patches.py")}'
        command += f' --detection_patches_folder "{network_detections_folder}"'
        command += f' --volume_patch_info_folder "{volume_patch_information_folder}"'
        command += f' --stabilized_masks_folder "{stabilized_masks_folder}"'
        # uncomment if required
        # command += f' --inverse_matrices_folder "{inverse_matrices_folder}"'
        command += f' --output_mask_folder "{merged_detections}"'
        os.system(command)
        print()

        # copy missing files from 2d detection masks to the reverse transformed i3d masks folder
        detection_mask_2d_paths = glob(os.path.join(detection_masks_2d_folder, '*'))
        for detection_mask_2d_path in tqdm(detection_mask_2d_paths, desc='Copying missing masks'):
            detection_mask_2d_filename = os.path.basename(detection_mask_2d_path)
            detection_mask_2d_output_path = os.path.join(merged_detections, detection_mask_2d_filename)
            if not os.path.exists(detection_mask_2d_output_path):
                copyfile(detection_mask_2d_path, detection_mask_2d_output_path)
        print()

        detections_masks_removed_fp_folder = os.path.join(epoch_folder, 'detection_masks_FP_removed')
        os.makedirs(detections_masks_removed_fp_folder, exist_ok=True)

        # remove boxes from 2D detections which aren't available in i3d detections
        command = f'python {os.path.join(BASE_PATH, "tools/remove_boxes_not_available_in_reference_masks.py")}'
        command += f' --masks_folder "{detection_masks_2d_folder}"'
        command += f' --reference_masks_folder "{merged_detections}"'
        command += f' --output_masks_folder "{detections_masks_removed_fp_folder}"'
        command += f' --distance_threshold 30'
        os.system(command)
        print()

        # add missing boxes
        detections_missing_boxes_added = os.path.join(epoch_folder, 'detection_missing_boxes_added')
        os.makedirs(detections_missing_boxes_added, exist_ok=True)
        add_missing_boxes_to_mask2(merged_detections, detections_masks_removed_fp_folder,
                                   detections_missing_boxes_added)

        # region prop
        region_prop_masks(detections_missing_boxes_added)

        # remove boxes less than threshold
        remove_boxes_less_than_threshold(detections_missing_boxes_added, 50)

        # merge detection scores
        merged_2d_3d_scores_folder = os.path.join(epoch_folder, 'merged_box_scores')
        command = f'python {os.path.join(BASE_PATH, "tools/add_3d_box_information_to_2d_box_information.py")}'
        command += f' --detection_masks_3d_folder "{detections_missing_boxes_added}"'
        command += f' --detection_masks_2d_folder "{detection_masks_2d_folder}"'
        command += f' --box_scores_2d_folder "{box_scores_2d_folder}"'
        command += f' --cuboid_patch_information_from_3d_folder "{cuboid_patch_information_folder}"'
        command += f' --scores_of_3d_file_path "{box_scores_3d_file_path}"'
        command += f' --box_scores_output_folder "{merged_2d_3d_scores_folder}"'
        os.system(command)
        print()

        annotation_detection_source_folder = detections_missing_boxes_added
        annotation_detection_output_folder = os.path.join(epoch_folder, 'metric', 'detections')
        annotation_gt_source_folder = ground_truths_folder
        annotation_gt_output_folder = os.path.join(epoch_folder, 'metric', 'gt')

        # generate annotation files
        command = f'python {os.path.join(BASE_PATH, "tools/generate_annotation_files.py")}'
        command += f' --detections_folder "{annotation_detection_source_folder}"'
        command += f' --detections_output_folder "{annotation_detection_output_folder}"'
        command += f' --ground_truth_folder "{annotation_gt_source_folder}"'
        command += f' --ground_truth_output_folder "{annotation_gt_output_folder}"'
        command += f' --score_boxes_folder "{merged_2d_3d_scores_folder}"'
        os.system(command)
        print()

        command = f'python "{evaluation_script}"'
        command += f' --detection_folder "{annotation_detection_output_folder}"'
        command += f' --ground_truth_folder "{annotation_gt_output_folder}"'
        command += f' --iou_threshold 0.5'
        os.system(command)
        print()

        with open(os.path.join(epoch_folder, 'metric', 'evaluation_scores.json'), 'r') as f:
            scores_list.append({
                'folder_name': os.path.basename(epoch_folder),
                'content': f.read(),
            })

    with open(os.path.join(network_detections_outer_folder, 'scores.txt'), 'w') as f:
        for scr in scores_list:
            f.write(scr['folder_name'] + '\n' + scr['content'] + '\n\n\n')


if __name__ == "__main__":
    main()
