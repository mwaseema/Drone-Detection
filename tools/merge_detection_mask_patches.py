import argparse
import json
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from tools.crf.crf_on_labels import convert_segmented_area_to_bounding_box


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Merge detections for volume patches with the original stabilized mask")
    parser.add_argument('--detection_patches_folder', type=str, required=True,
                        help="Folder containing detection patche masks")
    parser.add_argument('--volume_patch_info_folder', type=str, required=True,
                        help="Folder containing text files with information from where the patch was extracted")
    parser.add_argument('--stabilized_masks_folder', type=str, required=True, help="Folder containing stabilized masks")
    # parser.add_argument('--inverse_matrices_folder', type=str, required=True,
    #                     help="Folder containing inverse transformation matrices")
    parser.add_argument('--output_mask_folder', type=str, required=True, help="Folder where to save final masks")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    detection_patches_folder = args.detection_patches_folder
    volume_patch_info_folder = args.volume_patch_info_folder
    stabilized_masks_folder = args.stabilized_masks_folder
    # inverse_matrices_folder = args.inverse_matrices_folder
    output_mask_folder = args.output_mask_folder

    os.makedirs(output_mask_folder, exist_ok=True)

    detection_patch_paths = glob(os.path.join(detection_patches_folder, '*'))
    for detection_patch_path in tqdm(detection_patch_paths):
        filename = os.path.basename(detection_patch_path)
        filename_wo_ext = os.path.splitext(filename)[0]

        volume_path_info_path = os.path.join(volume_patch_info_folder, f'{filename_wo_ext}.txt')
        with open(volume_path_info_path, 'r') as f:
            volume_path_info_data = json.load(f)
        vpi_filename = volume_path_info_data['filename']
        vpi_filename_wo_ext, _ = os.path.splitext(vpi_filename)
        vpi_y1 = volume_path_info_data['box']['y1']
        vpi_x1 = volume_path_info_data['box']['x1']
        vpi_y2 = volume_path_info_data['box']['y2']
        vpi_x2 = volume_path_info_data['box']['x2']
        vpi_w = vpi_x2 - vpi_x1
        vpi_h = vpi_y2 - vpi_y1

        output_mask_path = os.path.join(output_mask_folder, vpi_filename_wo_ext + '.png')
        if os.path.exists(output_mask_path):
            stabilized_mask = cv2.imread(output_mask_path)
        else:
            stabilized_masks_path = os.path.join(stabilized_masks_folder, vpi_filename_wo_ext + '.png')
            stabilized_mask = cv2.imread(stabilized_masks_path)

        detection_patch = cv2.imread(detection_patch_path)
        detection_patch = cv2.resize(detection_patch, (vpi_w, vpi_h), interpolation=cv2.INTER_NEAREST)
        detection_patch[detection_patch > 0] = 255

        ###
        # patch_full_mask = np.zeros(stabilized_mask.shape, dtype=stabilized_mask.dtype)
        # patch_full_mask[vpi_y1:vpi_y2, vpi_x1:vpi_x2] = detection_patch
        #
        # patch_full_mask_boxes = get_region_props(patch_full_mask)
        # stabilized_mask_boxes = get_region_props(stabilized_mask)
        #
        # place = False
        # if len(stabilized_mask_boxes) == 0:
        #     place = True
        # elif len(patch_full_mask_boxes) > 0 and len(stabilized_mask_boxes) > 0:
        #     distance_matrix = get_distance_matrix(patch_full_mask_boxes, stabilized_mask_boxes)
        #     for dm in distance_matrix:
        #         min_distance = float(np.min(dm))
        #         if min_distance < 30:
        #             place = True
        #             break
        # if place:
        #     stabilized_mask[vpi_y1:vpi_y2, vpi_x1:vpi_x2] = detection_patch
        # else:
        #     stabilized_mask[vpi_y1:vpi_y2, vpi_x1:vpi_x2] = 0
        ###

        # if there is no foreground in the i3d detection, remove the box from 2d detection as well
        # otherwise keep it intact
        if np.max(detection_patch) == 0:
            stabilized_mask[vpi_y1:vpi_y2, vpi_x1:vpi_x2] = 0
        else:
            stabilized_mask[vpi_y1:vpi_y2, vpi_x1:vpi_x2] = detection_patch

        # inverse_matrix_path = os.path.join(inverse_matrices_folder, os.path.splitext(vpi_filename)[0] + '.npy')
        # inverse_matrix = np.load(inverse_matrix_path)
        # stabilized_mask = apply_inverse_transformations(stabilized_mask, inverse_matrix, stabilized_mask.shape[0],
        #                                                 stabilized_mask.shape[1])
        stabilized_mask[stabilized_mask > 0] = 255
        stabilized_mask = convert_segmented_area_to_bounding_box(stabilized_mask)

        cv2.imwrite(output_mask_path, stabilized_mask)


if __name__ == '__main__':
    main()
