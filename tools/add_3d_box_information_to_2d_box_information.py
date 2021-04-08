import argparse
import json
import os
from glob import glob

import cv2
from tqdm import tqdm

from crf.IoU_test import get_region_props, bb_intersection_over_union


def get_arguments():
    parser = argparse.ArgumentParser(description="Remove boxes which are not available in reference mask")
    parser.add_argument('--detection_masks_3d_folder', type=str, required=True, help="Folder containing 3D detections")
    parser.add_argument('--detection_masks_2d_folder', type=str, required=True, help="Folder containing 2D detections")
    parser.add_argument('--box_scores_2d_folder', type=str, required=True,
                        help="Folder containing score files for 2D detections")
    parser.add_argument('--cuboid_patch_information_from_3d_folder', type=str, required=True,
                        help="Folder containing cuboid information json files")
    parser.add_argument('--scores_of_3d_file_path', type=str, required=True,
                        help="Absolute path to file containing 3d detection scores")
    parser.add_argument('--box_scores_output_folder', type=str, required=True,
                        help="Folder where to output merged scores")
    args = parser.parse_args()
    return args


def get_corresponding_cuboid_box_filename(patch_information_folder: str, cuboids_filename_wo_ext: str,
                                          box_to_search) -> str:
    search_y1, search_x1, search_y2, search_x2 = box_to_search

    patch_information_paths = glob(os.path.join(patch_information_folder, f'{cuboids_filename_wo_ext}*'))
    for patch_information_path in patch_information_paths:
        filename = os.path.basename(patch_information_path)
        filename, _ = os.path.splitext(filename)

        with open(patch_information_path, 'r') as f:
            patch_information = json.load(f)

        y1 = patch_information['box']['y1']
        x1 = patch_information['box']['x1']
        y2 = patch_information['box']['y2']
        x2 = patch_information['box']['x2']

        search_box_tuple = (search_x1, search_y1, search_x2, search_y2)
        box_tuple = (x1, y1, x2, y2)
        iou = bb_intersection_over_union(search_box_tuple, box_tuple)

        if iou >= 0.7:
            return filename

        # if ((y1 <= search_y1 < y2) and (y1 < search_y2 <= y2)) and (x1 <= search_x1 < x2) and (x1 < search_x2 <= x2):
        #     return filename

    return ''


def get_score_from_3d_box_scores(filename, box_scores_3d) -> int:
    for bs in box_scores_3d:
        filename_wo_ext = bs['filename_wo_ext']
        score = bs['score']

        if filename == filename_wo_ext:
            return score
    return -1


def main():
    args = get_arguments()
    detection_masks_3d_folder = args.detection_masks_3d_folder
    detection_masks_2d_folder = args.detection_masks_2d_folder
    box_scores_2d_folder = args.box_scores_2d_folder
    cuboid_patch_information_from_3d_folder = args.cuboid_patch_information_from_3d_folder
    scores_of_3d_file_path = args.scores_of_3d_file_path
    box_scores_output_folder = args.box_scores_output_folder

    os.makedirs(box_scores_output_folder, exist_ok=True)

    with open(scores_of_3d_file_path, 'r') as f:
        scores_of_3d = json.load(f)

    detection_3d_paths = glob(os.path.join(detection_masks_3d_folder, '*'))

    for detection_3d_path in tqdm(detection_3d_paths):
        filename = os.path.basename(detection_3d_path)
        filename_wo_ext, _ = os.path.splitext(filename)

        box_scores_2d_path = os.path.join(box_scores_2d_folder, f'{filename_wo_ext}.json')
        with open(box_scores_2d_path) as f:
            box_scores_2d = json.load(f)

        detection_2d_path = os.path.join(detection_masks_2d_folder, filename)

        detection_3d = cv2.imread(detection_3d_path)
        detection_2d = cv2.imread(detection_2d_path)

        detection_3d_rps = get_region_props(detection_3d)
        detection_2d_rps = get_region_props(detection_2d)
        for detection_3d_rp in detection_3d_rps:
            y1_3d, x1_3d, y2_3d, x2_3d = detection_3d_rp.bbox

            ious = []
            for detection_2d_rp in detection_2d_rps:
                y1_2d, x1_2d, y2_2d, x2_2d = detection_2d_rp.bbox

                iou = bb_intersection_over_union((x1_3d, y1_3d, x2_3d, y2_3d), (x1_2d, y1_2d, x2_2d, y2_2d))
                ious.append(iou)

            # if there is no box in 2d mask, there are no ious
            if len(ious) == 0:
                ious.append(0)
            # if box does not exist in 2d, iou is 0
            max_iou = max(ious)
            if max_iou == 0:
                cuboid_filename = get_corresponding_cuboid_box_filename(cuboid_patch_information_from_3d_folder,
                                                                        filename_wo_ext, detection_3d_rp.bbox)
                if cuboid_filename != '':
                    s_3d = get_score_from_3d_box_scores(cuboid_filename, scores_of_3d)
                    if s_3d != -1:
                        box_scores_2d.append({
                            'box': {
                                'y1': y1_3d,
                                'x1': x1_3d,
                                'y2': y2_3d,
                                'x2': x2_3d,
                            },
                            'average_score': s_3d,
                            'max_score': s_3d,
                        })

        box_scores_output_path = os.path.join(box_scores_output_folder, f'{filename_wo_ext}.json')
        with open(box_scores_output_path, 'w') as f:
            json.dump(box_scores_2d, f)


if __name__ == '__main__':
    main()
