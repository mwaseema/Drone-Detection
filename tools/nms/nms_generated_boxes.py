import json
import multiprocessing
import os
import shutil
from glob import glob
from typing import List, Dict

import cv2
import numpy as np
from tqdm import tqdm

from tools.nms.nms import nms

# folder containing video frames
frames_folder = ''
# folder containing patch information json files
patch_information_folder = ''
# absolute path to the file containing score for every cuboid that is passed through temporal stage
box_scores_file_path = ''
# folder containing predictions generated by temporal stage
detection_masks_folder = ''

# folder where only retained patch information will be stored after applying NMS
# filenames of these patch information can be used to copy I3D prediction files which are retained after NMS
output_patch_information_folder = ''

# boxes with IoU equal to and above the value will be removed
nms_iou_threshold = 0.2


def get_box_scores(box_scores_file_path):
    with open(box_scores_file_path, 'r') as f:
        box_scores = json.load(f)

    box_scores_dict = {}
    for box_score in box_scores:
        filename_wo_ext = box_score['filename_wo_ext']
        score = box_score['score']

        if filename_wo_ext not in list(box_scores_dict.keys()):
            box_scores_dict[filename_wo_ext] = score
    return box_scores_dict


def get_boxes_and_scores_lists(patch_information_boxes: List[Dict]):
    boxes = []
    scores = []

    for patch_information_box in patch_information_boxes:
        boxes.append(patch_information_box['box'])
        scores.append(patch_information_box['score'])

    return boxes, scores


def find_corresponding_patch_information(patch_information_boxes, nms_boxes):
    patch_information_to_retain = []
    patch_information_box_used = []

    for nms_box in nms_boxes:
        nms_y1, nms_x1, nms_y2, nms_x2 = nms_box

        for i, patch_information_box in enumerate(patch_information_boxes):
            if i not in patch_information_box_used:
                pib_y1, pib_x1, pib_y2, pib_x2 = patch_information_box['box']

                if nms_y1 == pib_y1 and nms_x1 == pib_x1 and nms_y2 == pib_y2 and nms_x2 == pib_x2:
                    patch_information_to_retain.append(patch_information_box['filename_wo_ext'])
                    patch_information_box_used.append(i)
                    break
    return patch_information_to_retain


def main_process(params):
    frame_path = params['frame_path']
    box_scores = params['box_scores']
    detection_masks_new_folder = params['detection_masks_new_folder']

    filename = os.path.basename(frame_path)
    filename_wo_ext, _ = os.path.splitext(filename)

    # loading boxes from patch information boxes
    patch_information_boxes = []
    patch_information_paths = glob(os.path.join(patch_information_folder, f'{filename_wo_ext}*'))
    for patch_information_path in patch_information_paths:
        patch_information_filename = os.path.basename(patch_information_path)
        patch_information_filename_wo_ext, _ = os.path.splitext(patch_information_filename)

        detection_mask_path = os.path.join(detection_masks_folder, patch_information_filename_wo_ext + '.png')
        detection_mask = cv2.imread(detection_mask_path)
        if not (np.max(detection_mask) > 0):
            continue

        with open(patch_information_path, 'r') as f:
            patch_information = json.load(f)

        y1 = patch_information['box']['y1']
        x1 = patch_information['box']['x1']
        y2 = patch_information['box']['y2']
        x2 = patch_information['box']['x2']

        patch_information_boxes.append({
            'filename_wo_ext': patch_information_filename_wo_ext,
            'box': (y1, x1, y2, x2),
            'score': box_scores[patch_information_filename_wo_ext]
        })

    boxes, scores = get_boxes_and_scores_lists(patch_information_boxes)
    boxes, scores = nms(boxes, scores, nms_iou_threshold)

    retain_patch_information_files = find_corresponding_patch_information(patch_information_boxes, boxes)
    for retain_patch_information_file in retain_patch_information_files:
        source_path = os.path.join(patch_information_folder, f'{retain_patch_information_file}.txt')
        output_path = os.path.join(output_patch_information_folder, f'{retain_patch_information_file}.txt')
        shutil.copyfile(source_path, output_path)

        detection_mask_source_path = os.path.join(detection_masks_folder, f'{retain_patch_information_file}.png')
        detection_mask_target_path = os.path.join(detection_masks_new_folder, f'{retain_patch_information_file}.png')
        shutil.copyfile(detection_mask_source_path, detection_mask_target_path)


def main():
    os.makedirs(output_patch_information_folder, exist_ok=True)

    detection_masks_new_folder = detection_masks_folder + '_new'
    os.makedirs(detection_masks_new_folder, exist_ok=True)

    frame_paths = glob(os.path.join(frames_folder, '*'))
    box_scores = get_box_scores(box_scores_file_path)

    params = []
    for frame_path in frame_paths:
        params.append({
            'frame_path': frame_path,
            'box_scores': box_scores,
            'detection_masks_new_folder': detection_masks_new_folder,
        })

    pool = multiprocessing.Pool(os.cpu_count() - 1)
    for _ in tqdm(pool.imap_unordered(main_process, params), 'Processing frames', total=len(frame_paths)):
        pass

    os.rename(detection_masks_folder, detection_masks_folder + '_2')
    os.rename(detection_masks_new_folder, detection_masks_folder)


if __name__ == '__main__':
    main()
