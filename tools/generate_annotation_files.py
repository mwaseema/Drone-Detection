import argparse
import json
import os
from glob import glob
from pathlib import Path

import cv2
from tqdm import tqdm

from test_model.temporal.results_generation import calculate_distance
from utils.data_funcs import bb_intersection_over_union

parser = argparse.ArgumentParser(
    description="Generate annotation files that'll be used by metrics code to generate scores")
parser.add_argument('--detections_folder', type=str, required=True, help="Path to folder containing detection masks")
parser.add_argument('--detections_output_folder', type=str, required=True,
                    help="Path to folder which will have detection annotation files")
parser.add_argument('--ground_truth_folder', type=str, required=True,
                    help="Path to folder containing ground truth masks")
parser.add_argument('--ground_truth_output_folder', type=str, required=True,
                    help="Path to folder which will have ground truth annotation files")
parser.add_argument('--score_boxes_folder', type=str, default='',
                    help="Folder where scores for the boxes of the files are stored")
args = parser.parse_args()

detections_folder = args.detections_folder
detections_output_folder = args.detections_output_folder

ground_truth_folder = args.ground_truth_folder
ground_truth_output_folder = args.ground_truth_output_folder

score_boxes_folder = args.score_boxes_folder


def create_folders_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def delete_unmatched_files_from_destination(source_folder: str, dest_folder: str, extension_to_search: str = None):
    if extension_to_search is not None:
        search_filter = f"*{extension_to_search}"
    else:
        search_filter = "*"

    source_file_paths = glob(os.path.join(source_folder, search_filter))
    dest_file_paths = glob(os.path.join(dest_folder, search_filter))

    for dst_file_path in tqdm(dest_file_paths, desc='Synchronising files', unit='file'):
        dst_filename = os.path.basename(dst_file_path)

        exists = False
        for src_file_path in source_file_paths:
            src_filename = os.path.basename(src_file_path)

            if src_filename == dst_filename:
                exists = True
                break

        if not exists:
            os.remove(dst_file_path)


def get_files_from_folder(folder_path):
    all_files = []

    for fl in Path(folder_path).iterdir():
        filename = fl.name
        file_path = os.path.join(folder_path, filename)
        all_files.append(file_path)

    return all_files


def get_contours(image_file):
    image_file = image_file.copy()
    image_gray = cv2.cvtColor(image_file.copy(), cv2.COLOR_BGR2GRAY)
    _, image_thresh = cv2.threshold(image_gray, 127, 255, 0)

    # _, image_contours, _ = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # image_contours, _ = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ret = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(ret) == 3:
        _, image_contours, _ = ret
    else:
        image_contours, _ = ret

    return image_contours


def parse_contours(contours):
    new_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 3:
            x1, y1, w, h = cv2.boundingRect(contour)
            x2 = x1 + w
            y2 = y1 + h
            temp = {
                'x1': x1,
                'x2': x2,
                'y1': y1,
                'y2': y2,
            }

            new_contours.append(temp)
    return new_contours


def get_box_score_iou_based(score_boxes_folder: str, filename: str, contour):
    if score_boxes_folder == '':
        return 1.0
    else:
        y1 = contour['y1']
        x1 = contour['x1']
        y2 = contour['y2']
        x2 = contour['x2']

        filename_wo_ext, _ = os.path.splitext(filename)
        score_boxes_path = os.path.join(score_boxes_folder, f'{filename_wo_ext}.json')

        with open(score_boxes_path) as f:
            score_boxes = json.load(f)

        for sb in score_boxes:
            sb_y1 = sb['box']['y1']
            sb_x1 = sb['box']['x1']
            sb_y2 = sb['box']['y2']
            sb_x2 = sb['box']['x2']

            iou = bb_intersection_over_union((x1, y1, x2, y2), (sb_x1, sb_y1, sb_x2, sb_y2))
            if iou > 0.95:
                return sb['average_score']

    raise Exception('Score for the given box not found!')


def get_box_score(score_boxes_folder: str, filename: str, contour):
    if score_boxes_folder == '':
        return 1.0
    else:
        y1 = contour['y1']
        x1 = contour['x1']
        y2 = contour['y2']
        x2 = contour['x2']

        box_center_x1 = int(x1 + ((x2 - x1) / 2))
        box_center_y1 = int(y1 + ((y2 - y1) / 2))

        filename_wo_ext, _ = os.path.splitext(filename)
        score_boxes_path = os.path.join(score_boxes_folder, f'{filename_wo_ext}.json')

        with open(score_boxes_path) as f:
            score_boxes = json.load(f)

        distances = []
        for sb in score_boxes:
            sb_y1 = sb['box']['y1']
            sb_x1 = sb['box']['x1']
            sb_y2 = sb['box']['y2']
            sb_x2 = sb['box']['x2']

            box_center_x2 = int(sb_x1 + ((sb_x2 - sb_x1) / 2))
            box_center_y2 = int(sb_y1 + ((sb_y2 - sb_y1) / 2))

            distance = calculate_distance(box_center_y1, box_center_x1, box_center_y2, box_center_x2)
            distances.append(distance)

        if len(distances) > 0:
            min_distance_index = distances.index(min(distances))
            return score_boxes[min_distance_index]['average_score']
        else:
            return 0


def save_contours(contours, output_file_path, is_ground_truth, score_boxes_folder=None):
    with open(output_file_path, 'w') as f:
        for contour in contours:
            if is_ground_truth:
                string_to_write = f"UAV {contour['x1']} {contour['y1']} {contour['x2']} {contour['y2']}\n"
            else:
                box_score = get_box_score(score_boxes_folder, os.path.basename(output_file_path), contour)
                string_to_write = f"UAV {box_score} {contour['x1']} {contour['y1']} {contour['x2']} {contour['y2']}\n"

            f.write(string_to_write)


def generate_annotation_text_files(input_folder: str, output_folder: str, is_ground_truth: True):
    file_paths = get_files_from_folder(input_folder)

    create_folders_if_not_exists(output_folder)

    pbar_description = 'Generating text annotation files'
    pbar_description = pbar_description if not is_ground_truth else pbar_description + " for ground truth"
    pbar = tqdm(total=len(file_paths), desc=pbar_description, unit="image", dynamic_ncols=True)
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        filename_without_ext, file_ext = os.path.splitext(filename)

        fl = cv2.imread(file_path)
        contours = get_contours(fl)
        contours = parse_contours(contours)

        if is_ground_truth:
            if len(contours) > 0:
                annotation_output_file_path = os.path.join(output_folder, f"{filename_without_ext}.txt")
                save_contours(contours, annotation_output_file_path, is_ground_truth)
        else:
            annotation_output_file_path = os.path.join(output_folder, f"{filename_without_ext}.txt")
            save_contours(contours, annotation_output_file_path, is_ground_truth, score_boxes_folder)

        pbar.update()
    pbar.close()


if __name__ == '__main__':
    # for ground truth
    generate_annotation_text_files(detections_folder,
                                   detections_output_folder, False)
    # for detections
    generate_annotation_text_files(ground_truth_folder,
                                   ground_truth_output_folder, True)

    delete_unmatched_files_from_destination(ground_truth_output_folder, detections_output_folder)
