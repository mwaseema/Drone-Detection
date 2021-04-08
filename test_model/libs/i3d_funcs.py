import os
from glob import glob

from tqdm import tqdm
import numpy as np
import cv2
import skimage.measure


def i3d_prediction(model, features_folder, output_folder):
    feature_paths = glob(os.path.join(features_folder, '*'))
    feature_paths.sort()
    os.makedirs(output_folder, exist_ok=True)

    for feature_path in tqdm(feature_paths, desc="Making predictions", unit='prediction'):
        filename = os.path.basename(feature_path)
        filename_wo_ext, _ = os.path.splitext(filename)
        output_file_path = os.path.join(output_folder, f'{filename_wo_ext}.png')

        feature_array = np.load(feature_path)

        model.predict_segmentation(
            inp=feature_array,
            out_fname=output_file_path
        )


def region_prop_detections(detections_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    detections = glob(os.path.join(detections_folder, '*'))
    for det_path in tqdm(detections, desc='Region propping detections'):
        filename = os.path.basename(det_path)
        det = cv2.imread(det_path)
        det[det > 0] = 255
        label = skimage.measure.label(det)
        props = skimage.measure.regionprops(label)
        for prp in props:
            if prp.bbox_area > 3:
                min_row, min_col, _, max_row, max_col, _ = prp.bbox

                det[min_row:max_row, min_col:max_col, :] = 255

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, det)


def get_contours(image_file):
    image_file = image_file.copy()
    image_gray = cv2.cvtColor(image_file.copy(), cv2.COLOR_BGR2GRAY)
    _, image_thresh = cv2.threshold(image_gray, 127, 255, 0)
    _, image_contours, _ = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return image_contours


def parse_contours(contours):
    new_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 0:
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


def save_contours(contours, output_file_path, is_ground_truth):
    with open(output_file_path, 'w') as f:
        for contour in contours:
            if is_ground_truth:
                string_to_write = f"UAV {contour['x1']} {contour['y1']} {contour['x2']} {contour['y2']}\n"
            else:
                string_to_write = f"UAV 1.0 {contour['x1']} {contour['y1']} {contour['x2']} {contour['y2']}\n"

            f.write(string_to_write)


def generate_annotation_text_files(input_folder: str, output_folder: str, is_ground_truth: True):
    file_paths = glob(os.path.join(input_folder, '*'))

    os.makedirs(output_folder, exist_ok=True)

    desc_str = 'Generating text annotation files'
    if is_ground_truth:
        desc_str += ' for ground truth files'
    else:
        desc_str += ' for detection files'

    pbar = tqdm(total=len(file_paths), desc=desc_str, unit="image", dynamic_ncols=True)
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
            save_contours(contours, annotation_output_file_path, is_ground_truth)

        pbar.update()
    pbar.close()
