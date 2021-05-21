import json
import math
import os
from glob import glob
from os import path
from pathlib import Path
from shutil import copyfile
from typing import Union

import cv2
import numpy as np
import skimage.measure
from tqdm import tqdm

from .box_utils import get_region_props


def create_folders_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_files_from_folder(folder_path):
    all_files = []

    for fl in Path(folder_path).iterdir():
        filename = fl.name
        file_path = os.path.join(folder_path, filename)
        all_files.append(file_path)

    return all_files


def write_dict_to_json_file(file_path, dictionary_obj):
    with open(file_path, 'w') as fp:
        json.dump(dictionary_obj, fp)


def convert_segmented_area_to_bounding_box(segmented_mask):
    new_segmented_mask = segmented_mask.copy()
    label_image = skimage.measure.label(new_segmented_mask)
    region_props = skimage.measure.regionprops(label_image)

    for region_prop in region_props:
        min_row, min_col, _, max_row, max_col, _ = region_prop.bbox

        new_segmented_mask[min_row:max_row, min_col:max_col, :] = 255

    return new_segmented_mask


def get_frames_and_ground_truths_from_folder(frames_folder, ground_truths_folder):
    all_frame_paths = get_files_from_folder(frames_folder)

    frame_ground_truth_paths = []
    for frame_path in all_frame_paths:
        frame_filename = os.path.basename(frame_path)
        ground_truth_path = os.path.join(ground_truths_folder, frame_filename)

        frame_ground_truth_paths.append({
            'frame': frame_path,
            'ground_truth': ground_truth_path
        })

    return frame_ground_truth_paths


def get_frame_and_ground_truth_crop_4splits(frame_path, ground_truth_path=None):
    extra_area = 50

    frame = cv2.imread(frame_path)
    ground_truth = cv2.imread(ground_truth_path)

    height, width, _ = frame.shape
    mid_height = math.floor(height / 2)
    mid_width = math.floor(width / 2)

    # top left
    frame1 = frame[0:mid_height + extra_area, 0:mid_width + extra_area, :]

    # top right
    frame2 = frame[0:mid_height + extra_area, mid_width - extra_area:width, :]

    # bottom left
    frame3 = frame[mid_height - extra_area:height, 0:mid_width + extra_area, :]

    # bottom right
    frame4 = frame[mid_height - extra_area:height, mid_width - extra_area:width, :]

    if ground_truth_path is not None:
        # top left
        ground_truth1 = ground_truth[0:mid_height + extra_area, 0:mid_width + extra_area, :]
        # top right
        ground_truth2 = ground_truth[0:mid_height + extra_area, mid_width - extra_area:width, :]
        # bottom left
        ground_truth3 = ground_truth[mid_height - extra_area:height, 0:mid_width + extra_area, :]
        # bottom right
        ground_truth4 = ground_truth[mid_height - extra_area:height, mid_width - extra_area:width, :]

        return [frame1, frame2, frame3, frame4], [ground_truth1, ground_truth2, ground_truth3, ground_truth4]
    else:
        return [frame1, frame2, frame3, frame4]


def get_frame_and_ground_truth_crop(frame_path: Union[str, np.ndarray],
                                    ground_truth_path: Union[str, np.ndarray] = None):
    extra_area = 50

    assert isinstance(frame_path, str) or isinstance(frame_path,
                                                     np.ndarray), "Frame path should be string or numpy array"
    assert ground_truth_path is None or isinstance(ground_truth_path, str) or isinstance(ground_truth_path,
                                                                                         np.ndarray), "Frame path should be string or numpy array"

    if isinstance(frame_path, str):
        frame = cv2.imread(frame_path)
    else:
        frame = frame_path

    if isinstance(ground_truth_path, str):
        ground_truth = cv2.imread(ground_truth_path)
    else:
        ground_truth = ground_truth_path

    height, width, _ = frame.shape
    split_height = math.floor(height / 3)
    split_width = math.floor(width / 3)

    # top left
    frame1 = frame[0:split_height + extra_area, 0:split_width + extra_area, :]
    # top middle
    frame2 = frame[0:split_height + extra_area, split_width - extra_area:split_width + split_width + extra_area, :]
    # top right
    frame3 = frame[0:split_height + extra_area, (split_width + split_width) - extra_area:width, :]
    # middle left
    frame4 = frame[split_height - extra_area:split_height + split_height + extra_area, 0:split_width + extra_area, :]
    # middle middle
    frame5 = frame[split_height - extra_area:split_height + split_height + extra_area,
             split_width - extra_area:split_width + split_width + extra_area, :]
    # middle right
    frame6 = frame[split_height - extra_area:split_height + split_height + extra_area,
             (split_width + split_width) - extra_area:width, :]
    # bottom left
    frame7 = frame[(split_height + split_height) - extra_area:height, 0:split_width + extra_area, :]
    # bottom middle
    frame8 = frame[(split_height + split_height) - extra_area:height,
             split_width - extra_area:split_width + split_width + extra_area, :]
    # bottom right
    frame9 = frame[(split_height + split_height) - extra_area:height, (split_width + split_width) - extra_area:width, :]

    if ground_truth_path is not None:
        # top left
        ground_truth1 = ground_truth[0:split_height + extra_area, 0:split_width + extra_area, :]
        # top middle
        ground_truth2 = ground_truth[0:split_height + extra_area,
                        split_width - extra_area:split_width + split_width + extra_area, :]
        # top right
        ground_truth3 = ground_truth[0:split_height + extra_area, (split_width + split_width) - extra_area:width, :]
        # middle left
        ground_truth4 = ground_truth[split_height - extra_area:split_height + split_height + extra_area,
                        0:split_width + extra_area, :]
        # middle middle
        ground_truth5 = ground_truth[split_height - extra_area:split_height + split_height + extra_area,
                        split_width - extra_area:split_width + split_width + extra_area, :]
        # middle right
        ground_truth6 = ground_truth[split_height - extra_area:split_height + split_height + extra_area,
                        (split_width + split_width) - extra_area:width, :]
        # bottom left
        ground_truth7 = ground_truth[(split_height + split_height) - extra_area:height, 0:split_width + extra_area, :]
        # bottom middle
        ground_truth8 = ground_truth[(split_height + split_height) - extra_area:height,
                        split_width - extra_area:split_width + split_width + extra_area, :]
        # bottom right
        ground_truth9 = ground_truth[(split_height + split_height) - extra_area:height,
                        (split_width + split_width) - extra_area:width, :]

        return [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9], [ground_truth1, ground_truth2,
                                                                                          ground_truth3, ground_truth4,
                                                                                          ground_truth5, ground_truth6,
                                                                                          ground_truth7, ground_truth8,
                                                                                          ground_truth9]
    else:
        return [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9]


def crop_and_save_croped_data(input_frames_folder, input_ground_truth_folder, output_frames_folder,
                              output_ground_truth_folder):
    create_folders_if_not_exists(output_frames_folder)
    create_folders_if_not_exists(output_ground_truth_folder)

    # get all the frames and ground truths
    frames_and_ground_truth_paths = get_frames_and_ground_truths_from_folder(input_frames_folder,
                                                                             input_ground_truth_folder)

    pbar = tqdm(total=len(frames_and_ground_truth_paths), desc="Croping and saving", unit=" frame", dynamic_ncols=True)
    for frame_ground_truth_path in frames_and_ground_truth_paths:
        frame_crops, ground_truth_crops = get_frame_and_ground_truth_crop(frame_ground_truth_path['frame'],
                                                                          frame_ground_truth_path['ground_truth'])
        frame_filename = os.path.basename(frame_ground_truth_path['frame'])
        frame_filename_without_ext, frame_ext = os.path.splitext(frame_filename)

        # copy original frame and ground truth
        copyfile(frame_ground_truth_path['frame'], os.path.join(output_frames_folder, frame_filename))
        copyfile(frame_ground_truth_path['ground_truth'], os.path.join(output_ground_truth_folder, frame_filename))

        # for copying cropped files
        for i in range(len(frame_crops)):
            crop_filename = f"{frame_filename_without_ext}_crop_{i + 1}{frame_ext}"
            # saving cropped frame
            cv2.imwrite(os.path.join(output_frames_folder, crop_filename), frame_crops[i])
            # saving ground truth frame
            cv2.imwrite(os.path.join(output_ground_truth_folder, crop_filename), ground_truth_crops[i])

        pbar.update()
    pbar.close()


def merge_segmentation_masks_4splits(img_crops_list, replace_values_greater_than_0=True):
    extra_area = 50

    original_width = (img_crops_list[0].shape[1] - extra_area) + (img_crops_list[1].shape[1] - extra_area)
    original_height = (img_crops_list[0].shape[0] - extra_area) + (img_crops_list[2].shape[0] - extra_area)

    new_mask = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    start_height_for_row_1 = 0
    end_height_for_row_1 = img_crops_list[0].shape[0]

    start_height_for_row_2 = (end_height_for_row_1 - extra_area) - extra_area
    end_height_for_row_2 = original_height

    start_width_for_col_1 = 0
    end_width_for_col_1 = img_crops_list[0].shape[1]

    start_width_for_col_2 = (end_width_for_col_1 - extra_area) - extra_area
    end_width_for_col_2 = original_width

    mask_of_1 = new_mask.copy()
    mask_of_1[start_height_for_row_1:end_height_for_row_1, start_width_for_col_1:end_width_for_col_1] = img_crops_list[
        0]

    mask_of_2 = new_mask.copy()
    mask_of_2[start_height_for_row_1:end_height_for_row_1, start_width_for_col_2:end_width_for_col_2] = img_crops_list[
        1]

    mask_of_3 = new_mask.copy()
    mask_of_3[start_height_for_row_2:end_height_for_row_2, start_width_for_col_1:end_width_for_col_1] = img_crops_list[
        2]

    mask_of_4 = new_mask.copy()
    mask_of_4[start_height_for_row_2:end_height_for_row_2, start_width_for_col_2:end_width_for_col_2] = img_crops_list[
        3]

    if replace_values_greater_than_0:
        new_mask[mask_of_1 > 0] = 255
        new_mask[mask_of_2 > 0] = 255
        new_mask[mask_of_3 > 0] = 255
        new_mask[mask_of_4 > 0] = 255
    else:
        new_mask[mask_of_1 > 0] = mask_of_1[mask_of_1 > 0]
        new_mask[mask_of_2 > 0] = mask_of_2[mask_of_2 > 0]
        new_mask[mask_of_3 > 0] = mask_of_3[mask_of_3 > 0]
        new_mask[mask_of_4 > 0] = mask_of_4[mask_of_4 > 0]

    return new_mask


def merge_segmentation_masks(img_crops_list, replace_values_greater_than_0=True):
    extra_area = 50

    original_width = (img_crops_list[0].shape[1] - extra_area) + (
            img_crops_list[1].shape[1] - extra_area - extra_area) + (img_crops_list[2].shape[1] - extra_area)
    original_height = (img_crops_list[0].shape[0] - extra_area) + (
            img_crops_list[3].shape[0] - extra_area - extra_area) + (img_crops_list[6].shape[0] - extra_area)

    new_mask = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    start_height_for_row_1 = 0
    end_height_for_row_1 = img_crops_list[0].shape[0]

    start_height_for_row_2 = (end_height_for_row_1 - extra_area) - extra_area
    end_height_for_row_2 = start_height_for_row_2 + img_crops_list[3].shape[0]

    start_height_for_row_3 = (end_height_for_row_2 - extra_area) - extra_area
    end_height_for_row_3 = original_height

    start_width_for_col_1 = 0
    end_width_for_col_1 = img_crops_list[0].shape[1]

    start_width_for_col_2 = (end_width_for_col_1 - extra_area) - extra_area
    end_width_for_col_2 = start_width_for_col_2 + img_crops_list[1].shape[1]

    start_width_for_col_3 = (end_width_for_col_2 - extra_area) - extra_area
    end_width_for_col_3 = start_width_for_col_3 + original_width

    mask_of_1 = new_mask.copy()
    mask_of_1[start_height_for_row_1:end_height_for_row_1, start_width_for_col_1:end_width_for_col_1] = img_crops_list[
        0]

    mask_of_2 = new_mask.copy()
    mask_of_2[start_height_for_row_1:end_height_for_row_1, start_width_for_col_2:end_width_for_col_2] = img_crops_list[
        1]

    mask_of_3 = new_mask.copy()
    mask_of_3[start_height_for_row_1:end_height_for_row_1, start_width_for_col_3:end_width_for_col_3] = img_crops_list[
        2]

    mask_of_4 = new_mask.copy()
    mask_of_4[start_height_for_row_2:end_height_for_row_2, start_width_for_col_1:end_width_for_col_1] = img_crops_list[
        3]

    mask_of_5 = new_mask.copy()
    mask_of_5[start_height_for_row_2:end_height_for_row_2, start_width_for_col_2:end_width_for_col_2] = img_crops_list[
        4]

    mask_of_6 = new_mask.copy()
    mask_of_6[start_height_for_row_2:end_height_for_row_2, start_width_for_col_3:end_width_for_col_3] = img_crops_list[
        5]

    mask_of_7 = new_mask.copy()
    mask_of_7[start_height_for_row_3:end_height_for_row_3, start_width_for_col_1:end_width_for_col_1] = img_crops_list[
        6]

    mask_of_8 = new_mask.copy()
    mask_of_8[start_height_for_row_3:end_height_for_row_3, start_width_for_col_2:end_width_for_col_2] = img_crops_list[
        7]

    mask_of_9 = new_mask.copy()
    mask_of_9[start_height_for_row_3:end_height_for_row_3, start_width_for_col_3:end_width_for_col_3] = img_crops_list[
        8]

    if replace_values_greater_than_0:
        new_mask[mask_of_1 > 0] = 255
        new_mask[mask_of_2 > 0] = 255
        new_mask[mask_of_3 > 0] = 255
        new_mask[mask_of_4 > 0] = 255
        new_mask[mask_of_5 > 0] = 255
        new_mask[mask_of_6 > 0] = 255
        new_mask[mask_of_7 > 0] = 255
        new_mask[mask_of_8 > 0] = 255
        new_mask[mask_of_9 > 0] = 255
    else:
        new_mask[mask_of_1 > 0] = mask_of_1[mask_of_1 > 0]
        new_mask[mask_of_2 > 0] = mask_of_2[mask_of_2 > 0]
        new_mask[mask_of_3 > 0] = mask_of_3[mask_of_3 > 0]
        new_mask[mask_of_4 > 0] = mask_of_4[mask_of_4 > 0]
        new_mask[mask_of_5 > 0] = mask_of_5[mask_of_5 > 0]
        new_mask[mask_of_6 > 0] = mask_of_6[mask_of_6 > 0]
        new_mask[mask_of_7 > 0] = mask_of_7[mask_of_7 > 0]
        new_mask[mask_of_8 > 0] = mask_of_8[mask_of_8 > 0]
        new_mask[mask_of_9 > 0] = mask_of_9[mask_of_9 > 0]

    return new_mask


def predict_segmentation_for_files(model, image_paths, output_folder):
    # make output folder if doesn't exists
    create_folders_if_not_exists(output_folder)

    pbr = tqdm(total=len(image_paths), desc='Predicting results', unit="frame", dynamic_ncols=True)
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        filename_wo_ext, _ = os.path.splitext(filename)

        output_image_path = os.path.join(output_folder, filename)

        img_crops = get_frame_and_ground_truth_crop(img_path)
        img_segmentations = []
        img_probabilities = []
        for img_crop in img_crops:
            segm_pred, seg_probabilities = model.predict_segmentation(
                inp=img_crop
            )
            # appending to the list of predictions
            img_segmentations.append(segm_pred)
            img_probabilities.append(seg_probabilities[:, :, 1])

        # merge segmentations
        merged_mask = merge_segmentation_masks(img_segmentations)
        # converting predicted region to square
        merged_mask = convert_segmented_area_to_bounding_box(merged_mask)

        cv2.imwrite(output_image_path, merged_mask)

        merged_probabilities = merge_probabilities(merged_mask.shape, img_probabilities, 50)

        merged_mask_rps = get_region_props(merged_mask)
        box_scores = []
        for merged_mask_rp in merged_mask_rps:
            y1, x1, y2, x2 = merged_mask_rp.bbox
            probabilities_patch = merged_probabilities[y1:y2, x1:x2]

            box_scores.append({
                'box': {
                    'y1': y1,
                    'x1': x1,
                    'y2': y2,
                    'x2': x2,
                },
                'average_score': float(np.mean(probabilities_patch)),
                'max_score': float(np.max(probabilities_patch))
            })

        box_scores_path = path.abspath(
            path.join(path.dirname(output_image_path), '..', 'box_scores', f'{filename_wo_ext}.json'))
        os.makedirs(os.path.dirname(box_scores_path), exist_ok=True)
        with open(box_scores_path, 'w') as f:
            json.dump(box_scores, f)

        pbr.update()
    pbr.close()


def get_contours(image_file):
    image_file = image_file.copy()
    image_gray = cv2.cvtColor(image_file.copy(), cv2.COLOR_BGR2GRAY)
    _, image_thresh = cv2.threshold(image_gray, 127, 255, 0)
    find_contours = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # On some environment it was returning tuple of two while on other tuple of 3
    # To make it support both
    if len(find_contours) == 3:
        _, image_contours, _ = find_contours
    else:
        image_contours, _ = find_contours

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


def save_contours(contours, output_file_path, is_ground_truth):
    with open(output_file_path, 'w') as f:
        for contour in contours:
            if is_ground_truth:
                string_to_write = f"UAV {contour['x1']} {contour['y1']} {contour['x2']} {contour['y2']}\n"
            else:
                string_to_write = f"UAV 1.0 {contour['x1']} {contour['y1']} {contour['x2']} {contour['y2']}\n"

            f.write(string_to_write)


def generate_annotation_text_files(input_folder: str, output_folder: str, is_ground_truth: True):
    file_paths = get_files_from_folder(input_folder)

    create_folders_if_not_exists(output_folder)

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


def delete_extra_from_destination(source: str, destination: str):
    destination_paths = glob(path.join(destination, '*'))

    for destination_path in tqdm(destination_paths, desc='Deleting extra detection files', unit='file'):
        filename = path.basename(destination_path)

        source_path = path.join(source, filename)
        if not path.exists(source_path):
            os.remove(destination_path)


def merge_probabilities_4splits(mask_shape, probabilities_list, extra_pixels):
    probabilities_list = probabilities_list.copy()

    probability_mask = np.zeros(mask_shape[0:2], probabilities_list[0].dtype)

    probabilities_list[0] = probabilities_list[0][0:-extra_pixels, 0:-extra_pixels]
    probabilities_list[1] = probabilities_list[1][0:-extra_pixels, extra_pixels:]

    probabilities_list[2] = probabilities_list[2][extra_pixels:, 0:-extra_pixels]
    probabilities_list[3] = probabilities_list[3][extra_pixels:, extra_pixels:]

    row_1_start = 0
    row_1_end = row_1_start + probabilities_list[0].shape[0]

    row_2_start = row_1_end
    row_2_end = row_2_start + probabilities_list[2].shape[0]

    col_1_start = 0
    col_1_end = col_1_start + probabilities_list[0].shape[1]

    col_2_start = col_1_end
    col_2_end = col_2_start + probabilities_list[1].shape[1]

    probability_mask[row_1_start:row_1_end, col_1_start:col_1_end] = probabilities_list[0]
    probability_mask[row_1_start:row_1_end, col_2_start:col_2_end] = probabilities_list[1]
    probability_mask[row_2_start:row_2_end, col_1_start:col_1_end] = probabilities_list[2]
    probability_mask[row_2_start:row_2_end, col_2_start:col_2_end] = probabilities_list[3]

    return probability_mask


def merge_probabilities(mask_shape, probabilities_list, extra_pixels):
    probabilities_list = probabilities_list.copy()

    probability_mask = np.zeros(mask_shape[0:2], probabilities_list[0].dtype)

    probabilities_list[0] = probabilities_list[0][0:-extra_pixels, 0:-extra_pixels]
    probabilities_list[1] = probabilities_list[1][0:-extra_pixels, extra_pixels:-extra_pixels]
    probabilities_list[2] = probabilities_list[2][0:-extra_pixels, extra_pixels:]

    probabilities_list[3] = probabilities_list[3][extra_pixels:-extra_pixels, 0:-extra_pixels]
    probabilities_list[4] = probabilities_list[4][extra_pixels:-extra_pixels, extra_pixels:-extra_pixels]
    probabilities_list[5] = probabilities_list[5][extra_pixels:-extra_pixels, extra_pixels:]

    probabilities_list[6] = probabilities_list[6][extra_pixels:, 0:-extra_pixels]
    probabilities_list[7] = probabilities_list[7][extra_pixels:, extra_pixels:-extra_pixels]
    probabilities_list[8] = probabilities_list[8][extra_pixels:, extra_pixels:]

    row_1_start = 0
    row_1_end = row_1_start + probabilities_list[0].shape[0]

    row_2_start = row_1_end
    row_2_end = row_2_start + probabilities_list[3].shape[0]

    row_3_start = row_2_end
    row_3_end = row_3_start + probabilities_list[6].shape[0]

    col_1_start = 0
    col_1_end = col_1_start + probabilities_list[0].shape[1]

    col_2_start = col_1_end
    col_2_end = col_2_start + probabilities_list[1].shape[1]

    col_3_start = col_2_end
    col_3_end = col_3_start + probabilities_list[2].shape[1]

    probability_mask[row_1_start:row_1_end, col_1_start:col_1_end] = probabilities_list[0]
    probability_mask[row_1_start:row_1_end, col_2_start:col_2_end] = probabilities_list[1]
    probability_mask[row_1_start:row_1_end, col_3_start:col_3_end] = probabilities_list[2]
    probability_mask[row_2_start:row_2_end, col_1_start:col_1_end] = probabilities_list[3]
    probability_mask[row_2_start:row_2_end, col_2_start:col_2_end] = probabilities_list[4]
    probability_mask[row_2_start:row_2_end, col_3_start:col_3_end] = probabilities_list[5]
    probability_mask[row_3_start:row_3_end, col_1_start:col_1_end] = probabilities_list[6]
    probability_mask[row_3_start:row_3_end, col_2_start:col_2_end] = probabilities_list[7]
    probability_mask[row_3_start:row_3_end, col_3_start:col_3_end] = probabilities_list[8]

    return probability_mask
