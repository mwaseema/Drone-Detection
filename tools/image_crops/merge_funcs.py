from typing import List

import numpy as np
import skimage.measure


def convert_segmented_area_to_bounding_box(segmented_mask):
    new_segmented_mask = segmented_mask.copy()
    label_image = skimage.measure.label(new_segmented_mask)
    region_props = skimage.measure.regionprops(label_image)

    for region_prop in region_props:
        min_row, min_col, _, max_row, max_col, _ = region_prop.bbox

        new_segmented_mask[min_row:max_row, min_col:max_col, :] = 255

    return new_segmented_mask


def merge_segmentation_masks_4splits(img_crops_list):
    extra_area = 50

    original_width = (img_crops_list[0].shape[1] - extra_area) + (img_crops_list[1].shape[1] - extra_area)
    original_height = (img_crops_list[0].shape[0] - extra_area) + (img_crops_list[2].shape[0] - extra_area)

    new_mask = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    mask_of_1 = new_mask.copy()
    mask_of_1[0:img_crops_list[0].shape[0], 0:img_crops_list[0].shape[1]] = img_crops_list[0]

    mask_of_2 = new_mask.copy()
    mask_of_2[0:img_crops_list[1].shape[0], original_width - img_crops_list[1].shape[1]:original_width] = \
        img_crops_list[1]

    mask_of_3 = new_mask.copy()
    mask_of_3[original_height - img_crops_list[2].shape[0]:original_height, 0:img_crops_list[2].shape[1]]

    mask_of_4 = new_mask.copy()
    mask_of_4[original_height - img_crops_list[3].shape[0]:original_height,
    original_width - img_crops_list[3].shape[1]:original_width]

    new_mask[mask_of_1 > 0] = 255
    new_mask[mask_of_2 > 0] = 255
    new_mask[mask_of_3 > 0] = 255
    new_mask[mask_of_4 > 0] = 255

    return new_mask


def merge_segmentation_masks(img_crops_list: List[np.ndarray], extra_area=50):
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

    new_mask[mask_of_1 > 0] = 255
    new_mask[mask_of_2 > 0] = 255
    new_mask[mask_of_3 > 0] = 255
    new_mask[mask_of_4 > 0] = 255
    new_mask[mask_of_5 > 0] = 255
    new_mask[mask_of_6 > 0] = 255
    new_mask[mask_of_7 > 0] = 255
    new_mask[mask_of_8 > 0] = 255
    new_mask[mask_of_9 > 0] = 255

    return new_mask
