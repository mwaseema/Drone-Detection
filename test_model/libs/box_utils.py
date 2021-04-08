from typing import List, Dict, Tuple

import cv2
import numpy as np
import skimage.measure


def get_region_props(image: np.ndarray) -> List[skimage.measure._regionprops._RegionProperties]:
    image = image.copy()

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    label_image = skimage.measure.label(image)
    region_props = skimage.measure.regionprops(label_image)
    return region_props


def get_bounding_boxes(image: np.ndarray) -> List[Dict[str, int]]:
    props = get_region_props(image)

    coords = []
    for prop in props:
        min_row, min_col, max_row, max_col = prop.bbox
        coords.append({
            'x1': min_col,
            'x2': max_col,
            'y1': min_row,
            'y2': max_row,
        })
    return coords


def get_safe_coords(height, width, extra_pixels, y1, x1, y2, x2) -> Tuple[int, int, int, int]:
    y1 -= extra_pixels
    x1 -= extra_pixels

    y1 = y1 if y1 >= 0 else 0
    x1 = x1 if x1 >= 0 else 0

    y2 += extra_pixels
    x2 += extra_pixels

    y2 = y2 if y2 <= height else height
    x2 = x2 if x2 <= width else width

    return y1, x1, y2, x2
