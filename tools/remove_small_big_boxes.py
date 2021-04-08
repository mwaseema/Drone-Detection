import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from tools.crf.IoU_test import get_region_props

masks_folder = ''
output_folder = ''

min_box_threshold = 80
max_box_threshold = 10000


def main():
    os.makedirs(output_folder, exist_ok=True)

    mask_paths = glob(os.path.join(masks_folder, '*'))
    for mask_path in tqdm(mask_paths):
        filename = os.path.basename(mask_path)

        mask = cv2.imread(mask_path)
        rps = get_region_props(mask)

        new_mask = np.zeros(mask.shape, mask.dtype)
        for rp in rps:
            if min_box_threshold <= rp.bbox_area < max_box_threshold:
                y1, x1, y2, x2 = rp.bbox
                new_mask[y1:y2, x1:x2] = 255

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, new_mask)


if __name__ == '__main__':
    main()
