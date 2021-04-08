import json
import os
from glob import glob

import cv2
from tqdm import tqdm

boxes_list_folder = ''
# just used for dimensions
masks_folder = ''
output_folder = ''


def box_coords_from_center_points(max_height, max_width, cy, cx, dimension):
    either_side = dimension / 2

    y1 = cy - either_side
    y2 = cy + either_side

    x1 = cx - either_side
    x2 = cx + either_side

    y1 = y1 if y1 > 0 else 0
    x1 = x1 if x1 > 0 else 0

    y2 = y2 if y2 < max_height else max_height
    x2 = x2 if x2 < max_width else max_width

    return int(y1), int(x1), int(y2), int(x2)


def main():
    os.makedirs(output_folder, exist_ok=True)

    box_list_paths = glob(os.path.join(boxes_list_folder, '*'))
    for box_list_path in tqdm(box_list_paths):
        filename = os.path.basename(box_list_path)
        filename_wo_ext, _ = os.path.splitext(filename)

        mask_path = os.path.join(masks_folder, f'{filename_wo_ext}.png')
        mask = cv2.imread(mask_path)

        with open(box_list_path, 'r') as f:
            boxes = json.load(f)

        all_boxes = []
        for box in boxes:
            y1, x1, y2, x2 = box

            cy = y1 + ((y2 - y1) / 2)
            cx = x1 + ((x2 - x1) / 2)

            extended_box = box_coords_from_center_points(mask.shape[0], mask.shape[1], cy, cx, 100)

            all_boxes.append({
                'box': box,
                'extended_box': extended_box,
            })

        output_path = os.path.join(output_folder, filename_wo_ext + '.json')
        with open(output_path, 'w') as f:
            json.dump(all_boxes, f)


if __name__ == '__main__':
    main()
