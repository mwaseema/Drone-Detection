import json
import multiprocessing
import os
import random
from glob import glob
from typing import List, Tuple

import cv2 as cv
import numpy as np
from tqdm import tqdm

from tools.motion_stablization.motion_stabilization_library import get_motion_stabilization
from tools.video_tubes.remove_noisy_false_positives_by_tracking import get_video_wise_list

boxes_json_folder = ''
frames_folder = ''
mask_folder = ''
output_folder = ''
output_filename_prefix = ''

patch_dimensions = 100
volume_dimensions = 224
foreground_pixel_value = 1
random_patch_dimensions = False

frames_in_volume = 8
middle_frame_number = frames_in_volume // 2


def generate_patch_coords_from_box(height, width, box, patch_dimensions) -> Tuple[int, int, int, int]:
    """
    Generates safe coords for the patch given box

    :param height: Max height of the frame
    :param width:  Max width of the frame
    :param box: (x, y, w, h)
    :param patch_dimensions: Dimension of the patch in either side
    :return: (y1, x1, y2, x2)
    """
    dimension_on_either_side = patch_dimensions // 2

    x1, y1, w, h = box

    # get center points
    cx = x1 + (w // 2)
    cy = y1 + (h // 2)

    # get patch coordinates
    p_x1 = cx - dimension_on_either_side
    p_x1 = p_x1 if p_x1 > 0 else 0

    p_y1 = cy - dimension_on_either_side
    p_y1 = p_y1 if p_y1 > 0 else 0

    p_x2 = cx + dimension_on_either_side
    p_x2 = p_x2 if p_x2 < width else width

    p_y2 = cy + dimension_on_either_side
    p_y2 = p_y2 if p_y2 < height else height

    return int(p_y1), int(p_x1), int(p_y2), int(p_x2)


def get_patches_by_tracker(tracker, frames_list: List[np.ndarray], patch_dimensions):
    patches = []
    for frame in frames_list:
        ret, tracker_box = tracker.update(frame)
        p_y1, p_x1, p_y2, p_x2 = generate_patch_coords_from_box(frame.shape[0], frame.shape[1],
                                                                tracker_box, patch_dimensions)
        patches.append(frame[p_y1:p_y2, p_x1:p_x2])
    return patches


def extract_small_boxes(boxes: list):
    new_boxes = []
    for box in boxes:
        new_boxes.append(box['box'])
    return new_boxes


def get_tracked_volumes(frames_list: List[np.ndarray], boxes: List[Tuple[int, int, int, int]], key_frame_number: int,
                        patch_dimensions):
    key_frame = frames_list[key_frame_number]

    # from frame 0 to less than key frame
    previous_frames = frames_list[0:key_frame_number]
    # reverse frames
    previous_frames = previous_frames[-1::-1]

    # from keyframe+1 to end
    next_frames = frames_list[key_frame_number + 1:]

    final_volumes = []
    # get boxes for key frames
    for box in boxes:
        y1, x1, y2, x2 = box
        w = x2 - x1
        h = y2 - y1

        if w < 10:
            w = 10
            x2 = x1 + w
        if h < 10:
            h = 10
            y2 = y1 + h

        tracker = cv.TrackerCSRT_create()
        try:
            tracker.init(key_frame, (x1, y1, w, h))
        except:
            print('Box ignored because tracker could not initialize')
            continue

        key_frame_patch_location = generate_patch_coords_from_box(key_frame.shape[0], key_frame.shape[1],
                                                                  (x1, y1, w, h), patch_dimensions)
        key_frame_patch = key_frame[key_frame_patch_location[0]:key_frame_patch_location[2],
                          key_frame_patch_location[1]:key_frame_patch_location[3]]

        previous_patches = get_patches_by_tracker(tracker, previous_frames, patch_dimensions)
        previous_patches = previous_patches[-1::-1]

        next_patches = get_patches_by_tracker(tracker, next_frames, patch_dimensions)

        final_volume = []
        final_volume.extend(previous_patches)
        final_volume.append(key_frame_patch)
        final_volume.extend(next_patches)

        final_volumes.append({
            'location': key_frame_patch_location,
            'volume': final_volume
        })
    return final_volumes


def main_process(params):
    video_frames = params['video_frames']
    frame_number = params['frame_number']
    output_volumes_folder = params['output_volumes_folder']
    output_masks_folder = params['output_masks_folder']
    output_patch_information_folder = params['output_patch_information_folder']

    middle_frame_path = video_frames[frame_number + middle_frame_number]

    filename = os.path.basename(middle_frame_path)
    filename_wo_ext = os.path.splitext(filename)[0]

    boxes_json_file_path = os.path.join(boxes_json_folder, f'{filename_wo_ext}.json')
    if os.path.exists(boxes_json_file_path):
        middle_mask_path = os.path.join(mask_folder, f'{filename_wo_ext}.png')
        middle_mask = cv.imread(middle_mask_path)

        with open(boxes_json_file_path, 'r') as f:
            boxes = json.load(f)
        boxes = extract_small_boxes(boxes)

        n_boxes = random.randint(1, 2)
        if len(boxes) > n_boxes:
            boxes = random.sample(boxes, n_boxes)

        frames_list = []
        for i in range(frame_number, frame_number + frames_in_volume + 1, 1):
            f_path = video_frames[i]
            frames_list.append(cv.imread(f_path))

        try:
            frames_list, inverse_matrices, stabilized_masks, gt_stabilized_masks = get_motion_stabilization(frames_list)
        except:
            print(f'Failed to stabilize {filename}')
            return ''

        if random_patch_dimensions:
            patch_dimensions = random.randint(100, 200)
        tracked_volumes = get_tracked_volumes(frames_list, boxes, middle_frame_number, patch_dimensions)
        for volume_number, tracked_volume in enumerate(tracked_volumes):
            tvl_y1, tvl_x1, tvl_y2, tvl_x2 = tracked_volume['location']

            volume = tracked_volume['volume']
            for i in range(len(volume)):
                volume[i] = cv.resize(volume[i], (volume_dimensions, volume_dimensions))
            volume = np.array(volume)

            output_filename = f'{output_filename_prefix}{filename_wo_ext}_{str(volume_number).zfill(3)}'
            volume_output_path = os.path.join(output_volumes_folder, output_filename + '.npy')
            np.save(volume_output_path, volume)

            mask_output_path = os.path.join(output_masks_folder, output_filename + '.png')
            middle_mask_patch = middle_mask[tvl_y1:tvl_y2, tvl_x1:tvl_x2]
            middle_mask_patch = cv.resize(middle_mask_patch, (volume_dimensions, volume_dimensions),
                                          interpolation=cv.INTER_NEAREST)
            middle_mask_patch[middle_mask_patch > 0] = foreground_pixel_value
            cv.imwrite(mask_output_path, middle_mask_patch)

            patch_info = {
                'filename': filename,
                'box': {
                    'y1': tvl_y1,
                    'x1': tvl_x1,
                    'y2': tvl_y2,
                    'x2': tvl_x2,
                }
            }
            patch_info_output_path = os.path.join(output_patch_information_folder, output_filename + '.txt')
            with open(patch_info_output_path, 'w') as f:
                json.dump(patch_info, f)


def main():
    pool = multiprocessing.Pool(os.cpu_count() - 1)

    output_volumes_folder = os.path.join(output_folder, 'volumes')
    output_masks_folder = os.path.join(output_folder, 'masks')
    output_patch_information_folder = os.path.join(output_folder, 'patch_information')

    os.makedirs(output_volumes_folder, exist_ok=True)
    os.makedirs(output_masks_folder, exist_ok=True)
    os.makedirs(output_patch_information_folder, exist_ok=True)

    frame_paths = glob(os.path.join(frames_folder, '*'))
    frame_paths.sort()
    video_wise_frame_paths = get_video_wise_list(frame_paths)
    for video_name in tqdm(video_wise_frame_paths.keys(), desc='Processing videos'):
        video_frames = video_wise_frame_paths[video_name]
        video_frames.sort()

        frame_numbers = list(range(0, len(video_frames) - frames_in_volume - 1, 1))

        params = []
        for frame_number in frame_numbers:
            params.append({
                'video_frames': video_frames,
                'frame_number': frame_number,
                'output_volumes_folder': output_volumes_folder,
                'output_masks_folder': output_masks_folder,
                'output_patch_information_folder': output_patch_information_folder,
            })

        for _ in tqdm(pool.imap_unordered(main_process, params), 'Processing frames', len(params)):
            pass


if __name__ == '__main__':
    main()
