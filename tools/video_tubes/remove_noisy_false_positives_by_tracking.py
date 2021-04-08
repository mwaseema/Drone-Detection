import multiprocessing
import os
from glob import glob
from typing import List, Dict

import cv2
import numpy as np
from tqdm import tqdm

from tools.correlational_tracker.multi_tracker_with_merge_on_fly import calculate_distance
from tools.crf.IoU_test import get_region_props
from tools.crf.crf_on_labels import get_safe_extended_pixels

masks_folder = ''
# all of the video frames
frames_folder = ''

masks_output_folder = ''

# window will be on both previous and next side
window_of_search = 2
# should present in any backward or forward side
should_present_in_frames = 1


def get_index_of_frame(file_to_search: str, list_of_file_paths: List[str]):
    filename = os.path.basename(file_to_search)
    filename_wo_ext, _ = os.path.splitext(filename)

    for i in range(len(list_of_file_paths)):
        __filename = os.path.basename(list_of_file_paths[i])
        __filename_wo_ext, _ = os.path.splitext(__filename)

        if filename_wo_ext == __filename_wo_ext:
            return i
    return None


def is_box_present_in_mask(box, mask, distance_threshold):
    y1, x1, y2, x2 = box
    cx = x1 + (x2 - x1) / 2
    cy = y1 + (y2 - y1) / 2

    distances = []
    mask_rps = get_region_props(mask)
    for mask_rp in mask_rps:
        cy2, cx2 = mask_rp.centroid
        distance = calculate_distance(cy, cx, cy2, cx2)
        distances.append(distance)

    if len(distances) == 0:
        return False
    else:
        min_distance = min(distances)
        if min_distance <= distance_threshold:
            return True
        else:
            return False


def track_in_frames(init_frame: np.ndarray, init_box, tracking_frame_paths: List[str], masks_folder: str):
    y1, x1, y2, x2 = init_box
    w = x2 - x1
    h = y2 - y1

    tracker = cv2.TrackerCSRT_create()
    tracker.init(init_frame, (x1, y1, w, h))

    presence_count = 0

    for tracking_frame_path in tracking_frame_paths:
        tracking_frame = cv2.imread(tracking_frame_path)
        ok, bbox = tracker.update(tracking_frame)
        t_x1, t_y1, t_w, t_h = bbox
        t_x2 = t_x1 + t_w
        t_y2 = t_y1 + t_h

        t_x1 = int(t_x1)
        t_y1 = int(t_y1)
        t_x2 = int(t_x2)
        t_y2 = int(t_y2)

        filename = os.path.basename(tracking_frame_path)
        filename_wo_ext, _ = os.path.splitext(filename)

        mask_path = os.path.join(masks_folder, f'{filename_wo_ext}.png')
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
            is_box_present = is_box_present_in_mask((t_y1, t_x1, t_y2, t_x2), mask, 30)
            if is_box_present:
                presence_count += 1

    return presence_count


def main_process(params):
    frame_number = params['frame_number']
    video_masks = params['video_masks']
    frame_paths = params['frame_paths']

    left_frame_number = frame_number - window_of_search
    right_frame_number = frame_number + window_of_search

    left_mask_path = video_masks[left_frame_number]
    right_mask_path = video_masks[right_frame_number]
    middle_mask_path = video_masks[frame_number]

    left_frame_index = get_index_of_frame(left_mask_path, frame_paths)
    right_frame_index = get_index_of_frame(right_mask_path, frame_paths)
    middle_frame_index = get_index_of_frame(middle_mask_path, frame_paths)

    if left_frame_index is None or right_frame_index is None or middle_frame_index is None:
        print(
            f'Left, middle or right frame was not found {os.path.basename(middle_mask_path)}\nContinuing by neglecting it.')
        return ''

    # from left index to before middle index
    previous_frames = frame_paths[left_frame_index:middle_frame_index]
    previous_frames = previous_frames[-1::-1]

    # from next of middle index to right index
    next_frames = frame_paths[middle_frame_index + 1:right_frame_index + 1]

    middle_frame = cv2.imread(frame_paths[middle_frame_index])
    middle_mask = cv2.imread(middle_mask_path)
    middle_mask_rps = get_region_props(middle_mask)

    for middle_mask_rp in middle_mask_rps:
        y1, x1, y2, x2 = middle_mask_rp.bbox

        previous_frame_presence = track_in_frames(middle_frame, (y1, x1, y2, x2), previous_frames, masks_folder)
        next_frame_presence = track_in_frames(middle_frame, (y1, x1, y2, x2), next_frames, masks_folder)

        # if box is present in either previous or next frame for number of times
        if previous_frame_presence >= should_present_in_frames or next_frame_presence >= should_present_in_frames:
            is_tp = True
        else:
            is_tp = False

        if not is_tp:
            y1, x1, y2, x2 = get_safe_extended_pixels(middle_mask.shape[0], middle_mask.shape[1], 1, y1, x1, y2,
                                                      x2)
            middle_mask[y1:y2, x1:x2] = 0

    middle_mask_filename = os.path.basename(middle_mask_path)
    middle_mask_filename_wo_ext, _ = os.path.splitext(middle_mask_filename)
    mask_output_path = os.path.join(masks_output_folder, f'{middle_mask_filename_wo_ext}.png')
    cv2.imwrite(mask_output_path, middle_mask)


def get_video_wise_list(frames_list: List[str]):
    video_wise_frame_paths: Dict[str, List[str]] = {}
    for frame_path in tqdm(frames_list, desc="Making video wise list of frames"):
        # Clip_041_000052.png
        filename = os.path.basename(frame_path)
        # Clip_041
        video_name = filename[0:8]

        if video_name not in video_wise_frame_paths.keys():
            video_wise_frame_paths[video_name] = []

        video_wise_frame_paths[video_name].append(frame_path)
    return video_wise_frame_paths


def main():
    os.makedirs(masks_output_folder, exist_ok=True)

    pool = multiprocessing.Pool(os.cpu_count() - 2)

    frame_paths = glob(os.path.join(frames_folder, '*'))
    frame_paths.sort()

    mask_paths = glob(os.path.join(masks_folder, '*'))
    mask_paths.sort()

    video_wise_mask_paths = get_video_wise_list(mask_paths)
    for video_name in tqdm(video_wise_mask_paths.keys(), desc='Processing videos'):
        video_masks = video_wise_mask_paths[video_name]
        video_masks.sort()

        # for looping from window of search to n - window of search
        frame_numbers = list(range(window_of_search, len(video_masks) - window_of_search))

        params = []
        for frame_number in frame_numbers:
            params.append({
                'frame_number': frame_number,
                'video_masks': video_masks,
                'frame_paths': frame_paths,
            })

        for _ in tqdm(pool.imap_unordered(main_process, params), 'Processing frames', len(params)):
            pass


if __name__ == '__main__':
    main()
