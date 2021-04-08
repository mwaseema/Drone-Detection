import os
from glob import glob
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from tools.correlational_tracker.multi_tracker import initialize_tracker
# from correlational_tracker.multi_tracker_with_merge_on_fly import initialize_tracker
from tools.correlational_tracker.multi_tracker_with_merge_on_fly import merge_boxes_using_distance
# Folder containing all the video frames
from tools.video_tubes.remove_noisy_false_positives_by_tracking import get_video_wise_list

FRAMES_FOLDER = ''
# Folder containing detection masks
DETECTION_MASKS_FOLDER = ''
# Folder where to output tracker output
TRACKER_OUTPUT_FOLDER = ''


def get_tracked_mask(tracker, frame):
    ret, boxes = tracker.update(frame)

    new_mask = np.zeros(frame.shape[0:2], dtype=np.uint8)
    # adding tracked boxes to new mask
    for box in boxes:
        box = [int(round(b)) for b in box]
        box_x1, box_y1, box_w, box_h = box
        box_x2 = box_x1 + box_w
        box_y2 = box_y1 + box_h
        new_mask[box_y1:box_y2, box_x1: box_x2] = 255
    return new_mask


def track_frames(frames_list: List[str], detection_masks_folder: str, output_folder: str, reverse_track: bool = False):
    frames_list = frames_list.copy()
    if reverse_track:
        frames_list.sort(reverse=True)
        description = 'Reverse tracking objects in frames'
    else:
        frames_list.sort()
        description = 'Tracking objects in frames'

    tracker = None
    for frame_path in tqdm(frames_list, desc=description, unit='frame'):
        frame_filename = os.path.basename(frame_path)
        frame_filename = os.path.splitext(frame_filename)[0] + '.png'

        frame = cv2.imread(frame_path)
        detection_path = os.path.join(detection_masks_folder, frame_filename)

        if tracker is None:
            try:
                tracker = initialize_tracker(tracker, frame, detection_path)
            except:
                print('Code 0: Initialization failed')
                continue
        else:
            # get tracked predictions in frame if tracker is already initialized
            try:
                tracked_mask = get_tracked_mask(tracker, frame)
            except:
                print('Code 1: Tracking failed')
                tracker = initialize_tracker(tracker, frame, detection_path)
                continue
            output_path = os.path.join(output_folder, frame_filename)

            # if tracking in reverse order
            if reverse_track:
                # if previous tracked output exists
                if os.path.exists(output_path):
                    previous_tracked_mask = cv2.imread(output_path)
                    tracked_mask = merge_boxes_using_distance(previous_tracked_mask, tracked_mask, 100)

            cv2.imwrite(output_path, tracked_mask)
            tracker = initialize_tracker(tracker, frame, detection_path)
            # tracker = initialize_tracker(tracker, frame, detection_path, tracked_mask)


def main():
    os.makedirs(TRACKER_OUTPUT_FOLDER, exist_ok=True)

    frame_paths = glob(os.path.join(FRAMES_FOLDER, '*'))
    frame_paths.sort()

    video_wise_frame_paths = get_video_wise_list(frame_paths)
    for video_name in tqdm(video_wise_frame_paths.keys(), desc='Processing videos'):
        video_frames = video_wise_frame_paths[video_name]
        video_frames.sort()

        # forward tracking
        track_frames(frame_paths, DETECTION_MASKS_FOLDER, TRACKER_OUTPUT_FOLDER, False)
        # reverse tracking
        track_frames(frame_paths, DETECTION_MASKS_FOLDER, TRACKER_OUTPUT_FOLDER, True)


if __name__ == '__main__':
    main()
