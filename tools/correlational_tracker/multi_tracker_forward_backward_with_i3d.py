import os
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from tools.correlational_tracker.multi_tracker_with_merge_on_fly import initialize_tracker
from tools.correlational_tracker.multi_tracker_with_merge_on_fly import merge_boxes_using_distance
# from i3d_inception_net.testing.testing_with_volumes_on_the_fly import get_tracked_volumes
from tools.crf.IoU_test import get_bounding_boxes
from tools.video_tubes.remove_noisy_false_positives_by_tracking import get_video_wise_list

network_checkpoint = ''
# Folder containing all the video frames
FRAMES_FOLDER = ''
# Folder containing detection masks
DETECTION_MASKS_FOLDER = ''
# Folder where to output tracker output
TRACKER_OUTPUT_FOLDER = ''

frames_in_volume = 8
keyframe_number = frames_in_volume // 2

patch_dimensions = 100


def generate_patch_coords_from_box(height, width, box, patch_dimensions) -> Tuple[int, int, int, int]:
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


def get_patches_by_tracker(tracker, frames_list: List[np.ndarray]):
    patches = []
    for frame in frames_list:
        ret, tracker_box = tracker.update(frame)
        p_y1, p_x1, p_y2, p_x2 = generate_patch_coords_from_box(frame.shape[0], frame.shape[1],
                                                                tracker_box, patch_dimensions)
        patches.append(frame[p_y1:p_y2, p_x1:p_x2])
    return patches


def get_tracked_volumes(frames_list: List[np.ndarray], masks_list: List[np.ndarray], key_frame_number: int):
    key_frame = frames_list[key_frame_number]
    key_mask = masks_list[key_frame_number]
    key_mask[key_mask > 0] = 255

    # from frame 0 to less than key frame
    previous_frames = frames_list[0:key_frame_number]
    # reverse frames
    previous_frames = previous_frames[-1::-1]

    # from keyframe+1 to end
    next_frames = frames_list[key_frame_number + 1:]

    final_volumes = []
    # get boxes for key frames
    boxes = get_bounding_boxes(key_mask)
    for box in boxes:
        y1 = box['y1']
        y2 = box['y2']
        x1 = box['x1']
        x2 = box['x2']
        w = x2 - x1
        h = y2 - y1

        if w * h < 4 and (w <= 2 or h <= 2):
            print('Box ignored due to small size')
            continue

        tracker = cv2.TrackerCSRT_create()
        try:
            tracker.init(key_frame, (x1, y1, w, h))
        except:
            print('Box ignored because tracker could not initialize')
            continue

        key_frame_patch_location = generate_patch_coords_from_box(key_frame.shape[0], key_frame.shape[1],
                                                                  (x1, y1, w, h), patch_dimensions)
        key_frame_patch = key_frame[key_frame_patch_location[0]:key_frame_patch_location[2],
                          key_frame_patch_location[1]:key_frame_patch_location[3]]

        previous_patches = get_patches_by_tracker(tracker, previous_frames)
        previous_patches = previous_patches[-1::-1]

        next_patches = get_patches_by_tracker(tracker, next_frames)

        final_volume = []
        final_volume.extend(previous_patches)
        final_volume.append(key_frame_patch)
        final_volume.extend(next_patches)

        final_volumes.append({
            'location': key_frame_patch_location,
            'volume': final_volume
        })
    return final_volumes


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


def track_frames(frames_list: List[str], detection_masks_folder: str, output_folder: str, model,
                 reverse_track: bool = False):
    frames_list = frames_list.copy()
    if reverse_track:
        frames_list.sort(reverse=True)
        description = 'Reverse tracking objects in frames'
    else:
        frames_list.sort()
        description = 'Tracking objects in frames'

    tracker = None
    for frame_number in tqdm(list(range(0, len(frames_list) - 8)), desc=description, unit='frame'):
        frame_path = frames_list[frame_number]
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
                    plt.figure('Previous tracked mask')
                    plt.imshow(previous_tracked_mask)
                    plt.figure('tracked mask')
                    plt.imshow(tracked_mask)
                    tracked_mask = merge_boxes_using_distance(previous_tracked_mask, tracked_mask, 200)
                    plt.figure('tracked and previous merged')

            # get 8 frames
            __frames_list = []
            __masks_list = []
            for i in range(frames_in_volume):
                f_path = frames_list[frame_number + i]
                __frames_list.append(cv2.imread(f_path))
                f_filename = os.path.basename(f_path)
                m_path = os.path.join(detection_masks_folder, f_filename[0:-4] + '.png')
                __masks_list.append(cv2.imread(m_path))
            __masks_list[keyframe_number] = tracked_mask

            try:
                tracked_volumes = get_tracked_volumes(__frames_list, __masks_list, keyframe_number)
            except:
                print("Couldn't update trackr")
                continue

            for tracked_volume in tracked_volumes:
                tv_location = tracked_volume['location']
                tv_volume = tracked_volume['volume']

                for tvi in range(len(tv_volume)):
                    tv_volume[tvi] = cv2.resize(tv_volume[tvi], (224, 224))

                tv_volume = np.array(tv_volume)

                segmentation_output, _ = model.predict_segmentation(
                    input_img_frames_patch_volume=tv_volume.copy(),
                    threshold=0.2,
                )
                segmentation_output = cv2.cvtColor(segmentation_output.astype(np.uint8), cv2.COLOR_BGR2GRAY)

                p_y1, p_x1, p_y2, p_x2 = tv_location
                p_w = p_x2 - p_x1
                p_h = p_y2 - p_y1

                plt.figure('Tracked mask 2')
                plt.imshow(tracked_mask[p_y1:p_y2, p_x1:p_x2])

                if np.max(segmentation_output) > 0:
                    segmentation_output = cv2.resize(segmentation_output, (p_w, p_h), interpolation=cv2.INTER_NEAREST)

                    tracked_mask[p_y1:p_y2, p_x1:p_x2] = segmentation_output

                    plt.figure('sementation output')
                    plt.imshow(segmentation_output)
                else:
                    tracked_mask[p_y1:p_y2, p_x1:p_x2] = 0

            cv2.imwrite(output_path, tracked_mask)
            tracker = initialize_tracker(tracker, frame, detection_path, tracked_mask)

            plt.show()
            plt.close('all')


def main():
    import keras_segmentation

    os.makedirs(TRACKER_OUTPUT_FOLDER, exist_ok=True)

    frame_paths = glob(os.path.join(FRAMES_FOLDER, '*'))
    frame_paths.sort()

    model = keras_segmentation.predict.model_from_checkpoint_given_path(network_checkpoint)

    video_wise_frame_paths = get_video_wise_list(frame_paths)
    for video_name in tqdm(video_wise_frame_paths.keys(), desc='Processing videos'):
        video_frames = video_wise_frame_paths[video_name]
        video_frames.sort()

        # forward tracking
        track_frames(frame_paths, DETECTION_MASKS_FOLDER, TRACKER_OUTPUT_FOLDER, model, False)
        # reverse tracking
        track_frames(frame_paths, DETECTION_MASKS_FOLDER, TRACKER_OUTPUT_FOLDER, model, True)


if __name__ == '__main__':
    main()
