import json
import os
from glob import glob
from math import sqrt

import cv2
import numpy as np
from tqdm import tqdm

from tools.correlational_tracker.multi_tracker import get_video_frames
from tools.crf.IoU_test import get_bounding_boxes, get_region_props, bb_intersection_over_union

videos_folder = ''
detections_folder = ''

detections_output_folder = ''

tracker_initialization_counter = 0


def calculate_distance(y1, x1, y2, x2):
    val = pow(x2 - x1, 2) + pow(y2 - y1, 2)
    val = sqrt(val)
    return val


def write_log_file(log_file_path: str, filename, y1, x1, y2, x2):
    with open(log_file_path, 'a') as f:
        d = {
            'filename': filename,
            'y1': y1,
            'x1': x1,
            'y2': y2,
            'x2': x2,
        }
        d = json.dumps(d)
        f.write(d)
        f.write("\n")


def merge_boxes_using_distance(mask1: np.ndarray, mask2: np.ndarray, distance_threshold=30,
                               log_file_path=None, filename=None) -> np.ndarray:
    mask1 = mask1.copy()
    mask2 = mask2.copy()

    if len(mask1.shape) == 3 and mask1.shape[2] == 3:
        mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
    if len(mask2.shape) == 3 and mask2.shape[2] == 3:
        mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

    mask1_region_props = get_region_props(mask1)
    mask2_region_props = get_region_props(mask2)

    new_mask = mask1.copy()

    if len(mask1_region_props) == 0:
        for mask2_region_prop in mask2_region_props:
            y1, x1, y2, x2 = mask2_region_prop.bbox
            new_mask[y1:y2, x1:x2] = 255

            if log_file_path is not None and filename is not None:
                write_log_file(log_file_path, filename, y1, x1, y2, x2)
    else:
        distance_matrix = np.zeros((len(mask2_region_props), len(mask1_region_props)), dtype=np.float)

        for mask2_region_prop_index, mask2_region_prop in enumerate(mask2_region_props):
            y2, x2 = mask2_region_prop.centroid

            for mask1_region_prop_index, mask1_region_prop in enumerate(mask1_region_props):
                y1, x1 = mask1_region_prop.centroid

                distance = calculate_distance(y1, x1, y2, x2)
                distance_matrix[mask2_region_prop_index, mask1_region_prop_index] = distance

        for mask2_region_prop_index, box_distances in enumerate(distance_matrix):
            # minimum distance between current box with the boxes mask 1
            min_distance = float(np.min(box_distances))

            if min_distance > distance_threshold:
                region_prop = mask2_region_props[mask2_region_prop_index]
                y1, x1, y2, x2 = region_prop.bbox
                new_mask[y1:y2, x1:x2] = 255

                if log_file_path is not None and filename is not None:
                    write_log_file(log_file_path, filename, y1, x1, y2, x2)
    return new_mask


def merge_boxes_using_iou(mask1: np.ndarray, mask2: np.ndarray, iou_threshold=0.1) -> np.ndarray:
    mask1 = mask1.copy()
    mask2 = mask2.copy()

    if len(mask1.shape) == 3 and mask1.shape[2] == 3:
        mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
    if len(mask2.shape) == 3 and mask2.shape[2] == 3:
        mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

    mask1_region_props = get_region_props(mask1)
    mask2_region_props = get_region_props(mask2)

    new_mask = mask1.copy()

    if len(mask1_region_props) == 0:
        for mask2_region_prop in mask2_region_props:
            y1, x1, y2, x2 = mask2_region_prop.bbox
            new_mask[y1:y2, x1:x2] = 255
    else:
        iou_matrix = np.zeros((len(mask2_region_props), len(mask1_region_props)), dtype=np.float)

        for mask2_region_prop_index, mask2_region_prop in enumerate(mask2_region_props):
            mask2_y1, mask2_x1, mask2_y2, mask2_x2 = mask2_region_prop.bbox

            for mask1_region_prop_index, mask1_region_prop in enumerate(mask1_region_props):
                mask1_y1, mask1_x1, mask1_y2, mask1_x2 = mask1_region_prop.bbox

                box1 = (mask1_x1, mask1_y1, mask1_x2, mask1_y2)
                box2 = (mask2_x1, mask2_y1, mask2_x2, mask2_y2)
                iou = bb_intersection_over_union(box1, box2)
                iou_matrix[mask2_region_prop_index, mask1_region_prop_index] = iou

        for mask2_region_prop_index, iou_values in enumerate(iou_matrix):
            # maximum iou between current box with the boxes of mask 1
            max_iou = float(np.max(iou_values))

            if max_iou < iou_threshold:
                region_prop = mask2_region_props[mask2_region_prop_index]
                y1, x1, y2, x2 = region_prop.bbox
                new_mask[y1:y2, x1:x2] = 255
    return new_mask


def initialize_tracker(tracker, frame: np.ndarray, detection_path: str, tracked_mask: np.ndarray = None):
    global tracker_initialization_counter

    if os.path.exists(detection_path):
        tracker = cv2.MultiTracker_create()

        detection_mask = cv2.imread(detection_path)

        # if tracked mask is give, merge detection mask and tracked mask before initializing
        # if tracker is initialized on merged detections for less than equal to 10 times
        if tracked_mask is not None and tracker_initialization_counter <= 5:
            # detection_mask = merge_boxes_using_distance(detection_mask, tracked_mask, 30)
            detection_mask = merge_boxes_using_iou(detection_mask, tracked_mask, 0.1)
            tracker_initialization_counter += 1
        else:
            tracker_initialization_counter = 0

        boxes = get_bounding_boxes(detection_mask)

        new_boxes = []
        for box in boxes:
            x1 = box['x1']
            y1 = box['y1']
            x2 = box['x2']
            y2 = box['y2']
            new_boxes.append((x1, y1, x2 - x1, y2 - y1))

        for box in new_boxes:
            if box[2] > 2 and box[3] > 2:
                tracker.add(cv2.TrackerCSRT_create(), frame, box)

    return tracker


def main():
    os.makedirs(detections_output_folder, exist_ok=True)

    video_paths = glob(os.path.join(videos_folder, '*'))

    for video_path in tqdm(video_paths, desc='Processing videos'):
        video_filename = os.path.basename(video_path)
        video_filename_wo_ext = os.path.splitext(video_filename)[0]

        tracker = None

        pbar = tqdm(desc="Processing video frames", unit='frame')
        for total_frames, cur_frame_number, frame in get_video_frames(video_path):
            frame_filename = f"{video_filename_wo_ext}_{str(cur_frame_number).zfill(6)}.png"
            detection_mask_path = os.path.join(detections_folder, frame_filename)

            # if tracker is not already initialized
            if tracker is None:
                tracker = initialize_tracker(tracker, frame, detection_mask_path)
            # if tracker is already initialized
            else:
                # new mask for storing tracked boxes
                new_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

                # update tracker with current frame and get new tracked boxes
                try:
                    ret, boxes = tracker.update(frame)
                except:
                    print(f"Could not track {frame_filename}")
                    # reinitialize tracker
                    tracker = initialize_tracker(tracker, frame, detection_mask_path)
                    continue

                # adding tracked boxes to new mask
                for box in boxes:
                    box = [int(round(b)) for b in box]
                    box_x1, box_y1, box_w, box_h = box
                    box_x2 = box_x1 + box_w
                    box_y2 = box_y1 + box_h
                    new_mask[box_y1:box_y2, box_x1: box_x2] = 255

                detection_mask_output_path = os.path.join(detections_output_folder, frame_filename)
                cv2.imwrite(detection_mask_output_path, new_mask)

                # reinitialize tracker
                tracker = initialize_tracker(tracker, frame, detection_mask_path, new_mask)

            pbar.total = total_frames
            pbar.update()
        pbar.close()


if __name__ == '__main__':
    main()
