import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from tools.crf.IoU_test import get_bounding_boxes

videos_folder = ''
detections_folder = ''

detections_output_folder = ''


def initialize_tracker(tracker, frame: np.ndarray, detection_path: str):
    if os.path.exists(detection_path):
        tracker = cv2.MultiTracker_create()

        detection_mask = cv2.imread(detection_path)
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


def get_video_frames(video_path: str):
    assert os.path.isfile(video_path), "Provided path should exist and it should be a video file"

    cap = cv2.VideoCapture(video_path)

    cur_frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        yield cap.get(cv2.CAP_PROP_FRAME_COUNT), cur_frame_number, frame

        cur_frame_number += 1
    cap.release()


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
                ret, boxes = tracker.update(frame)

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
                tracker = initialize_tracker(tracker, frame, detection_mask_path)

            pbar.total = total_frames
            pbar.update()
        pbar.close()


if __name__ == '__main__':
    main()
