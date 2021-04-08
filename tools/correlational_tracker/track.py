import cv2
import sys
import os
import tools.correlational_tracker.tracking_lib as tracking_lib
from tqdm import tqdm

video_file_path = '/home/hec/waseem/UAV/data/Videos/train/Clip_042.mov'
video_filename = os.path.basename(video_file_path)
video_filename_without_ext, video_file_ext = os.path.splitext(video_filename)

detection_folder_path = '/home/hec/waseem/UAV/tracking/data/detections'

tracking_output_folder = '/home/hec/waseem/UAV/tracking/data/tracked'


def main():
    os.makedirs(tracking_output_folder, exist_ok=True)

    # Set up tracker.
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    video = cv2.VideoCapture(video_file_path)

    if not video.isOpened():
        print
        "Could not open video"
        sys.exit()

    total_video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=total_video_frames, desc="Processing video frames", dynamic_ncols=True, unit='frame')
    frame_count = -1
    tracker_initialized = False
    while True:
        pbar.update()
        frame_count += 1
        ok, frame = video.read()
        if not ok:
            break

        bboxes = tracking_lib.get_bounding_boxes(detection_folder_path, video_filename_without_ext, frame_count)

        if tracker_initialized == False:
            if bboxes is not None and len(bboxes) > 0:
                tracker.init(frame, bboxes[0])
                tracker_initialized = True

                bbox = bboxes[0]
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                tracked_output_path = os.path.join(tracking_output_folder,
                                                   tracking_lib.format_frame_number(frame_count) + '.jpg')
                cv2.imwrite(tracked_output_path, frame)
            continue
        else:
            if bboxes is not None and len(bboxes) > 0:
                tracker.init(frame, bboxes[0])

                bbox = bboxes[0]
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                tracked_output_path = os.path.join(tracking_output_folder,
                                                   tracking_lib.format_frame_number(frame_count) + '.jpg')
                cv2.imwrite(tracked_output_path, frame)
            continue

        timer = cv2.getTickCount()

        ok, bbox = tracker.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        tracked_output_path = os.path.join(tracking_output_folder,
                                           tracking_lib.format_frame_number(frame_count) + '.jpg')
        cv2.imwrite(tracked_output_path, frame)

    pbar.close()


if __name__ == '__main__':
    main()
