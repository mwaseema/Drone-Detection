import os
from glob import glob
from os import path

import cv2 as cv
from tqdm import tqdm
import numpy as np

# folder containing videos
videos_folder = ''
# file extension of the videos
video_file_extension = '.mov'
# folder containing
annotations_folder = ''

# if save_frame = 4, this means every 4th frame is saved
# if save_frame = 1, every frame will be saved
save_frame = 4
# value which will be placed in mask where an object exists
foreground_in_mask = 255
# folder where to output all the frames
frames_output_folder = ''
masks_output_folder = ''


def parse_annotations_data(data_list):
    annotations = {}
    for d in data_list:
        d = d.replace(' ', '').split(',')
        frame_number = int(d[0])
        bb_coords = []
        bbs = [int(i) for i in d[2:]]
        for i in range(0, len(bbs), 4):
            bb_coords.append(bbs[i:i + 4])

        annotations[frame_number] = bb_coords
    return annotations


def get_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        data_list = [ln.strip().replace('\n', '') for ln in f.readlines()]
    annotations = parse_annotations_data(data_list)
    return annotations


def main():
    os.makedirs(frames_output_folder, exist_ok=True)
    os.makedirs(masks_output_folder, exist_ok=True)

    video_paths = glob(path.join(videos_folder, f'*{video_file_extension}'))

    for video_path in tqdm(video_paths, 'Processing videos'):
        filename = path.basename(video_path)
        filename_wo_ext, file_ext = path.splitext(filename)

        annotations_path = path.join(annotations_folder, f'{filename_wo_ext}.txt')
        annotations = get_annotations(annotations_path)

        video_cap = cv.VideoCapture(video_path)
        total_frames = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))

        frame_number = 0
        pbar = tqdm(desc='Processing frames', total=total_frames)
        while video_cap.isOpened():
            ret, frame = video_cap.read()

            if not ret:
                break

            if frame_number % save_frame == 0:
                frame_output_path = path.join(frames_output_folder,
                                              f'{filename_wo_ext}_{str(frame_number).zfill(6)}.png')
                cv.imwrite(frame_output_path, frame)

                mask = np.zeros(frame.shape, dtype=frame.dtype)
                if frame_number in annotations.keys():
                    bounding_boxes = annotations[frame_number]
                    for bb in bounding_boxes:
                        x1, y1, x2, y2 = bb
                        mask[y1:y2, x1:x2, :] = foreground_in_mask

                mask_output_path = path.join(masks_output_folder, f'{filename_wo_ext}_{str(frame_number).zfill(6)}.png')
                cv.imwrite(mask_output_path, mask)

            frame_number += 1
            pbar.update()
        pbar.close()
        video_cap.release()


if __name__ == '__main__':
    main()
