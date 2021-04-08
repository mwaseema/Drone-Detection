import multiprocessing
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from tools.motion_stablization.motion_stabilization_library import get_motion_stabilization, \
    apply_inverse_transformations
from tools.optical_flow_motion_boundaries import generate_optical_flow
from tools.video_tubes.remove_noisy_false_positives_by_tracking import get_video_wise_list

# folder containing all the video frames
frames_folder = ''
# folder containing reference files whose motion boundaries are required
# name of these files are used to generate motion boundaries of only those corresponding files
reference_frames_folder = ''

# folder where to output motion boundary files as image file
motion_boundaries_output_folder = ''
# visualization folder is optional
visualized_frames_output_folder = ''
# intensity above or equal to threshold value is set to max and the intensity lower than threshold is set to 0
motion_boundary_threshold_for_visualization = 30


def convert_frame_to_grayscale(frame: np.ndarray):
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def main_process(frame_path_pair):
    middle_frame_filename = os.path.basename(frame_path_pair[1])
    middle_frame_filename_wo_ext, _ = os.path.splitext(middle_frame_filename)

    frames = []
    for f_path in frame_path_pair:
        frame = cv2.imread(f_path)
        frames.append(frame)

    try:
        stabilized_frames, inverse_matrices, _, _ = get_motion_stabilization(frames)
        is_stabilized = True
    except:
        print(
            f'Could not calculate stabilization for f{middle_frame_filename_wo_ext}. Computing MB without stabilization.')
        is_stabilized = False
        stabilized_frames = frames[:-1]

    for j in range(len(stabilized_frames)):
        stabilized_frames[j] = convert_frame_to_grayscale(stabilized_frames[j])

    __motion_boundary = None

    for fr_n in range(len(stabilized_frames) - 1):
        __mb1 = generate_optical_flow(stabilized_frames[fr_n], stabilized_frames[fr_n + 1])
        __mb2 = generate_optical_flow(stabilized_frames[fr_n + 1], stabilized_frames[fr_n])
        __mb = np.maximum(__mb1, __mb2)

        if __motion_boundary is None:
            __motion_boundary = __mb.copy()
        else:
            __motion_boundary = np.maximum(__motion_boundary, __mb)

    mb = 255 * cv2.normalize(__motion_boundary, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mb = mb.astype(np.uint8)

    if is_stabilized:
        mb = apply_inverse_transformations(mb, inverse_matrices[1], mb.shape[0], mb.shape[1])

    motion_boundaries_output_path = os.path.join(motion_boundaries_output_folder,
                                                 f'{middle_frame_filename_wo_ext}.png')
    cv2.imwrite(motion_boundaries_output_path, mb)

    if visualized_frames_output_folder != '':
        mb[mb < motion_boundary_threshold_for_visualization] = 0
        mb[mb >= motion_boundary_threshold_for_visualization] = 255

        frame = frames[1]
        frame[:, :, 2] = mb

        os.makedirs(visualized_frames_output_folder, exist_ok=True)
        visualized_frames_output_path = os.path.join(visualized_frames_output_folder,
                                                     f'{middle_frame_filename_wo_ext}.jpg')
        cv2.imwrite(visualized_frames_output_path, frame)


def main():
    os.makedirs(motion_boundaries_output_folder, exist_ok=True)

    if visualized_frames_output_folder != '':
        os.makedirs(visualized_frames_output_folder, exist_ok=True)

    pool = multiprocessing.Pool(2)

    frame_paths = glob(os.path.join(frames_folder, '*'))
    video_wise_frame_paths = get_video_wise_list(frame_paths)
    for video_name in tqdm(video_wise_frame_paths.keys(), desc='Processing videos'):
        video_frames_paths = video_wise_frame_paths[video_name]
        video_frames_paths.sort()

        frame_path_pairs = []

        for frame_number in tqdm(list(range(0, len(video_frames_paths) - 4, 1)), 'Processing frames of video'):
            middle_frame_number = frame_number + 1

            middle_frame_path = video_frames_paths[middle_frame_number]
            middle_frame_filename = os.path.basename(middle_frame_path)
            middle_frame_filename_wo_ext, _ = os.path.splitext(middle_frame_filename)

            reference_frame_path = os.path.join(reference_frames_folder, f'{middle_frame_filename_wo_ext}.png')
            if os.path.exists(reference_frame_path):
                __frame_paths = []
                for i in range(frame_number, frame_number + 4):
                    __frame_paths.append(video_frames_paths[i])
                frame_path_pairs.append(__frame_paths)

        for _ in tqdm(pool.imap_unordered(main_process, frame_path_pairs), 'Processing frame pairs',
                      len(frame_path_pairs)):
            pass


if __name__ == '__main__':
    main()
